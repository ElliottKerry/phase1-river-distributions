"""
preprocess.py — Clean raw 15-min readings and resample to daily averages.

Reads  : data/raw/<stationReference>.parquet  (all stations, via glob)
Writes : data/processed/river_data.duckdb
           └─ raw_readings     : verbatim union of all raw files
           └─ clean_readings   : QC-filtered, datetime cast to TIMESTAMP
           └─ daily_averages   : one row per (station, date), ≥75% completeness
           └─ monthly_series   : one row per (station, year, month)
           └─ qc_report        : per-station counts of removed records

Also exports:
    data/processed/daily_averages.parquet
    data/processed/monthly_series.parquet   ← input for distributions.py
    data/processed/qc_report.parquet

Usage
-----
    python src/preprocess.py                 # full run
    python src/preprocess.py --force         # drop and rebuild all tables
    python src/preprocess.py --station 51107 # single station (smoke-test)

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import duckdb

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH    = PROC_DIR / "river_data.duckdb"

# ── Thresholds ─────────────────────────────────────────────────────────────────
# A day requires at least this many 15-min readings to count as valid.
# 96 readings/day × 0.75 = 72.  Chosen to tolerate brief sensor gaps
# without accepting days that are mostly interpolated or missing.
MIN_READINGS_PER_DAY = 72   # 75 % of 96

# Quality labels from the EA API that are treated as invalid
BAD_QUALITY = ("Bad", "Rejected", "Estimated")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "preprocess.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    result = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_name = ?", [name]
    ).fetchone()
    return result[0] > 0


# ── Build steps ────────────────────────────────────────────────────────────────

def build_raw_readings(con: duckdb.DuckDBPyConnection, station_filter: str | None) -> None:
    """
    Union all per-station Parquet files into a single raw_readings table.
    Adds a 'ts' column (TIMESTAMP) cast from the raw 'datetime' string.
    """
    glob = str(RAW_DIR / "*.parquet")
    where = ""
    if station_filter:
        where = f"WHERE station_reference = '{station_filter}'"

    log.info("Loading raw Parquet files …  (glob: %s)", glob)
    t0 = time.perf_counter()

    con.execute("DROP TABLE IF EXISTS raw_readings")
    con.execute(f"""
        CREATE TABLE raw_readings AS
        SELECT
            station_reference,
            datetime                                    AS datetime_str,
            TRY_CAST(datetime AS TIMESTAMP)             AS ts,
            value,
            quality
        FROM read_parquet('{glob}', union_by_name = true)
        {where}
    """)

    n = con.execute("SELECT COUNT(*) FROM raw_readings").fetchone()[0]
    log.info("raw_readings: {:,} rows  ({})".format(n, _elapsed(t0)))


def build_clean_readings(con: duckdb.DuckDBPyConnection) -> None:
    """
    Apply QC filters and write clean_readings.

    Removes:
      - rows where ts could not be parsed (malformed datetime)
      - non-finite or NULL values
      - negative depths (physically impossible for a stage gauge)
      - readings with explicitly bad quality flags

    Also writes a qc_report table summarising removed counts per station.
    """
    bad_q = ", ".join(f"'{q}'" for q in BAD_QUALITY)

    log.info("Applying QC filters …")
    t0 = time.perf_counter()

    con.execute("DROP TABLE IF EXISTS clean_readings")
    con.execute(f"""
        CREATE TABLE clean_readings AS
        SELECT
            station_reference,
            ts,
            CAST(ts AS DATE)  AS date,
            value             AS level_m,
            quality
        FROM raw_readings
        WHERE
            ts          IS NOT NULL
            AND value   IS NOT NULL
            AND isfinite(value)
            AND value   >= 0
            AND quality NOT IN ({bad_q})
    """)

    n_clean = con.execute("SELECT COUNT(*) FROM clean_readings").fetchone()[0]
    n_raw   = con.execute("SELECT COUNT(*) FROM raw_readings").fetchone()[0]
    n_removed = n_raw - n_clean
    pct = 100 * n_removed / n_raw if n_raw else 0
    log.info(
        "clean_readings: {:,} rows retained  ({:,} removed, {:.2f}%)  ({})".format(
            n_clean, n_removed, pct, _elapsed(t0)
        )
    )

    # Per-station QC report
    con.execute("DROP TABLE IF EXISTS qc_report")
    con.execute(f"""
        CREATE TABLE qc_report AS
        SELECT
            r.station_reference,
            COUNT(*)                                                  AS n_raw,
            SUM(CASE WHEN r.ts IS NULL               THEN 1 ELSE 0 END) AS n_bad_datetime,
            SUM(CASE WHEN r.value IS NULL
                      OR NOT isfinite(r.value)        THEN 1 ELSE 0 END) AS n_non_finite,
            SUM(CASE WHEN r.value < 0                THEN 1 ELSE 0 END) AS n_negative,
            SUM(CASE WHEN r.quality IN ({bad_q})     THEN 1 ELSE 0 END) AS n_bad_quality,
            COUNT(*) - COUNT(c.ts)                                    AS n_total_removed
        FROM raw_readings r
        LEFT JOIN clean_readings c
               ON r.station_reference = c.station_reference
              AND r.ts                = c.ts
        GROUP BY r.station_reference
    """)


def build_daily_averages(con: duckdb.DuckDBPyConnection) -> None:
    """
    Aggregate clean 15-min readings to daily means.

    A day is included only if it has ≥ MIN_READINGS_PER_DAY valid readings
    (75 % of 96).  This tolerates brief gaps without accepting days that are
    mostly missing or gap-filled.

    Columns: station_reference, date, mean_level_m, n_readings,
             min_level_m, max_level_m, range_m
    """
    log.info("Aggregating to daily averages (min %d readings/day) …", MIN_READINGS_PER_DAY)
    t0 = time.perf_counter()

    con.execute("DROP TABLE IF EXISTS daily_averages")
    con.execute(f"""
        CREATE TABLE daily_averages AS
        SELECT
            station_reference,
            date,
            AVG(level_m)                      AS mean_level_m,
            COUNT(*)                           AS n_readings,
            MIN(level_m)                       AS min_level_m,
            MAX(level_m)                       AS max_level_m,
            MAX(level_m) - MIN(level_m)        AS range_m,
            STDDEV_SAMP(level_m)               AS std_level_m
        FROM clean_readings
        GROUP BY station_reference, date
        HAVING COUNT(*) >= {MIN_READINGS_PER_DAY}
        ORDER BY station_reference, date
    """)

    n = con.execute("SELECT COUNT(*) FROM daily_averages").fetchone()[0]
    n_stn = con.execute(
        "SELECT COUNT(DISTINCT station_reference) FROM daily_averages"
    ).fetchone()[0]
    log.info(
        "daily_averages: {:,} station-days across {:,} stations  ({})".format(
            n, n_stn, _elapsed(t0)
        )
    )


def build_monthly_series(con: duckdb.DuckDBPyConnection) -> None:
    """
    Collapse daily averages to monthly series.

    A month is included only if it has ≥ 20 valid daily observations
    (~ 65 % of days — tolerates short months and brief gaps while
    ensuring there is enough data to fit a distribution reliably).

    Columns: station_reference, year, month, n_days,
             mean_level_m, std_level_m, skewness, min_level_m, max_level_m,
             p5_m, p25_m, p50_m, p75_m, p95_m
    These per-month summary statistics are stored alongside the list of daily
    values used by distributions.py for MLE fitting.
    """
    log.info("Building monthly series (min 20 days/month) …")
    t0 = time.perf_counter()

    con.execute("DROP TABLE IF EXISTS monthly_series")
    con.execute("""
        CREATE TABLE monthly_series AS
        SELECT
            station_reference,
            YEAR(date)                              AS year,
            MONTH(date)                             AS month,
            COUNT(*)                                AS n_days,
            AVG(mean_level_m)                       AS mean_level_m,
            STDDEV_SAMP(mean_level_m)               AS std_level_m,
            -- skewness: 3*(mean-median)/std  (Pearson's second coefficient)
            -- approximate; full MLE skewness computed in distributions.py
            CASE
                WHEN STDDEV_SAMP(mean_level_m) > 0
                THEN 3.0 * (AVG(mean_level_m) - MEDIAN(mean_level_m))
                         / STDDEV_SAMP(mean_level_m)
                ELSE NULL
            END                                     AS skewness_approx,
            MIN(mean_level_m)                       AS min_level_m,
            MAX(mean_level_m)                       AS max_level_m,
            PERCENTILE_CONT(0.05) WITHIN GROUP
                (ORDER BY mean_level_m)             AS p5_m,
            PERCENTILE_CONT(0.25) WITHIN GROUP
                (ORDER BY mean_level_m)             AS p25_m,
            PERCENTILE_CONT(0.50) WITHIN GROUP
                (ORDER BY mean_level_m)             AS p50_m,
            PERCENTILE_CONT(0.75) WITHIN GROUP
                (ORDER BY mean_level_m)             AS p75_m,
            PERCENTILE_CONT(0.95) WITHIN GROUP
                (ORDER BY mean_level_m)             AS p95_m
        FROM daily_averages
        WHERE YEAR(date) >= 1980
        GROUP BY station_reference, YEAR(date), MONTH(date)
        HAVING COUNT(*) >= 20
        ORDER BY station_reference, year, month
    """)

    n = con.execute("SELECT COUNT(*) FROM monthly_series").fetchone()[0]
    n_stn = con.execute(
        "SELECT COUNT(DISTINCT station_reference) FROM monthly_series"
    ).fetchone()[0]
    log.info(
        "monthly_series: {:,} station-months across {:,} stations  ({})".format(
            n, n_stn, _elapsed(t0)
        )
    )


def export_parquet(con: duckdb.DuckDBPyConnection) -> None:
    """Export key tables to Parquet for portability and open deposit."""
    exports = {
        "daily_averages":  PROC_DIR / "daily_averages.parquet",
        "monthly_series":  PROC_DIR / "monthly_series.parquet",
        "qc_report":       PROC_DIR / "qc_report.parquet",
    }
    for table, path in exports.items():
        log.info("Exporting %s → %s", table, path)
        con.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    log.info("Parquet exports complete.")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force",   action="store_true",
                   help="Drop and rebuild all tables even if they already exist")
    p.add_argument("--station", default=None,
                   help="Process only this stationReference (smoke-test)")
    p.add_argument("--no-export", action="store_true",
                   help="Skip Parquet export step")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.perf_counter()

    log.info("Opening DuckDB: %s", DB_PATH)
    con = duckdb.connect(str(DB_PATH))

    # Set DuckDB to use all available cores for parallel Parquet reads
    con.execute(f"SET threads TO {os.cpu_count() or 4}")
    con.execute("SET memory_limit = '8GB'")

    steps = [
        ("raw_readings",    build_raw_readings),
        ("clean_readings",  build_clean_readings),
        ("daily_averages",  build_daily_averages),
        ("monthly_series",  build_monthly_series),
    ]

    for table_name, fn in steps:
        if _table_exists(con, table_name) and not args.force:
            log.info("Table '%s' already exists — skipping (use --force to rebuild).",
                     table_name)
            continue
        if table_name == "raw_readings":
            fn(con, args.station)
        else:
            fn(con)

    if not args.no_export:
        export_parquet(con)

    con.close()
    log.info("preprocess.py complete  (total: %s)", _elapsed(t_total))


if __name__ == "__main__":
    main()
