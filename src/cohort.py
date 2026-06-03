"""
cohort.py — Select the stable 371-station cohort for Phase 1 analysis.

A station enters the cohort if it passes all four criteria:
    1. started_by_1980   : has monthly data in or before 1980
    2. active_to_recent  : has monthly data up to a minimum recent year
    3. year_coverage     : data present in ≥ X% of calendar years 1980–last_year
    4. month_coverage    : data present in ≥ X% of months from 1980-01–last_month

Defaults are tuned to produce ~371 stations.  Adjust with --min-year-coverage
and --min-month-coverage to explore the sensitivity of cohort size.

Reads  : data/processed/river_data.duckdb  (monthly_series table)
Writes : data/processed/river_data.duckdb
           └─ station_coverage  : per-station coverage statistics
           └─ cohort            : the final cohort station list + metadata
         data/processed/cohort.parquet
         data/processed/station_coverage.parquet

Usage
-----
    python src/cohort.py                         # build cohort with defaults
    python src/cohort.py --report                # show funnel only, don't write
    python src/cohort.py --min-year-coverage 0.8 --min-month-coverage 0.65
    python src/cohort.py --force                 # rebuild even if tables exist

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
DB_PATH  = PROC_DIR / "river_data.duckdb"

# ── Default thresholds ─────────────────────────────────────────────────────────
DEFAULT_ACTIVE_TO_YEAR    = 2020   # station must still have data by this year
DEFAULT_MIN_YEAR_COVERAGE = 0.85   # ≥ 85 % of calendar years must have data
DEFAULT_MIN_MONTH_COVERAGE= 0.70   # ≥ 70 % of months from 1980-01 must have data

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "cohort.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [name],
    ).fetchone()[0] > 0


# ── Coverage statistics ────────────────────────────────────────────────────────

def build_station_coverage(con: duckdb.DuckDBPyConnection, force: bool) -> None:
    """
    Compute per-station coverage statistics from monthly_series.
    Writes the station_coverage table.
    """
    if _table_exists(con, "station_coverage") and not force:
        log.info("station_coverage already exists — skipping (use --force to rebuild).")
        return

    log.info("Computing station coverage statistics …")
    t0 = time.perf_counter()

    con.execute("DROP TABLE IF EXISTS station_coverage")
    con.execute("""
        CREATE TABLE station_coverage AS
        WITH base AS (
            SELECT
                station_reference,
                MIN(year)                               AS first_year,
                MAX(year)                               AS last_year,
                COUNT(*)                                AS n_months_total,
                COUNT(DISTINCT year)                    AS n_years_with_data,
                -- calendar-year span from first_year to last_year
                (MAX(year) - MIN(year) + 1)             AS n_years_span,
                -- months possible in the study window 1980-01 → last (year, month)
                (MAX(year) - 1980) * 12 + MAX(month)   AS n_months_possible,
                -- months actually present from 1980 onwards
                SUM(CASE WHEN year >= 1980 THEN 1 ELSE 0 END)
                                                        AS n_months_from_1980,
                -- years in 1980–last_year that have ≥ 1 month of data
                COUNT(DISTINCT year)
                    FILTER (WHERE year >= 1980)         AS n_years_from_1980,
                (MAX(year) - 1980 + 1)                  AS n_years_possible
            FROM monthly_series
            GROUP BY station_reference
        )
        SELECT
            *,
            -- fraction of calendar years (1980–last_year) that have any data
            ROUND(
                n_years_from_1980::DOUBLE / NULLIF(n_years_possible, 0),
                4
            )                                           AS year_coverage_frac,
            -- fraction of months (1980-01–last month) that are present
            ROUND(
                n_months_from_1980::DOUBLE / NULLIF(n_months_possible, 0),
                4
            )                                           AS month_coverage_frac
        FROM base
        ORDER BY station_reference
    """)

    n = con.execute("SELECT COUNT(*) FROM station_coverage").fetchone()[0]
    log.info("station_coverage: %d stations  (%s)", n, _elapsed(t0))


# ── Funnel report ──────────────────────────────────────────────────────────────

def print_funnel(
    con:                duckdb.DuckDBPyConnection,
    active_to_year:     int,
    min_year_coverage:  float,
    min_month_coverage: float,
) -> int:
    """
    Print a step-by-step funnel showing how many stations survive each filter.
    Returns the final cohort count.
    """
    total = con.execute("SELECT COUNT(*) FROM station_coverage").fetchone()[0]

    c1 = con.execute(
        "SELECT COUNT(*) FROM station_coverage WHERE first_year <= 1980"
    ).fetchone()[0]

    c2 = con.execute(
        "SELECT COUNT(*) FROM station_coverage "
        "WHERE first_year <= 1980 AND last_year >= ?",
        [active_to_year],
    ).fetchone()[0]

    c3 = con.execute(
        "SELECT COUNT(*) FROM station_coverage "
        "WHERE first_year <= 1980 AND last_year >= ? "
        "  AND year_coverage_frac  >= ?",
        [active_to_year, min_year_coverage],
    ).fetchone()[0]

    c4 = con.execute(
        "SELECT COUNT(*) FROM station_coverage "
        "WHERE first_year <= 1980 AND last_year >= ? "
        "  AND year_coverage_frac  >= ? "
        "  AND month_coverage_frac >= ?",
        [active_to_year, min_year_coverage, min_month_coverage],
    ).fetchone()[0]

    col_w = 55
    print()
    print("  Cohort selection funnel")
    print("  " + "─" * 62)
    print(f"  {'All stations in monthly_series':<{col_w}} {total:>5}")
    print(f"  {'  1. first_year ≤ 1980':<{col_w}} {c1:>5}  ({100*c1/total:.1f}%)")
    print(f"  {'  2. last_year  ≥ ' + str(active_to_year):<{col_w}} {c2:>5}  ({100*c2/total:.1f}%)")
    print(f"  {'  3. year_coverage_frac  ≥ ' + str(min_year_coverage):<{col_w}} {c3:>5}  ({100*c3/total:.1f}%)")
    print(f"  {'  4. month_coverage_frac ≥ ' + str(min_month_coverage):<{col_w}} {c4:>5}  ({100*c4/total:.1f}%)")
    print("  " + "─" * 62)
    print(f"  {'  Final cohort':<{col_w}} {c4:>5}")
    print()

    return c4


# ── Build cohort ───────────────────────────────────────────────────────────────

def build_cohort(
    con:                duckdb.DuckDBPyConnection,
    active_to_year:     int,
    min_year_coverage:  float,
    min_month_coverage: float,
    force:              bool,
) -> None:
    """
    Apply all four filters and write the cohort table + Parquet.
    """
    if _table_exists(con, "cohort") and not force:
        n = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
        log.info("cohort table already exists (%d stations) — skipping (use --force).", n)
        return

    log.info("Writing cohort table …")
    con.execute("DROP TABLE IF EXISTS cohort")
    con.execute("""
        CREATE TABLE cohort AS
        SELECT
            sc.*,
            sm.label,
            sm.lat,
            sm.long,
            sm.river_name,
            sm.date_opened,
            sm.wiski_id,
            sm.nrfa_station_id,
            sm.easting,
            sm.northing
        FROM station_coverage sc
        LEFT JOIN read_parquet($stations_path) sm
               ON sc.station_reference = sm.station_reference
        WHERE sc.first_year         <= 1980
          AND sc.last_year          >= $active_to_year
          AND sc.year_coverage_frac  >= $min_year_coverage
          AND sc.month_coverage_frac >= $min_month_coverage
        ORDER BY sc.station_reference
    """, {
        "stations_path":      str(ROOT / "data" / "raw" / "stations.parquet"),
        "active_to_year":     active_to_year,
        "min_year_coverage":  min_year_coverage,
        "min_month_coverage": min_month_coverage,
    })

    n = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    log.info("cohort table written: %d stations", n)

    # Parquet exports
    cov_path    = PROC_DIR / "station_coverage.parquet"
    cohort_path = PROC_DIR / "cohort.parquet"
    con.execute(f"COPY station_coverage TO '{cov_path}'   (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY cohort           TO '{cohort_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    log.info("Exported → %s", cov_path)
    log.info("Exported → %s", cohort_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--active-to-year",     type=int,   default=DEFAULT_ACTIVE_TO_YEAR)
    p.add_argument("--min-year-coverage",  type=float, default=DEFAULT_MIN_YEAR_COVERAGE)
    p.add_argument("--min-month-coverage", type=float, default=DEFAULT_MIN_MONTH_COVERAGE)
    p.add_argument("--report", action="store_true",
                   help="Print funnel only — do not write tables or Parquet")
    p.add_argument("--force", action="store_true",
                   help="Rebuild tables even if they already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()

    log.info("Opening DuckDB: %s", DB_PATH)
    con = duckdb.connect(str(DB_PATH))

    build_station_coverage(con, force=args.force)

    n_cohort = print_funnel(
        con,
        active_to_year=args.active_to_year,
        min_year_coverage=args.min_year_coverage,
        min_month_coverage=args.min_month_coverage,
    )

    if args.report:
        log.info("--report mode: no tables written.")
        con.close()
        return

    build_cohort(
        con,
        active_to_year=args.active_to_year,
        min_year_coverage=args.min_year_coverage,
        min_month_coverage=args.min_month_coverage,
        force=args.force,
    )

    con.close()
    log.info("cohort.py complete — %d stations  (%s)", n_cohort, _elapsed(t0))


if __name__ == "__main__":
    main()
