"""
fit_series.py — Fit distributions to the full monthly time series per station.

Two fitting modes, both using monthly mean river levels as data points:

  global   For each station, fit all 9 distributions to the complete
           1980–present monthly series (~540 values).  Identifies which
           distribution best characterises each station overall.

  rolling  For each station, fit all 9 distributions inside a sliding
           window that advances year by year across the 45-year period.
           Produces one set of fitted parameters per (station, window),
           allowing temporal evolution of location, scale, and shape to
           be tracked.  Default: 10-year window, 1-year step.

With ~540 data points the AIC/BIC parameter penalty is negligible compared to
fit quality, which is why Johnson SU is expected to dominate — its extra
flexibility earns a large log-likelihood gain that easily offsets 2 extra
parameters.

Outputs
-------
    outputs/fitted_params/global/<station>.parquet
    outputs/fitted_params/rolling/<station>.parquet
    outputs/model_scores/global_scores.parquet
    outputs/model_scores/rolling_scores.parquet

Usage
-----
    python src/fit_series.py                        # both modes, all stations
    python src/fit_series.py --mode global          # global fits only
    python src/fit_series.py --mode rolling         # rolling fits only
    python src/fit_series.py --window-years 10      # rolling window size (default 10)
    python src/fit_series.py --step-years 1         # window step (default 1)
    python src/fit_series.py --station 51107        # single station (smoke-test)
    python src/fit_series.py --workers 4
    python src/fit_series.py --force

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from scipy.optimize import OptimizeWarning
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
DB_PATH    = PROC_DIR / "river_data.duckdb"

GLOBAL_DIR        = ROOT / "outputs" / "fitted_params" / "global"
ROLLING_DIR       = ROOT / "outputs" / "fitted_params" / "rolling"
DAILY_GLOBAL_DIR  = ROOT / "outputs" / "fitted_params" / "daily_global"
DAILY_ROLLING_DIR = ROOT / "outputs" / "fitted_params" / "daily_rolling"
SCORES_DIR        = ROOT / "outputs" / "model_scores"

for d in (GLOBAL_DIR, ROLLING_DIR, DAILY_GLOBAL_DIR, DAILY_ROLLING_DIR, SCORES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Distribution registry (identical to distributions.py) ─────────────────────
DISTRIBUTIONS: dict[str, tuple] = {
    "johnson_su":   (stats.johnsonsu,   4),
    "gen_logistic": (stats.genlogistic, 3),
    "pearson3":     (stats.pearson3,    3),
    "normal":       (stats.norm,        2),
    "lognormal":    (stats.lognorm,     3),
    "gamma":        (stats.gamma,       3),
    "weibull":      (stats.weibull_min, 3),
    "gev":          (stats.genextreme,  3),
    "gumbel":       (stats.gumbel_r,    2),
}

MIN_OBS       = 24    # minimum monthly values required to attempt a fit
MIN_DAILY_OBS = 365  # minimum daily values required (at least one full year)

# ── Schemas ────────────────────────────────────────────────────────────────────
_GLOBAL_SCHEMA = pa.schema([
    pa.field("station_reference", pa.string()),
    pa.field("distribution",      pa.string()),
    pa.field("n_obs",             pa.int16()),
    pa.field("p1",                pa.float64()),
    pa.field("p2",                pa.float64()),
    pa.field("p3",                pa.float64()),
    pa.field("p4",                pa.float64()),
    pa.field("log_likelihood",    pa.float64()),
    pa.field("aic",               pa.float64()),
    pa.field("bic",               pa.float64()),
    pa.field("ks_statistic",      pa.float64()),
    pa.field("ks_pvalue",         pa.float64()),
    pa.field("converged",         pa.bool_()),
])

_ROLLING_SCHEMA = pa.schema([
    pa.field("station_reference", pa.string()),
    pa.field("window_start_year", pa.int16()),
    pa.field("window_end_year",   pa.int16()),
    pa.field("window_mid_year",   pa.float32()),  # for plotting on a continuous axis
    pa.field("distribution",      pa.string()),
    pa.field("n_obs",             pa.int16()),
    pa.field("p1",                pa.float64()),
    pa.field("p2",                pa.float64()),
    pa.field("p3",                pa.float64()),
    pa.field("p4",                pa.float64()),
    pa.field("log_likelihood",    pa.float64()),
    pa.field("aic",               pa.float64()),
    pa.field("bic",               pa.float64()),
    pa.field("ks_statistic",      pa.float64()),
    pa.field("ks_pvalue",         pa.float64()),
    pa.field("converged",         pa.bool_()),
])

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "fit_series.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ── Fit one distribution ───────────────────────────────────────────────────────

class FitResult(NamedTuple):
    params:         tuple
    log_likelihood: float
    aic:            float
    bic:            float
    ks_statistic:   float
    ks_pvalue:      float
    converged:      bool


def fit_one(dist_class, data: np.ndarray, n_params: int) -> FitResult:
    """Fit one distribution via MLE; suppress numerical noise from optimiser."""
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            params = dist_class.fit(data)
        except Exception:
            return FitResult((np.nan,) * 4, np.nan, np.nan, np.nan, np.nan, np.nan, False)

        if any(issubclass(w.category, OptimizeWarning) for w in caught):
            converged = False

    log_like = np.sum(dist_class.logpdf(data, *params))
    if not np.isfinite(log_like):
        padded = tuple(params) + (np.nan,) * (4 - len(params))
        return FitResult(padded, np.nan, np.nan, np.nan, np.nan, np.nan, False)

    n   = len(data)
    aic = 2 * n_params - 2 * log_like
    bic = n_params * np.log(n) - 2 * log_like
    ks_stat, ks_p = stats.kstest(data, dist_class.cdf, args=params)
    padded = tuple(params) + (np.nan,) * (4 - len(params))
    return FitResult(padded, log_like, aic, bic, ks_stat, ks_p, converged)


def _fit_all_dists(data: np.ndarray) -> dict[str, FitResult]:
    """Fit all 9 distributions to one data vector."""
    return {
        name: fit_one(dist_class, data, n_params)
        for name, (dist_class, n_params) in DISTRIBUTIONS.items()
    }


# ── Per-station workers ────────────────────────────────────────────────────────

def _fit_global(
    station_reference: str,
    db_path:           str,
    out_dir:           str,
    force:             bool,
) -> tuple[str, str]:
    """
    Fit all 9 distributions to the complete monthly series for one station.
    One row per distribution → 9 rows total per station.
    """
    out_path = Path(out_dir) / f"{station_reference}.parquet"
    if out_path.exists() and not force:
        return station_reference, "skipped"

    try:
        con = duckdb.connect(db_path, read_only=True)
        df  = con.execute("""
            SELECT mean_level_m
            FROM monthly_series
            WHERE station_reference = ?
              AND year >= 1980
              AND mean_level_m IS NOT NULL
            ORDER BY year, month
        """, [station_reference]).df()
        con.close()

        data = df["mean_level_m"].to_numpy(dtype=np.float64)
        if len(data) < MIN_OBS:
            return station_reference, "ok:too_few"

        rows = []
        for dist_name, res in _fit_all_dists(data).items():
            rows.append({
                "station_reference": station_reference,
                "distribution":      dist_name,
                "n_obs":             np.int16(len(data)),
                "p1": res.params[0], "p2": res.params[1],
                "p3": res.params[2], "p4": res.params[3],
                "log_likelihood": res.log_likelihood,
                "aic": res.aic, "bic": res.bic,
                "ks_statistic": res.ks_statistic,
                "ks_pvalue":    res.ks_pvalue,
                "converged":    res.converged,
            })

        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(rows), schema=_GLOBAL_SCHEMA,
                                 preserve_index=False),
            out_path,
        )
        return station_reference, f"ok:{len(data)}"

    except Exception as exc:
        return station_reference, f"error:{exc}"


def _fit_rolling(
    station_reference: str,
    db_path:           str,
    out_dir:           str,
    window_years:      int,
    step_years:        int,
    force:             bool,
) -> tuple[str, str]:
    """
    Fit all 9 distributions inside a sliding window for one station.
    One row per (distribution, window) → 9 × n_windows rows per station.
    """
    out_path = Path(out_dir) / f"{station_reference}.parquet"
    if out_path.exists() and not force:
        return station_reference, "skipped"

    try:
        con = duckdb.connect(db_path, read_only=True)
        df  = con.execute("""
            SELECT year, month, mean_level_m
            FROM monthly_series
            WHERE station_reference = ?
              AND year >= 1980
              AND mean_level_m IS NOT NULL
            ORDER BY year, month
        """, [station_reference]).df()
        con.close()

        if df.empty:
            return station_reference, "ok:0"

        # Build a year-indexed array for fast windowing
        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

        rows = []
        # Slide the window start from min_year to (max_year - window_years)
        for w_start in range(min_year, max_year - window_years + 2, step_years):
            w_end = w_start + window_years - 1
            if w_end > max_year:
                break

            mask = (df["year"] >= w_start) & (df["year"] <= w_end)
            data = df.loc[mask, "mean_level_m"].dropna().to_numpy(dtype=np.float64)

            if len(data) < MIN_OBS:
                continue

            w_mid = round((w_start + w_end) / 2, 1)

            for dist_name, res in _fit_all_dists(data).items():
                rows.append({
                    "station_reference": station_reference,
                    "window_start_year": np.int16(w_start),
                    "window_end_year":   np.int16(w_end),
                    "window_mid_year":   np.float32(w_mid),
                    "distribution":      dist_name,
                    "n_obs":             np.int16(len(data)),
                    "p1": res.params[0], "p2": res.params[1],
                    "p3": res.params[2], "p4": res.params[3],
                    "log_likelihood": res.log_likelihood,
                    "aic": res.aic, "bic": res.bic,
                    "ks_statistic": res.ks_statistic,
                    "ks_pvalue":    res.ks_pvalue,
                    "converged":    res.converged,
                })

        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(rows), schema=_ROLLING_SCHEMA,
                                 preserve_index=False),
            out_path,
        )
        n_windows = len(rows) // len(DISTRIBUTIONS) if rows else 0
        return station_reference, f"ok:{n_windows}_windows"

    except Exception as exc:
        return station_reference, f"error:{exc}"


# ── Daily workers ─────────────────────────────────────────────────────────────

def _fit_daily_global(
    station_reference: str,
    db_path:           str,
    out_dir:           str,
    force:             bool,
) -> tuple[str, str]:
    """
    Fit all 9 distributions to the complete daily-average series for one station.
    ~16,000 data points; no monthly aggregation.  One row per distribution (9 total).
    """
    out_path = Path(out_dir) / f"{station_reference}.parquet"
    if out_path.exists() and not force:
        return station_reference, "skipped"

    try:
        con = duckdb.connect(db_path, read_only=True)
        df  = con.execute("""
            SELECT mean_level_m
            FROM daily_averages da
            INNER JOIN cohort c USING (station_reference)
            WHERE da.station_reference = ?
              AND YEAR(date) >= 1980
              AND mean_level_m IS NOT NULL
            ORDER BY date
        """, [station_reference]).df()
        con.close()

        data = df["mean_level_m"].to_numpy(dtype=np.float64)
        if len(data) < MIN_DAILY_OBS:
            return station_reference, "ok:too_few"

        rows = []
        for dist_name, res in _fit_all_dists(data).items():
            rows.append({
                "station_reference": station_reference,
                "distribution":      dist_name,
                "n_obs":             np.int16(min(len(data), 32_767)),
                "p1": res.params[0], "p2": res.params[1],
                "p3": res.params[2], "p4": res.params[3],
                "log_likelihood": res.log_likelihood,
                "aic": res.aic, "bic": res.bic,
                "ks_statistic": res.ks_statistic,
                "ks_pvalue":    res.ks_pvalue,
                "converged":    res.converged,
            })

        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(rows), schema=_GLOBAL_SCHEMA,
                                 preserve_index=False),
            out_path,
        )
        return station_reference, f"ok:{len(data)}"

    except Exception as exc:
        return station_reference, f"error:{exc}"


def _fit_daily_rolling(
    station_reference: str,
    db_path:           str,
    out_dir:           str,
    window_years:      int,
    step_years:        int,
    force:             bool,
) -> tuple[str, str]:
    """
    Fit all 9 distributions inside a sliding window of daily averages.
    A 10-year window contains ~3,650 data points — enough for Johnson SU's
    extra parameters to demonstrate genuine fit quality gains over simpler
    distributions.
    """
    out_path = Path(out_dir) / f"{station_reference}.parquet"
    if out_path.exists() and not force:
        return station_reference, "skipped"

    try:
        con = duckdb.connect(db_path, read_only=True)
        df  = con.execute("""
            SELECT YEAR(date) AS year, mean_level_m
            FROM daily_averages da
            INNER JOIN cohort c USING (station_reference)
            WHERE da.station_reference = ?
              AND YEAR(date) >= 1980
              AND mean_level_m IS NOT NULL
            ORDER BY date
        """, [station_reference]).df()
        con.close()

        if df.empty:
            return station_reference, "ok:0"

        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

        rows = []
        for w_start in range(min_year, max_year - window_years + 2, step_years):
            w_end = w_start + window_years - 1
            if w_end > max_year:
                break

            data = df.loc[
                (df["year"] >= w_start) & (df["year"] <= w_end),
                "mean_level_m"
            ].dropna().to_numpy(dtype=np.float64)

            if len(data) < MIN_DAILY_OBS:
                continue

            w_mid = round((w_start + w_end) / 2, 1)

            for dist_name, res in _fit_all_dists(data).items():
                rows.append({
                    "station_reference": station_reference,
                    "window_start_year": np.int16(w_start),
                    "window_end_year":   np.int16(w_end),
                    "window_mid_year":   np.float32(w_mid),
                    "distribution":      dist_name,
                    "n_obs":             np.int16(min(len(data), 32_767)),
                    "p1": res.params[0], "p2": res.params[1],
                    "p3": res.params[2], "p4": res.params[3],
                    "log_likelihood": res.log_likelihood,
                    "aic": res.aic, "bic": res.bic,
                    "ks_statistic": res.ks_statistic,
                    "ks_pvalue":    res.ks_pvalue,
                    "converged":    res.converged,
                })

        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(rows), schema=_ROLLING_SCHEMA,
                                 preserve_index=False),
            out_path,
        )
        n_windows = len(rows) // len(DISTRIBUTIONS) if rows else 0
        return station_reference, f"ok:{n_windows}_windows"

    except Exception as exc:
        return station_reference, f"error:{exc}"


# ── Score merging ──────────────────────────────────────────────────────────────

def build_scores(mode: str, force: bool = False) -> None:
    """Merge per-station Parquets into a single scores file."""
    src_dir = {
        "global":        GLOBAL_DIR,
        "rolling":       ROLLING_DIR,
        "daily_global":  DAILY_GLOBAL_DIR,
        "daily_rolling": DAILY_ROLLING_DIR,
    }[mode]
    out_path = SCORES_DIR / f"{mode}_scores.parquet"
    label    = f"{mode}_scores"

    if out_path.exists() and not force:
        log.info("%s already exists — skipping (use --force).", label)
        return

    glob = str(src_dir / "*.parquet")
    log.info("Merging %s from %s …", label, glob)
    t0 = time.perf_counter()

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{glob}', union_by_name = true)
            WHERE converged = true
        )
        TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    log.info("%s written  (%s)", label, _elapsed(t0))


# ── Orchestrator ───────────────────────────────────────────────────────────────

def _run_mode(
    mode:      str,
    stations:  list[str],
    db_path:   str,
    workers:   int,
    force:     bool,
    window_years: int = 10,
    step_years:   int = 1,
) -> None:
    out_dir = str({
        "global":        GLOBAL_DIR,
        "rolling":       ROLLING_DIR,
        "daily_global":  DAILY_GLOBAL_DIR,
        "daily_rolling": DAILY_ROLLING_DIR,
    }[mode])

    with ProcessPoolExecutor(max_workers=workers) as pool:
        if mode == "global":
            futures = {
                pool.submit(_fit_global, stn, db_path, out_dir, force): stn
                for stn in stations
            }
        elif mode == "rolling":
            futures = {
                pool.submit(_fit_rolling, stn, db_path, out_dir,
                            window_years, step_years, force): stn
                for stn in stations
            }
        elif mode == "daily_global":
            futures = {
                pool.submit(_fit_daily_global, stn, db_path, out_dir, force): stn
                for stn in stations
            }
        else:  # daily_rolling
            futures = {
                pool.submit(_fit_daily_rolling, stn, db_path, out_dir,
                            window_years, step_years, force): stn
                for stn in stations
            }

        ok = skipped = errors = 0
        with tqdm(total=len(futures), desc=f"Fitting [{mode}]", unit=" stn") as pbar:
            for future in as_completed(futures):
                ref, status = future.result()
                if status.startswith("ok"):
                    ok += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1
                    log.warning("%-12s  %s", ref, status)
                pbar.set_postfix_str(status[:35])
                pbar.update(1)

    log.info("[%s] ok=%d  skipped=%d  errors=%d", mode, ok, skipped, errors)
    build_scores(mode, force=force)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",
                   choices=["global", "rolling", "both",
                            "daily_global", "daily_rolling", "daily_both"],
                   default="both")
    p.add_argument("--window-years", type=int, default=10,
                   help="Rolling window size in years (default 10)")
    p.add_argument("--step-years",   type=int, default=1,
                   help="Window advance step in years (default 1)")
    p.add_argument("--station",      default=None)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--force",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()

    con     = duckdb.connect(str(DB_PATH), read_only=True)
    cohort  = con.execute(
        "SELECT station_reference FROM cohort ORDER BY station_reference"
    ).df()
    con.close()

    if args.station:
        cohort = cohort[cohort["station_reference"] == args.station]
        if cohort.empty:
            log.error("Station '%s' not in cohort.", args.station)
            sys.exit(1)

    stations = cohort["station_reference"].tolist()
    log.info(
        "Mode=%s  stations=%d  workers=%d  window=%dy  step=%dy",
        args.mode, len(stations), args.workers, args.window_years, args.step_years,
    )

    if args.mode in ("global", "both"):
        _run_mode("global", stations, str(DB_PATH),
                  args.workers, args.force)

    if args.mode in ("rolling", "both"):
        _run_mode("rolling", stations, str(DB_PATH),
                  args.workers, args.force,
                  window_years=args.window_years,
                  step_years=args.step_years)

    if args.mode in ("daily_global", "daily_both"):
        _run_mode("daily_global", stations, str(DB_PATH),
                  args.workers, args.force)

    if args.mode in ("daily_rolling", "daily_both"):
        _run_mode("daily_rolling", stations, str(DB_PATH),
                  args.workers, args.force,
                  window_years=args.window_years,
                  step_years=args.step_years)

    log.info("fit_series.py complete  (%s)", _elapsed(t0))


if __name__ == "__main__":
    main()
