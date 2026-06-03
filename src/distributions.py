"""
distributions.py — Fit 9 candidate distributions to each cohort station-month.

For every (station, year, month) in the study window, the daily mean river
levels for that month are extracted from daily_averages and nine distributions
are fitted via maximum likelihood estimation (MLE).  Model fit is assessed with
AIC, BIC, and the Kolmogorov–Smirnov statistic.

Distributions fitted
--------------------
    johnson_su      scipy.stats.johnsonsu    4 params  (a, b, loc, scale)
    gen_logistic    scipy.stats.genlogistic  3 params  (c, loc, scale)
    pearson3        scipy.stats.pearson3     3 params  (skew, loc, scale)
    normal          scipy.stats.norm         2 params  (loc, scale)
    lognormal       scipy.stats.lognorm      3 params  (s, loc, scale)
    gamma           scipy.stats.gamma        3 params  (a, loc, scale)
    weibull         scipy.stats.weibull_min  3 params  (c, loc, scale)
    gev             scipy.stats.genextreme   3 params  (c, loc, scale)
    gumbel          scipy.stats.gumbel_r     2 params  (loc, scale)

Outputs
-------
    outputs/fitted_params/<station_reference>.parquet   (one per station)
    outputs/model_scores/all_scores.parquet             (merged AIC/BIC/KS table)

Each per-station Parquet has columns:
    station_reference, year, month, distribution, n_obs,
    p1..p4 (fitted params), log_likelihood, aic, bic,
    ks_statistic, ks_pvalue, converged

Usage
-----
    python src/distributions.py                   # full cohort
    python src/distributions.py --station 51107   # single station (smoke-test)
    python src/distributions.py --workers 6       # parallel workers (default 4)
    python src/distributions.py --force           # refit already-done stations
    python src/distributions.py --scores-only     # skip fitting; rebuild scores

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import os
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
ROOT         = Path(__file__).resolve().parent.parent
PROC_DIR     = ROOT / "data" / "processed"
PARAMS_DIR   = ROOT / "outputs" / "fitted_params"
SCORES_DIR   = ROOT / "outputs" / "model_scores"
DB_PATH      = PROC_DIR / "river_data.duckdb"

PARAMS_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# ── Distribution registry ──────────────────────────────────────────────────────
# Each entry: name → (scipy rv_continuous class, number of shape+loc+scale params)
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

# Minimum daily observations needed to attempt a fit.
# 20 days matches the preprocess.py threshold for monthly_series inclusion.
MIN_OBS = 20

# ── Output schema ──────────────────────────────────────────────────────────────
_SCHEMA = pa.schema([
    pa.field("station_reference", pa.string()),
    pa.field("year",              pa.int16()),
    pa.field("month",             pa.int8()),
    pa.field("distribution",      pa.string()),
    pa.field("n_obs",             pa.int16()),
    pa.field("p1",                pa.float64()),   # param 1 (shape / a / c / skew / s)
    pa.field("p2",                pa.float64()),   # param 2 (shape / b / loc)
    pa.field("p3",                pa.float64()),   # loc or scale
    pa.field("p4",                pa.float64()),   # scale (4-param dists only)
    pa.field("log_likelihood",    pa.float64()),
    pa.field("aic",               pa.float64()),
    pa.field("bic",               pa.float64()),
    pa.field("ks_statistic",      pa.float64()),
    pa.field("ks_pvalue",         pa.float64()),
    pa.field("converged",         pa.bool_()),
])

# ── Logging (main process only) ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "distributions.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ── Single-distribution fit ────────────────────────────────────────────────────

class FitResult(NamedTuple):
    params:         tuple
    log_likelihood: float
    aic:            float
    bic:            float
    ks_statistic:   float
    ks_pvalue:      float
    converged:      bool


def fit_one(dist_class, data: np.ndarray, n_params: int) -> FitResult:
    """
    Fit a single scipy distribution to data via MLE.
    Returns a FitResult; converged=False only if scipy raises OptimizeWarning
    (genuine failure to find an optimum) or the log-likelihood is non-finite.

    RuntimeWarnings (e.g. overflow in exp during gen_logistic fitting) are
    suppressed — they are numerical noise from the optimiser probing extreme
    parameter values and do not indicate a bad final fit.
    """
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Silence RuntimeWarning from scipy internals (overflow in exp etc.)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            params = dist_class.fit(data)
        except Exception:
            # Fitting completely failed — return nulls
            nan4 = (np.nan,) * 4
            return FitResult(nan4, np.nan, np.nan, np.nan, np.nan, np.nan, False)

        # Only OptimizeWarning means scipy couldn't find a genuine optimum
        if any(issubclass(w.category, OptimizeWarning) for w in caught):
            converged = False

    log_like = np.sum(dist_class.logpdf(data, *params))
    if not np.isfinite(log_like):
        return FitResult(
            tuple(params) + (np.nan,) * (4 - len(params)),
            np.nan, np.nan, np.nan, np.nan, np.nan, False,
        )

    n   = len(data)
    aic = 2 * n_params - 2 * log_like
    bic = n_params * np.log(n) - 2 * log_like

    ks_stat, ks_p = stats.kstest(data, dist_class.cdf, args=params)

    # Pad params to length 4 for uniform storage
    padded = tuple(params) + (np.nan,) * (4 - len(params))

    return FitResult(padded, log_like, aic, bic, ks_stat, ks_p, converged)


# ── Per-station worker ─────────────────────────────────────────────────────────

def _fit_station(
    station_reference: str,
    db_path:           str,
    out_dir:           str,
    force:             bool,
) -> tuple[str, str]:
    """
    Fit all 9 distributions to every month for one station.
    Designed to run in a subprocess (no shared state).

    Returns (station_reference, status):
        'skipped'       — output file exists and force=False
        'ok:<n_months>' — completed successfully
        'error:<msg>'   — exception raised
    """
    out_path = Path(out_dir) / f"{station_reference}.parquet"

    if out_path.exists() and not force:
        return station_reference, "skipped"

    try:
        con = duckdb.connect(db_path, read_only=True)

        # Load all daily values for this station from 1980, indexed by (year, month)
        df = con.execute("""
            SELECT
                YEAR(date)  AS year,
                MONTH(date) AS month,
                mean_level_m
            FROM daily_averages da
            INNER JOIN cohort c USING (station_reference)
            WHERE da.station_reference = ?
              AND YEAR(date) >= 1980
            ORDER BY date
        """, [station_reference]).df()
        con.close()

        if df.empty:
            return station_reference, "ok:0"

        rows: list[dict] = []

        for (year, month), group in df.groupby(["year", "month"], sort=True):
            data = group["mean_level_m"].dropna().to_numpy(dtype=np.float64)
            if len(data) < MIN_OBS:
                continue

            for dist_name, (dist_class, n_params) in DISTRIBUTIONS.items():
                res = fit_one(dist_class, data, n_params)
                rows.append({
                    "station_reference": station_reference,
                    "year":              np.int16(year),
                    "month":             np.int8(month),
                    "distribution":      dist_name,
                    "n_obs":             np.int16(len(data)),
                    "p1":                res.params[0],
                    "p2":                res.params[1],
                    "p3":                res.params[2],
                    "p4":                res.params[3],
                    "log_likelihood":    res.log_likelihood,
                    "aic":               res.aic,
                    "bic":               res.bic,
                    "ks_statistic":      res.ks_statistic,
                    "ks_pvalue":         res.ks_pvalue,
                    "converged":         res.converged,
                })

        result_df = pd.DataFrame(rows)
        pq.write_table(
            pa.Table.from_pandas(result_df, schema=_SCHEMA, preserve_index=False),
            out_path,
        )
        return station_reference, f"ok:{result_df['year'].nunique() * 12}"

    except Exception as exc:  # noqa: BLE001
        return station_reference, f"error:{exc}"


# ── Merge scores ───────────────────────────────────────────────────────────────

def build_scores(force: bool = False) -> None:
    """
    Merge all per-station fitted_params Parquets into a single
    outputs/model_scores/all_scores.parquet, keeping only the columns
    needed for model selection (AIC, BIC, KS).
    """
    scores_path = SCORES_DIR / "all_scores.parquet"
    if scores_path.exists() and not force:
        log.info("all_scores.parquet already exists — skipping (use --force).")
        return

    parquet_glob = str(PARAMS_DIR / "*.parquet")
    log.info("Merging scores from %s …", parquet_glob)
    t0 = time.perf_counter()

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                station_reference, year, month, distribution,
                n_obs, aic, bic, ks_statistic, ks_pvalue, converged
            FROM read_parquet('{parquet_glob}', union_by_name = true)
            WHERE converged = true
            ORDER BY station_reference, year, month, aic
        )
        TO '{scores_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()

    log.info("all_scores.parquet written  (%s)", _elapsed(t0))


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station",     default=None, help="Single stationReference (smoke-test)")
    p.add_argument("--workers",     type=int, default=4, help="Parallel worker processes")
    p.add_argument("--force",       action="store_true", help="Refit even if file exists")
    p.add_argument("--scores-only", action="store_true", help="Skip fitting; rebuild scores only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()

    if args.scores_only:
        build_scores(force=True)
        return

    # Load cohort station list
    con = duckdb.connect(str(DB_PATH), read_only=True)
    cohort = con.execute("SELECT station_reference FROM cohort ORDER BY station_reference").df()
    con.close()

    if args.station:
        cohort = cohort[cohort["station_reference"] == args.station]
        if cohort.empty:
            log.error("Station '%s' not in cohort.", args.station)
            sys.exit(1)

    stations = cohort["station_reference"].tolist()
    log.info(
        "Fitting %d distributions × %d stations  (workers=%d)",
        len(DISTRIBUTIONS), len(stations), args.workers,
    )

    results: dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _fit_station,
                stn,
                str(DB_PATH),
                str(PARAMS_DIR),
                args.force,
            ): stn
            for stn in stations
        }

        with tqdm(total=len(futures), desc="Fitting", unit=" stn") as pbar:
            for future in as_completed(futures):
                ref, status = future.result()
                results[ref] = status
                if status.startswith("error"):
                    log.warning("%-12s  %s", ref, status)
                pbar.set_postfix_str(status[:30])
                pbar.update(1)

    from collections import Counter
    counts = Counter(
        ("ok" if v.startswith("ok") else "skipped" if v == "skipped" else "error")
        for v in results.values()
    )
    log.info(
        "Fitting done — ok=%d  skipped=%d  errors=%d  (%s)",
        counts["ok"], counts["skipped"], counts["error"], _elapsed(t0),
    )

    # Build merged scores table
    build_scores(force=args.force)
    log.info("distributions.py complete  (%s)", _elapsed(t0))


if __name__ == "__main__":
    main()
