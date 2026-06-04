"""
fit_log.py — Fit distributions to log-transformed daily water levels.

Uses the same 257-station cohort and 9 distributions as the main daily
pipeline but applies log(mean_level_m) before fitting.  Produces RBD
subgroup summaries that can be compared directly with the raw-data results
in outputs/subgroups/daily/.

Outputs
-------
    outputs/model_scores/log_daily_global_scores.parquet
        Per-station per-distribution fit statistics (same schema as
        daily_global_scores.parquet).  Parameters are in log-space.

    outputs/subgroups/log_daily/rbd_selection_summary.parquet
        Per-RBD win rates for each distribution.

    outputs/subgroups/log_daily/rbd_js_params.parquet
        Johnson SU parameter medians per RBD (log-space).

Usage
-----
    python src/fit_log.py
    python src/fit_log.py --force
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
SCORES_DIR  = ROOT / "outputs" / "model_scores"
SUBGROUPS_DIR = ROOT / "outputs" / "subgroups"

DB_PATH     = DATA_DIR / "river_data.duckdb"
SCORES_OUT  = SCORES_DIR / "log_daily_global_scores.parquet"
OUT_DIR     = SUBGROUPS_DIR / "log_daily"

START_YEAR  = 1980
END_YEAR    = 2024
MIN_OBS     = 1_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "fit_log.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Distribution map: internal name → (scipy dist, n_params)
DISTRIBUTIONS: dict[str, tuple] = {
    "johnson_su":   (stats.johnsonsu,   4),
    "gen_logistic": (stats.genlogistic,  3),
    "gev":          (stats.genextreme,   3),
    "pearson3":     (stats.pearson3,     3),
    "lognormal":    (stats.lognorm,      3),
    "gamma":        (stats.gamma,        3),
    "weibull":      (stats.weibull_min,  3),
    "gumbel":       (stats.gumbel_r,     2),
    "normal":       (stats.norm,         2),
}


# ── Fitting ────────────────────────────────────────────────────────────────────

def fit_station(values: np.ndarray) -> list[dict]:
    """Fit all 9 distributions to log-transformed values. Returns list of row dicts."""
    n = len(values)
    rows = []
    for dist_name, (dist, k) in DISTRIBUTIONS.items():
        row: dict = {
            "distribution": dist_name,
            "n_obs":        np.int16(n),
            "p1": np.nan, "p2": np.nan, "p3": np.nan, "p4": np.nan,
            "log_likelihood": np.nan,
            "aic":          np.nan,
            "bic":          np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue":    np.nan,
            "converged":    False,
        }
        try:
            params = dist.fit(values)
            ll     = float(dist.logpdf(values, *params).sum())
            if not np.isfinite(ll):
                raise ValueError("non-finite log-likelihood")

            aic = 2 * k - 2 * ll
            bic = k * math.log(n) - 2 * ll
            ks_stat, ks_p = stats.kstest(values, dist.cdf, args=params)

            param_list = list(params)
            row.update({
                "p1":           param_list[0] if len(param_list) > 0 else np.nan,
                "p2":           param_list[1] if len(param_list) > 1 else np.nan,
                "p3":           param_list[2] if len(param_list) > 2 else np.nan,
                "p4":           param_list[3] if len(param_list) > 3 else np.nan,
                "log_likelihood": ll,
                "aic":          aic,
                "bic":          bic,
                "ks_statistic": ks_stat,
                "ks_pvalue":    ks_p,
                "converged":    True,
            })
        except Exception:
            pass
        rows.append(row)
    return rows


# ── Selection columns ──────────────────────────────────────────────────────────

def add_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    gc = ["station_reference"]
    df = df.copy()
    df["min_aic"]   = df.groupby(gc)["aic"].transform("min")
    df["min_bic"]   = df.groupby(gc)["bic"].transform("min")
    df["delta_aic"] = df["aic"] - df["min_aic"]
    df["delta_bic"] = df["bic"] - df["min_bic"]
    df.drop(columns=["min_aic", "min_bic"], inplace=True)

    w_aic = np.exp(-0.5 * df["delta_aic"])
    w_bic = np.exp(-0.5 * df["delta_bic"])
    df["akaike_weight"] = w_aic / df.groupby(gc)["delta_aic"].transform(
        lambda x: np.exp(-0.5 * x).sum()
    )
    df["bic_weight"] = w_bic / df.groupby(gc)["delta_bic"].transform(
        lambda x: np.exp(-0.5 * x).sum()
    )

    df["rank_aic"] = df.groupby(gc)["aic"].rank(method="min").astype("Int64")
    df["rank_bic"] = df.groupby(gc)["bic"].rank(method="min").astype("Int64")
    df["rank_ks"]  = df.groupby(gc)["ks_statistic"].rank(method="min").astype("Int64")
    return df


# ── RBD summaries ──────────────────────────────────────────────────────────────

def build_rbd_selection_summary(scores: pd.DataFrame,
                                cohort_rbd: pd.DataFrame) -> pd.DataFrame:
    merged = scores.merge(
        cohort_rbd[["station_reference", "rbd_name"]],
        on="station_reference", how="left",
    )
    rows = []
    for rbd, grp in merged.groupby("rbd_name"):
        n_stations = grp["station_reference"].nunique()
        for dist, dg in grp.groupby("distribution"):
            n_win = int((dg["rank_aic"] == 1).sum())
            rows.append({
                "rbd_name":     rbd,
                "n_stations":   n_stations,
                "distribution": dist,
                "n_win_aic":    n_win,
                "pct_win_aic":  round(100 * n_win / n_stations, 2),
                "mean_ks":      round(dg["ks_statistic"].mean(), 4),
                "median_ks":    round(dg["ks_statistic"].median(), 4),
                "mean_akaike":  round(dg["akaike_weight"].mean(), 4),
            })
    return pd.DataFrame(rows)


def build_rbd_js_params(scores: pd.DataFrame,
                        cohort_rbd: pd.DataFrame) -> pd.DataFrame:
    js = scores[
        (scores["distribution"] == "johnson_su") & (scores["converged"])
    ].merge(
        cohort_rbd[["station_reference", "rbd_name"]],
        on="station_reference", how="left",
    )
    rows = []
    for rbd, grp in js.groupby("rbd_name"):
        rows.append({
            "rbd_name":     rbd,
            "n_stations":   len(grp),
            "a_median":     grp["p1"].median(),
            "b_median":     grp["p2"].median(),
            "loc_median":   grp["p3"].median(),
            "scale_median": grp["p4"].median(),
        })
    return pd.DataFrame(rows)


# ── Print report ───────────────────────────────────────────────────────────────

def print_report(summary: pd.DataFrame, cohort_rbd: pd.DataFrame) -> None:
    print()
    print("=" * 65)
    print("  LOG-DAILY  --  STATIONS PER RBD")
    print("=" * 65)
    counts = (cohort_rbd.groupby("rbd_name").size()
              .rename("n_stations").sort_values(ascending=False).reset_index())
    print(counts.to_string(index=False))

    print()
    print("=" * 65)
    print("  WIN RATE BY DISTRIBUTION (AIC)  --  LOG-TRANSFORMED DAILY")
    print("=" * 65)
    national = (
        summary.groupby("distribution")
        .apply(lambda g: pd.Series({
            "n_stations": g["n_stations"].iloc[0],
            "n_win_aic":  g["n_win_aic"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    total = summary["n_stations"].max()
    national["pct_win_aic"] = 100 * national["n_win_aic"] / total
    national = national.sort_values("pct_win_aic", ascending=False)
    print(national[["distribution", "n_win_aic", "pct_win_aic"]].to_string(index=False))

    print()
    print("=" * 65)
    print("  JOHNSON SU WIN RATE BY RBD")
    print("=" * 65)
    js = (summary[summary["distribution"] == "johnson_su"]
          .sort_values("pct_win_aic", ascending=False)
          [["rbd_name", "n_stations", "pct_win_aic", "median_ks", "mean_akaike"]])
    print(js.to_string(index=False))
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true",
                   help="Refit even if outputs already exist")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    t0    = time.perf_counter()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cohort + RBD assignments
    cohort    = pd.read_parquet(DATA_DIR / "cohort.parquet")
    cohort_rbd = pd.read_parquet(SUBGROUPS_DIR / "cohort_rbd.parquet")
    station_refs = cohort["station_reference"].tolist()
    log.info("Cohort: %d stations", len(station_refs))

    # ── Fit or load ────────────────────────────────────────────────────────────
    out_summary = OUT_DIR / "rbd_selection_summary.parquet"
    out_params  = OUT_DIR / "rbd_js_params.parquet"

    if SCORES_OUT.exists() and out_summary.exists() and not args.force:
        log.info("Outputs exist — loading (use --force to refit).")
        scores  = pd.read_parquet(SCORES_OUT)
        summary = pd.read_parquet(out_summary)
        print_report(summary, cohort_rbd)
        log.info("fit_log.py complete  (%.1fs)", time.perf_counter() - t0)
        return

    # ── Fit distributions ──────────────────────────────────────────────────────
    con  = duckdb.connect(str(DB_PATH), read_only=True)
    rows = []

    for ref in tqdm(station_refs, desc="Log-daily fits"):
        df_vals = con.execute(f"""
            SELECT mean_level_m
            FROM daily_averages
            WHERE station_reference = '{ref}'
              AND date >= '{START_YEAR}-01-01'
              AND date < '{END_YEAR + 1}-01-01'
              AND mean_level_m > 0
        """).df()

        values = df_vals["mean_level_m"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) < MIN_OBS:
            log.warning("Skipping %s — only %d valid observations", ref, len(values))
            continue

        log_vals = np.log(values)

        for row in fit_station(log_vals):
            row["station_reference"] = ref
            rows.append(row)

    con.close()

    scores = pd.DataFrame(rows)
    scores = scores[["station_reference", "distribution", "n_obs",
                     "p1", "p2", "p3", "p4",
                     "log_likelihood", "aic", "bic",
                     "ks_statistic", "ks_pvalue", "converged"]]
    scores = add_selection_columns(scores)
    scores.to_parquet(SCORES_OUT, index=False)
    log.info("Written: %s  (%d rows)", SCORES_OUT.name, len(scores))

    # ── RBD summaries ──────────────────────────────────────────────────────────
    clean = scores.dropna(subset=["aic", "ks_statistic"]).copy()

    summary = build_rbd_selection_summary(clean, cohort_rbd)
    summary.to_parquet(out_summary, index=False)
    log.info("Written: %s  (%d rows)", out_summary.name, len(summary))

    params = build_rbd_js_params(clean, cohort_rbd)
    params.to_parquet(out_params, index=False)
    log.info("Written: %s  (%d rows)", out_params.name, len(params))

    print_report(summary, cohort_rbd)
    log.info("fit_log.py complete  (%.1fs)", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
