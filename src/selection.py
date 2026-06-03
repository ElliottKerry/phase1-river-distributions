"""
selection.py — Formal model selection from fitted distribution scores.

Operates on the daily-level fits (daily_global and daily_rolling) produced
by fit_series.py.  For every station (global) and every station-window
(rolling), computes:

    - Best model by AIC, BIC, and KS statistic
    - ΔAIC / ΔBIC relative to the best model in each comparison set
    - Akaike weights  w_i = exp(-0.5·Δ_i) / Σ exp(-0.5·Δ_j)
      (the probability that model i is the best, given the candidate set)

Aggregate outputs
-----------------
    outputs/model_scores/selection_global.parquet
        One row per station × distribution: ΔAIC, ΔBIC, Akaike weight,
        rank_aic, rank_bic, rank_ks.

    outputs/model_scores/selection_rolling.parquet
        One row per (station, window) × distribution: same columns.

    outputs/model_scores/selection_summary.parquet
        One row per distribution: win rates, mean Akaike weight,
        median ΔAIC, evidence ratio vs runner-up.

Printed report
--------------
    Concise tables suitable for inclusion in the paper's results section.

Usage
-----
    python src/selection.py             # full analysis, print report
    python src/selection.py --no-print  # write files only
    python src/selection.py --force     # rebuild even if files exist

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
SCORES_DIR = ROOT / "outputs" / "model_scores"
DB_PATH    = ROOT / "data" / "processed" / "river_data.duckdb"

DG_PATH = SCORES_DIR / "daily_global_scores.parquet"
DR_PATH = SCORES_DIR / "daily_rolling_scores.parquet"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "selection.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DIST_ORDER = [
    "johnson_su", "gen_logistic", "pearson3", "normal",
    "lognormal",  "gamma",        "weibull",  "gev",   "gumbel",
]


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ── Core selection logic ───────────────────────────────────────────────────────

def add_selection_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Given a DataFrame with one row per (group × distribution), add:
        delta_aic, delta_bic       — difference from best in group
        akaike_weight              — model probability from ΔAIC
        bic_weight                 — model probability from ΔBIC
        rank_aic, rank_bic, rank_ks

    group_cols defines what constitutes one comparison set, e.g.
    ["station_reference"] for global or
    ["station_reference", "window_start_year"] for rolling.
    """
    df = df.copy()

    # ── ΔAIC / ΔBIC ───────────────────────────────────────────────────────────
    df["min_aic"] = df.groupby(group_cols)["aic"].transform("min")
    df["min_bic"] = df.groupby(group_cols)["bic"].transform("min")
    df["delta_aic"] = df["aic"] - df["min_aic"]
    df["delta_bic"] = df["bic"] - df["min_bic"]
    df.drop(columns=["min_aic", "min_bic"], inplace=True)

    # ── Akaike / BIC weights ──────────────────────────────────────────────────
    # exp(-0.5 * Δ), normalised within each group
    df["_w_aic"] = np.exp(-0.5 * df["delta_aic"])
    df["_w_bic"] = np.exp(-0.5 * df["delta_bic"])

    df["akaike_weight"] = df["_w_aic"] / df.groupby(group_cols)["_w_aic"].transform("sum")
    df["bic_weight"]    = df["_w_bic"] / df.groupby(group_cols)["_w_bic"].transform("sum")
    df.drop(columns=["_w_aic", "_w_bic"], inplace=True)

    # ── Ranks ─────────────────────────────────────────────────────────────────
    df["rank_aic"] = df.groupby(group_cols)["aic"].rank(method="min").astype(int)
    df["rank_bic"] = df.groupby(group_cols)["bic"].rank(method="min").astype(int)
    df["rank_ks"]  = df.groupby(group_cols)["ks_statistic"].rank(method="min").astype(int)

    return df


# ── Summary table ──────────────────────────────────────────────────────────────

def build_summary(sel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Collapse per-group × distribution selection results into a single
    summary row per distribution suitable for the paper's Table 1.
    """
    n_groups = sel.groupby(group_cols).ngroups

    summary = (
        sel.groupby("distribution")
        .agg(
            n_win_aic   = ("rank_aic",      lambda x: (x == 1).sum()),
            n_win_bic   = ("rank_bic",      lambda x: (x == 1).sum()),
            n_win_ks    = ("rank_ks",       lambda x: (x == 1).sum()),
            mean_akaike = ("akaike_weight", "mean"),
            mean_delta_aic = ("delta_aic",  "mean"),
            median_delta_aic = ("delta_aic","median"),
            mean_ks     = ("ks_statistic",  "mean"),
            median_ks   = ("ks_statistic",  "median"),
        )
        .reset_index()
    )

    summary["pct_win_aic"] = 100 * summary["n_win_aic"] / n_groups
    summary["pct_win_bic"] = 100 * summary["n_win_bic"] / n_groups
    summary["pct_win_ks"]  = 100 * summary["n_win_ks"]  / n_groups

    # Evidence ratio: mean Akaike weight of each dist relative to johnson_su
    js_weight = summary.loc[
        summary["distribution"] == "johnson_su", "mean_akaike"
    ].values
    if len(js_weight):
        summary["evidence_vs_johnson_su"] = js_weight[0] / summary["mean_akaike"]
    else:
        summary["evidence_vs_johnson_su"] = np.nan

    # Sort by AIC win rate
    summary = summary.sort_values("n_win_aic", ascending=False).reset_index(drop=True)
    return summary


# ── Temporal breakdown (rolling only) ─────────────────────────────────────────

def build_temporal(sel_rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Win rates by decade for the rolling fits — shows how distributional
    preference evolves over the study period.
    """
    sel = sel_rolling.copy()
    sel["decade"] = (sel["window_start_year"] // 10 * 10).astype(int)

    rows = []
    for decade, grp in sel.groupby("decade"):
        n = grp.groupby(["station_reference", "window_start_year"]).ngroups
        wins = grp[grp["rank_aic"] == 1].groupby("distribution").size()
        for dist in DIST_ORDER:
            rows.append({
                "decade":       decade,
                "distribution": dist,
                "n_wins":       int(wins.get(dist, 0)),
                "n_windows":    n,
                "pct_win_aic":  round(100 * wins.get(dist, 0) / n, 2) if n else 0,
            })

    return pd.DataFrame(rows)


# ── Print report ───────────────────────────────────────────────────────────────

def print_report(
    summary_global:  pd.DataFrame,
    summary_rolling: pd.DataFrame,
    temporal:        pd.DataFrame,
) -> None:

    w = 16
    cols_pct = ["distribution", "pct_win_aic", "pct_win_bic", "pct_win_ks",
                "mean_akaike", "median_ks"]

    def _fmt(df: pd.DataFrame) -> str:
        d = df[cols_pct].copy()
        d.columns = ["Distribution", "AIC wins %", "BIC wins %",
                     "KS wins %", "Mean Ak. wt.", "Median KS"]
        d["AIC wins %"]   = d["AIC wins %"].map("{:.1f}".format)
        d["BIC wins %"]   = d["BIC wins %"].map("{:.1f}".format)
        d["KS wins %"]    = d["KS wins %"].map("{:.1f}".format)
        d["Mean Ak. wt."] = d["Mean Ak. wt."].map("{:.4f}".format)
        d["Median KS"]    = d["Median KS"].map("{:.4f}".format)
        return d.to_string(index=False)

    print()
    print("=" * 72)
    print("  MODEL SELECTION — Daily global fits (~16,000 obs per station)")
    print("=" * 72)
    print(_fmt(summary_global))

    print()
    print("=" * 72)
    print("  MODEL SELECTION — Daily rolling fits (10-year windows)")
    print("=" * 72)
    print(_fmt(summary_rolling))

    print()
    print("=" * 72)
    print("  TEMPORAL EVOLUTION — AIC win rate by decade (rolling fits)")
    print("=" * 72)
    pivot = (
        temporal[temporal["distribution"].isin(
            ["johnson_su", "lognormal", "gamma", "weibull", "gev"]
        )]
        .pivot(index="distribution", columns="decade", values="pct_win_aic")
        .reindex(["johnson_su", "lognormal", "gamma", "weibull", "gev"])
    )
    pivot.columns = [str(c) + "s" for c in pivot.columns]
    print(pivot.round(1).to_string())

    # Johnson SU evidence summary
    js_row = summary_global[summary_global["distribution"] == "johnson_su"]
    if not js_row.empty:
        row = js_row.iloc[0]
        print()
        print("=" * 72)
        print("  JOHNSON SU EVIDENCE SUMMARY (global)")
        print("=" * 72)
        runner_up = summary_global[summary_global["distribution"] != "johnson_su"].iloc[0]
        print(f"  AIC wins:        {row['n_win_aic']:>4}  ({row['pct_win_aic']:.1f}% of stations)")
        print(f"  KS wins:         {row['n_win_ks']:>4}  ({row['pct_win_ks']:.1f}% of stations)")
        print(f"  Mean Ak. weight: {row['mean_akaike']:.4f}")
        print(f"  Runner-up:       {runner_up['distribution']}  "
              f"(mean Ak. weight {runner_up['mean_akaike']:.4f}, "
              f"evidence ratio {row['mean_akaike']/runner_up['mean_akaike']:.1f}×)")
        print(f"  Median dAIC vs best: {row['median_delta_aic']:.2f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-print", action="store_true", help="Suppress printed report")
    p.add_argument("--force",    action="store_true", help="Rebuild output files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()

    for path in (DG_PATH, DR_PATH):
        if not path.exists():
            log.error("Missing input: %s — run fit_series.py --mode daily_both first.", path)
            sys.exit(1)

    # ── Load scores ───────────────────────────────────────────────────────────
    log.info("Loading daily global scores …")
    dg = pd.read_parquet(DG_PATH)

    log.info("Loading daily rolling scores …")
    dr = pd.read_parquet(DR_PATH)

    # ── Compute selection columns ─────────────────────────────────────────────
    log.info("Computing selection statistics (global) …")
    sel_global = add_selection_columns(dg, group_cols=["station_reference"])

    log.info("Computing selection statistics (rolling) …")
    sel_rolling = add_selection_columns(
        dr, group_cols=["station_reference", "window_start_year"]
    )

    # ── Build summaries ───────────────────────────────────────────────────────
    summary_global  = build_summary(sel_global,  ["station_reference"])
    summary_rolling = build_summary(sel_rolling, ["station_reference", "window_start_year"])
    temporal        = build_temporal(sel_rolling)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_global  = SCORES_DIR / "selection_global.parquet"
    out_rolling = SCORES_DIR / "selection_rolling.parquet"
    out_summary = SCORES_DIR / "selection_summary.parquet"
    out_temporal= SCORES_DIR / "selection_temporal.parquet"

    for path, df in [
        (out_global,  sel_global),
        (out_rolling, sel_rolling),
        (out_summary, summary_global),
        (out_temporal, temporal),
    ]:
        if path.exists() and not args.force:
            log.info("%s already exists — skipping (use --force).", path.name)
        else:
            df.to_parquet(path, index=False)
            log.info("Written: %s", path.name)

    # ── Print report ──────────────────────────────────────────────────────────
    if not args.no_print:
        print_report(summary_global, summary_rolling, temporal)

    log.info("selection.py complete  (%s)", _elapsed(t0))


if __name__ == "__main__":
    main()
