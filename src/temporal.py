"""
temporal.py — Track how fitted distribution parameters evolve over 1980–2024.

Uses the daily rolling window fits from fit_series.py to examine how location,
scale, skewness, and tail behaviour of UK river levels have changed nationally
over the 45-year study period.

Focus distribution: Johnson SU (the preferred model by AIC and KS).
Also tracks the winning distribution's parameters regardless of family.

Parameter interpretation for Johnson SU  (scipy: a, b, loc, scale)
-------------------------------------------------------------------
    p1  a      Skewness shape  — positive = right-skewed (heavy upper flood tail)
    p2  b      Tail shape      — smaller b = heavier tails (more extreme events)
    p3  loc    Location        — shifts the centre of the distribution
    p4  scale  Scale           — spread / variance proxy

Outputs
-------
    outputs/temporal/national_trends.parquet
        National median (and IQR) of each parameter per window — the headline
        time series for figures.

    outputs/temporal/station_trajectories.parquet
        Per-station parameter trajectory — for spatial analysis and maps.

    outputs/temporal/trend_tests.parquet
        Mann-Kendall trend test results for each parameter.

    outputs/temporal/event_context.parquet
        Parameter values averaged over documented UK drought/flood windows,
        for contextual annotation of figures.

Usage
-----
    python src/temporal.py              # full analysis + printed report
    python src/temporal.py --no-print   # write files only
    python src/temporal.py --force

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
SCORES_DIR  = ROOT / "outputs" / "model_scores"
PARAMS_DIR  = ROOT / "outputs" / "fitted_params" / "daily_rolling"
TEMPORAL_DIR= ROOT / "outputs" / "temporal"
TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

DR_PATH  = SCORES_DIR / "daily_rolling_scores.parquet"
SEL_PATH = SCORES_DIR / "selection_rolling.parquet"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "temporal.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ── Documented UK climate events ───────────────────────────────────────────────
# Used to annotate figures and contextualise parameter shifts.
UK_EVENTS = {
    "Drought 1995-96":   (1985, 1996),   # window_start_year range that captures it
    "Floods 2000":       (1991, 2000),
    "Drought 2003":      (1994, 2003),
    "Floods 2007":       (1998, 2007),
    "Floods 2013-14":    (2004, 2013),
    "Drought 2018":      (2009, 2018),
    "Floods 2019-20":    (2010, 2019),
    "Drought 2022":      (2013, 2022),
}


# ── Mann-Kendall trend test ────────────────────────────────────────────────────

def mann_kendall(series: pd.Series) -> dict:
    """
    Mann-Kendall trend test on a time series.
    Uses scipy.stats.kendalltau against a monotone index as a proxy.
    Returns tau, p-value, and a plain-English trend label.
    """
    x = series.dropna().values
    if len(x) < 8:
        return {"tau": np.nan, "pvalue": np.nan, "trend": "insufficient data"}

    n   = len(x)
    idx = np.arange(n)
    tau, pvalue = stats.kendalltau(idx, x)

    if pvalue < 0.05:
        trend = "increasing" if tau > 0 else "decreasing"
    else:
        trend = "no significant trend"

    return {"tau": round(tau, 4), "pvalue": round(pvalue, 4), "trend": trend}


# ── Load and prepare data ──────────────────────────────────────────────────────

def load_johnson_su_rolling() -> pd.DataFrame:
    """
    Load rolling window fits for Johnson SU only, from all per-station Parquets.
    Returns (station_reference, window_start_year, window_mid_year,
             n_obs, p1=a, p2=b, p3=loc, p4=scale, aic, ks_statistic).
    """
    log.info("Loading Johnson SU rolling fits …")
    t0 = time.perf_counter()

    import duckdb
    glob = str(PARAMS_DIR / "*.parquet")
    con  = duckdb.connect()
    df   = con.execute(f"""
        SELECT
            station_reference,
            window_start_year,
            window_end_year,
            window_mid_year,
            n_obs,
            p1   AS a,        -- skewness shape
            p2   AS b,        -- tail shape (smaller = heavier tails)
            p3   AS loc,
            p4   AS scale,
            aic,
            ks_statistic
        FROM read_parquet('{glob}', union_by_name = true)
        WHERE distribution = 'johnson_su'
          AND converged    = true
          AND p2 > 0       -- b must be positive (validity check)
          AND p4 > 0       -- scale must be positive
        ORDER BY station_reference, window_start_year
    """).df()
    con.close()

    log.info("Loaded %d Johnson SU window fits  (%s)", len(df), _elapsed(t0))
    return df


def load_best_fit_rolling() -> pd.DataFrame:
    """
    Load the best-fit distribution parameters for every (station, window),
    regardless of which distribution won.  Uses the pre-computed selection
    ranks from selection.py.
    """
    log.info("Loading best-fit rolling parameters …")
    import duckdb
    glob = str(PARAMS_DIR / "*.parquet")
    sel  = pd.read_parquet(SEL_PATH)

    # Keep only rank_aic == 1 rows
    best = sel[sel["rank_aic"] == 1][
        ["station_reference", "window_start_year", "distribution"]
    ]

    con = duckdb.connect()
    all_params = con.execute(f"""
        SELECT station_reference, window_start_year, window_mid_year,
               distribution, n_obs, p1, p2, p3, p4, aic, ks_statistic
        FROM read_parquet('{glob}', union_by_name = true)
        WHERE converged = true
    """).df()
    con.close()

    merged = best.merge(all_params,
                        on=["station_reference", "window_start_year", "distribution"],
                        how="left")
    log.info("Best-fit params: %d rows", len(merged))
    return merged


# ── National trends ────────────────────────────────────────────────────────────

def build_national_trends(js: pd.DataFrame) -> pd.DataFrame:
    """
    For each rolling window year, compute the national median (and IQR) of
    Johnson SU parameters across all cohort stations.
    """
    grp = js.groupby("window_start_year")

    rows = []
    for year, g in grp:
        rows.append({
            "window_start_year": year,
            "window_mid_year":   float(g["window_mid_year"].iloc[0]),
            "n_stations":        len(g),
            # Scale — variance proxy
            "scale_median":      g["scale"].median(),
            "scale_q25":         g["scale"].quantile(0.25),
            "scale_q75":         g["scale"].quantile(0.75),
            # a — skewness shape
            "a_median":          g["a"].median(),
            "a_q25":             g["a"].quantile(0.25),
            "a_q75":             g["a"].quantile(0.75),
            # b — tail shape (smaller = heavier upper tail)
            "b_median":          g["b"].median(),
            "b_q25":             g["b"].quantile(0.25),
            "b_q75":             g["b"].quantile(0.75),
            # loc — location
            "loc_median":        g["loc"].median(),
            "loc_q25":           g["loc"].quantile(0.25),
            "loc_q75":           g["loc"].quantile(0.75),
            # Fit quality
            "ks_median":         g["ks_statistic"].median(),
            "aic_median":        g["aic"].median(),
        })

    return pd.DataFrame(rows).sort_values("window_start_year").reset_index(drop=True)


# ── Trend tests ────────────────────────────────────────────────────────────────

def build_trend_tests(national: pd.DataFrame) -> pd.DataFrame:
    """Run Mann-Kendall on each national parameter series."""
    params = {
        "scale":  "Scale (variance proxy)",
        "a":      "a (skewness shape)",
        "b":      "b (tail shape, smaller=heavier)",
        "loc":    "loc (location)",
    }
    rows = []
    for col, label in params.items():
        mk = mann_kendall(national[f"{col}_median"])
        rows.append({
            "parameter":   col,
            "label":       label,
            "tau":         mk["tau"],
            "pvalue":      mk["pvalue"],
            "trend":       mk["trend"],
            "start_value": round(national[f"{col}_median"].iloc[0],  4),
            "end_value":   round(national[f"{col}_median"].iloc[-1], 4),
            "change_pct":  round(
                100 * (national[f"{col}_median"].iloc[-1] -
                       national[f"{col}_median"].iloc[0]) /
                abs(national[f"{col}_median"].iloc[0]), 2
            ),
        })
    return pd.DataFrame(rows)


# ── Event context ──────────────────────────────────────────────────────────────

def build_event_context(national: pd.DataFrame) -> pd.DataFrame:
    """
    For each documented UK climate event, report the national parameter
    values in the rolling window that captures the peak of that event.
    """
    rows = []
    for event, (w_start, w_end) in UK_EVENTS.items():
        mask = (national["window_start_year"] >= w_start) & \
               (national["window_start_year"] <= w_end)
        subset = national[mask]
        if subset.empty:
            continue
        # Use the window closest to the event end year
        closest = subset.iloc[(subset["window_start_year"] - w_end).abs().argsort()[:1]]
        rows.append({
            "event":              event,
            "window_start_year":  int(closest["window_start_year"].iloc[0]),
            "scale_median":       round(closest["scale_median"].iloc[0], 4),
            "a_median":           round(closest["a_median"].iloc[0], 4),
            "b_median":           round(closest["b_median"].iloc[0], 4),
            "loc_median":         round(closest["loc_median"].iloc[0], 4),
        })
    return pd.DataFrame(rows)


# ── Station trajectories ───────────────────────────────────────────────────────

def build_station_trajectories(js: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-station trend direction for each parameter.
    Useful for spatial maps: which regions show the strongest increases
    in scale (variance) or tail weight?
    """
    rows = []
    for stn, grp in js.groupby("station_reference"):
        grp = grp.sort_values("window_start_year")
        for param in ["scale", "a", "b", "loc"]:
            mk = mann_kendall(grp[param])
            rows.append({
                "station_reference": stn,
                "parameter":         param,
                "tau":               mk["tau"],
                "pvalue":            mk["pvalue"],
                "trend":             mk["trend"],
                "start_value":       round(grp[param].iloc[0],  4),
                "end_value":         round(grp[param].iloc[-1], 4),
            })
    return pd.DataFrame(rows)


# ── Print report ───────────────────────────────────────────────────────────────

def print_report(national: pd.DataFrame, trends: pd.DataFrame,
                 events: pd.DataFrame) -> None:

    print()
    print("=" * 70)
    print("  TEMPORAL PARAMETER TRENDS  (Johnson SU, national medians)")
    print("=" * 70)

    # Show every 5 years
    display = national[national["window_start_year"] % 5 == 0][
        ["window_start_year", "scale_median", "a_median", "b_median", "loc_median", "n_stations"]
    ].copy()
    display.columns = ["Win.start", "Scale", "a (skew)", "b (tail)", "Loc", "N stn"]
    display = display.round(4)
    print(display.to_string(index=False))

    print()
    print("=" * 70)
    print("  MANN-KENDALL TREND TESTS  (p < 0.05 = significant)")
    print("=" * 70)
    print(trends[["label", "tau", "pvalue", "trend",
                  "start_value", "end_value", "change_pct"]]
          .rename(columns={
              "label":        "Parameter",
              "tau":          "Kendall tau",
              "pvalue":       "p-value",
              "trend":        "Trend",
              "start_value":  "1980s val",
              "end_value":    "2020s val",
              "change_pct":   "Change %",
          })
          .to_string(index=False))

    print()
    print("=" * 70)
    print("  PARAMETER VALUES AROUND KEY UK CLIMATE EVENTS")
    print("=" * 70)
    print(events.to_string(index=False))
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-print", action="store_true")
    p.add_argument("--force",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()

    for path in (DR_PATH, SEL_PATH):
        if not path.exists():
            log.error("Missing: %s — run fit_series.py and selection.py first.", path)
            sys.exit(1)

    js       = load_johnson_su_rolling()
    national = build_national_trends(js)
    trends   = build_trend_tests(national)
    events   = build_event_context(national)
    traj     = build_station_trajectories(js)

    outputs = {
        TEMPORAL_DIR / "national_trends.parquet":       national,
        TEMPORAL_DIR / "trend_tests.parquet":           trends,
        TEMPORAL_DIR / "event_context.parquet":         events,
        TEMPORAL_DIR / "station_trajectories.parquet":  traj,
    }
    for path, df in outputs.items():
        if path.exists() and not args.force:
            log.info("%s exists — skipping (use --force).", path.name)
        else:
            df.to_parquet(path, index=False)
            log.info("Written: %s  (%d rows)", path.name, len(df))

    if not args.no_print:
        print_report(national, trends, events)

    log.info("temporal.py complete  (%s)", _elapsed(t0))


if __name__ == "__main__":
    main()
