"""
subgroups.py — Assign stations to River Basin Districts via spatial join.

Downloads the official EA River Basin District boundaries from the
Environment Agency's WFS service and joins them to the cohort station
coordinates.  Outputs per-RBD model selection summaries and Johnson SU
parameter trajectories for both daily and monthly fitting modes.

Outputs  (written to outputs/subgroups/<mode>/)
-------
    cohort_rbd.parquet
        Cohort with rbd_name column added.  Shared across modes.

    rbd_selection_summary.parquet
        Per-RBD win rates for each distribution.

    rbd_js_params.parquet
        Johnson SU parameter medians per RBD from rolling fits.

Usage
-----
    python src/subgroups.py                  # daily (default)
    python src/subgroups.py --mode monthly   # monthly pooled fits
    python src/subgroups.py --mode both      # run both
    python src/subgroups.py --force

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.parse as _up
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
SCORES_DIR  = ROOT / "outputs" / "model_scores"
OUT_BASE    = ROOT / "outputs" / "subgroups"

# Score files per mode
SCORE_FILES = {
    "daily": {
        "global":  SCORES_DIR / "daily_global_scores.parquet",
        "rolling": SCORES_DIR / "daily_rolling_scores.parquet",
    },
    "monthly": {
        "global":  SCORES_DIR / "global_scores.parquet",
        "rolling": SCORES_DIR / "rolling_scores.parquet",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "subgroups.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── EA WFS ─────────────────────────────────────────────────────────────────────
_RBD_LAYER = (
    "dataset-779ada21-d465-11e4-8a4f-f0def148f590"
    ":WFD_River_Basin_Districts_Cycle_1"
)
RBD_WFS_URL = (
    "https://environment.data.gov.uk/spatialdata/wfd-river-basin-districts/wfs"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=" + _up.quote(_RBD_LAYER) +
    "&outputFormat=application/json"
)

# Shared boundaries file (written once, reused across modes)
RBD_GEOJSON = OUT_BASE / "rbd_boundaries.geojson"


# ── Spatial helpers ────────────────────────────────────────────────────────────

def fetch_rbd_boundaries() -> gpd.GeoDataFrame:
    """Fetch RBD polygons from EA WFS. Served in EPSG:27700; reprojected to 4326."""
    import urllib.request, json

    log.info("Fetching RBD boundaries from EA WFS ...")
    with urllib.request.urlopen(RBD_WFS_URL, timeout=30) as r:
        data = json.loads(r.read())

    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:27700")
    gdf = gdf.to_crs("EPSG:4326")
    log.info("Fetched %d RBD polygons: %s",
             len(gdf), sorted(gdf["rbd_name"].tolist()))
    return gdf[["rbd_name", "geometry"]]


def load_or_fetch_boundaries() -> gpd.GeoDataFrame:
    if RBD_GEOJSON.exists():
        log.info("Loading cached RBD boundaries from %s", RBD_GEOJSON.name)
        return gpd.read_file(RBD_GEOJSON)
    gdf = fetch_rbd_boundaries()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    gdf_save = gdf.copy()
    gdf_save["geometry"] = gdf_save["geometry"].simplify(0.01, preserve_topology=True)
    gdf_save.to_file(RBD_GEOJSON, driver="GeoJSON")
    log.info("Saved RBD boundaries to %s", RBD_GEOJSON.name)
    return gdf


def assign_rbd(cohort: pd.DataFrame,
               rbd: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point-in-polygon join: assign each station to its RBD."""
    stations_gdf = gpd.GeoDataFrame(
        cohort.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(cohort["long"], cohort["lat"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(stations_gdf, rbd, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    result = cohort.copy()
    result["rbd_name"] = joined["rbd_name"].values
    n_unmatched = result["rbd_name"].isna().sum()
    if n_unmatched:
        log.warning("%d stations did not match any RBD polygon", n_unmatched)
    return result


# ── Selection columns (mirrors selection.py logic) ─────────────────────────────

def add_selection_columns(df: pd.DataFrame,
                          group_cols: list[str]) -> pd.DataFrame:
    """Compute delta_aic, akaike_weight, rank_aic/bic/ks in-place."""
    df = df.copy()
    df["min_aic"] = df.groupby(group_cols)["aic"].transform("min")
    df["min_bic"] = df.groupby(group_cols)["bic"].transform("min")
    df["delta_aic"] = df["aic"] - df["min_aic"]
    df["delta_bic"] = df["bic"] - df["min_bic"]
    df.drop(columns=["min_aic", "min_bic"], inplace=True)

    df["_w_aic"] = np.exp(-0.5 * df["delta_aic"])
    df["_w_bic"] = np.exp(-0.5 * df["delta_bic"])
    df["akaike_weight"] = df["_w_aic"] / df.groupby(group_cols)["_w_aic"].transform("sum")
    df["bic_weight"]    = df["_w_bic"] / df.groupby(group_cols)["_w_bic"].transform("sum")
    df.drop(columns=["_w_aic", "_w_bic"], inplace=True)

    df["rank_aic"] = df.groupby(group_cols)["aic"].rank(method="min").astype(int)
    df["rank_bic"] = df.groupby(group_cols)["bic"].rank(method="min").astype(int)
    df["rank_ks"]  = df.groupby(group_cols)["ks_statistic"].rank(method="min").astype(int)
    return df


def load_with_selection(path: Path, group_cols: list[str]) -> pd.DataFrame:
    """Load a score parquet, adding selection columns if not already present."""
    df = pd.read_parquet(path)
    if "rank_aic" not in df.columns:
        log.info("Computing selection columns for %s ...", path.name)
        df = add_selection_columns(df, group_cols)
    return df


# ── Summary builders ───────────────────────────────────────────────────────────

def build_rbd_selection_summary(cohort_rbd: pd.DataFrame,
                                mode: str) -> pd.DataFrame:
    """Per-RBD win rates for each distribution, global fits."""
    sg = load_with_selection(
        SCORE_FILES[mode]["global"],
        group_cols=["station_reference"],
    )
    merged = sg.merge(cohort_rbd[["station_reference", "rbd_name"]],
                      on="station_reference", how="left")

    rows = []
    for rbd, grp in merged.groupby("rbd_name"):
        n_stations = grp["station_reference"].nunique()
        for dist, dg in grp.groupby("distribution"):
            n_win = (dg["rank_aic"] == 1).sum()
            rows.append({
                "rbd_name":     rbd,
                "n_stations":   n_stations,
                "distribution": dist,
                "n_win_aic":    int(n_win),
                "pct_win_aic":  round(100 * n_win / n_stations, 2),
                "mean_ks":      round(dg["ks_statistic"].mean(), 4),
                "median_ks":    round(dg["ks_statistic"].median(), 4),
                "mean_akaike":  round(dg["akaike_weight"].mean(), 4),
            })
    return pd.DataFrame(rows)


def build_rbd_js_params(cohort_rbd: pd.DataFrame,
                        mode: str) -> pd.DataFrame:
    """Median Johnson SU parameters per RBD per rolling window."""
    sr = load_with_selection(
        SCORE_FILES[mode]["rolling"],
        group_cols=["station_reference", "window_start_year"],
    )
    sr = sr.loc[(sr["distribution"] == "johnson_su") & (sr["converged"] == True)]
    sr = sr.merge(cohort_rbd[["station_reference", "rbd_name"]],
                  on="station_reference", how="left")

    rows = []
    for (rbd, yr), grp in sr.groupby(["rbd_name", "window_start_year"]):
        rows.append({
            "rbd_name":          rbd,
            "window_start_year": yr,
            "window_mid_year":   yr + 5,
            "n_stations":        len(grp),
            "a_median":          grp["p1"].median(),
            "b_median":          grp["p2"].median(),
            "loc_median":        grp["p3"].median(),
            "scale_median":      grp["p4"].median(),
        })
    return pd.DataFrame(rows)


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(mode: str,
                 cohort_rbd: pd.DataFrame,
                 summary: pd.DataFrame) -> None:
    print()
    print("=" * 65)
    print(f"  MODE: {mode.upper()}  —  STATIONS PER RBD")
    print("=" * 65)
    counts = (cohort_rbd.groupby("rbd_name").size()
              .rename("n_stations").sort_values(ascending=False).reset_index())
    print(counts.to_string(index=False))

    print()
    print("=" * 65)
    print("  JOHNSON SU WIN RATE (AIC) BY RBD")
    print("=" * 65)
    js = (summary[summary["distribution"] == "johnson_su"]
          .sort_values("pct_win_aic", ascending=False)
          [["rbd_name", "n_stations", "pct_win_aic", "median_ks", "mean_akaike"]])
    print(js.to_string(index=False))
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def run_mode(mode: str, cohort_rbd: pd.DataFrame, force: bool) -> None:
    out_dir = OUT_BASE / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary = out_dir / "rbd_selection_summary.parquet"
    out_params  = out_dir / "rbd_js_params.parquet"

    if out_summary.exists() and out_params.exists() and not force:
        log.info("[%s] Outputs exist — skipping (use --force).", mode)
        summary = pd.read_parquet(out_summary)
    else:
        summary = build_rbd_selection_summary(cohort_rbd, mode)
        summary.to_parquet(out_summary, index=False)
        log.info("[%s] Written: %s  (%d rows)", mode, out_summary.name, len(summary))

        params = build_rbd_js_params(cohort_rbd, mode)
        params.to_parquet(out_params, index=False)
        log.info("[%s] Written: %s  (%d rows)", mode, out_params.name, len(params))

    print_report(mode, cohort_rbd, summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",  choices=["daily", "monthly", "both"], default="daily")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.perf_counter()
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Cohort + RBD assignment (shared across modes)
    cohort_rbd_path = OUT_BASE / "cohort_rbd.parquet"
    if cohort_rbd_path.exists() and not args.force:
        log.info("Loading cached cohort_rbd from %s", cohort_rbd_path.name)
        cohort_rbd = pd.read_parquet(cohort_rbd_path)
    else:
        cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
        rbd    = load_or_fetch_boundaries()
        cohort_rbd = assign_rbd(cohort, rbd)
        cohort_rbd.to_parquet(cohort_rbd_path, index=False)
        log.info("Written: %s", cohort_rbd_path.name)

    modes = ["daily", "monthly"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_mode(mode, cohort_rbd, args.force)

    log.info("subgroups.py complete  (%.1fs)", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
