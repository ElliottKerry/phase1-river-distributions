"""
subgroups.py — Assign stations to River Basin Districts via spatial join.

Downloads the official EA River Basin District boundaries from the
Environment Agency's WFS service and joins them to the cohort station
coordinates.  Outputs a 'rbd' column added to cohort data, plus
per-RBD model selection summaries.

Outputs
-------
    outputs/subgroups/cohort_rbd.parquet
        Cohort with rbd_name column added.

    outputs/subgroups/rbd_selection_summary.parquet
        Per-RBD win rates for each distribution.

    outputs/subgroups/rbd_js_params.parquet
        Johnson SU national medians per RBD from rolling fits.

Usage
-----
    python src/subgroups.py
    python src/subgroups.py --force

Run from the project root (phase1_river_distributions/).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
SCORES_DIR  = ROOT / "outputs" / "model_scores"
TEMPORAL_DIR= ROOT / "outputs" / "temporal"
OUT_DIR     = ROOT / "outputs" / "subgroups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "subgroups.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# EA WFS endpoint for River Basin Districts (England & Wales)
# Source: Environment Agency open data — WFD River Basin Districts Cycle 1
_RBD_LAYER = (
    "dataset-779ada21-d465-11e4-8a4f-f0def148f590"
    ":WFD_River_Basin_Districts_Cycle_1"
)
import urllib.parse as _up
RBD_WFS_URL = (
    "https://environment.data.gov.uk/spatialdata/wfd-river-basin-districts/wfs"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=" + _up.quote(_RBD_LAYER) +
    "&outputFormat=application/json"
)


def fetch_rbd_boundaries() -> gpd.GeoDataFrame:
    """
    Fetch River Basin District polygon boundaries from the EA WFS service.
    Returns a GeoDataFrame in EPSG:4326 with a 'rbd_name' column.
    The WFS serves data in EPSG:27700 (British National Grid); we reproject.
    """
    import urllib.request, json

    log.info("Fetching RBD boundaries from EA WFS ...")
    with urllib.request.urlopen(RBD_WFS_URL, timeout=30) as r:
        data = json.loads(r.read())

    # WFS returns EPSG:27700 — set explicitly then reproject to WGS84
    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:27700")
    gdf = gdf.to_crs("EPSG:4326")
    log.info("Fetched %d RBD polygons: %s",
             len(gdf), sorted(gdf["rbd_name"].tolist()))
    return gdf[["rbd_name", "geometry"]]


def assign_rbd(cohort: pd.DataFrame,
               rbd: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Point-in-polygon join: assign each station to its RBD.
    Stations that fall outside all polygons (e.g. cross-border) get NaN.
    """
    stations_gdf = gpd.GeoDataFrame(
        cohort.copy(),
        geometry=[Point(lon, lat)
                  for lon, lat in zip(cohort["long"], cohort["lat"])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(stations_gdf, rbd, how="left", predicate="within")

    # sjoin may duplicate rows if a point falls on a boundary — keep first match
    joined = joined[~joined.index.duplicated(keep="first")]

    result = cohort.copy()
    result["rbd_name"] = joined["rbd_name"].values
    n_unmatched = result["rbd_name"].isna().sum()
    if n_unmatched:
        log.warning("%d stations did not match any RBD polygon", n_unmatched)
        log.warning(result[result["rbd_name"].isna()][
            ["station_reference", "label", "lat", "long"]].to_string())
    return result


def build_rbd_selection_summary(cohort_rbd: pd.DataFrame) -> pd.DataFrame:
    """
    Per-RBD win rates for each distribution, from daily global fits.
    """
    sg = pd.read_parquet(SCORES_DIR / "selection_global.parquet")
    merged = sg.merge(cohort_rbd[["station_reference", "rbd_name"]],
                      on="station_reference", how="left")

    rows = []
    for rbd, grp in merged.groupby("rbd_name"):
        n_stations = grp["station_reference"].nunique()
        for dist, dg in grp.groupby("distribution"):
            n_win = (dg["rank_aic"] == 1).sum()
            rows.append({
                "rbd_name":        rbd,
                "n_stations":      n_stations,
                "distribution":    dist,
                "n_win_aic":       int(n_win),
                "pct_win_aic":     round(100 * n_win / n_stations, 2),
                "mean_ks":         round(dg["ks_statistic"].mean(), 4),
                "median_ks":       round(dg["ks_statistic"].median(), 4),
                "mean_akaike":     round(dg["akaike_weight"].mean(), 4),
            })
    return pd.DataFrame(rows)


def build_rbd_js_params(cohort_rbd: pd.DataFrame) -> pd.DataFrame:
    """
    Median Johnson SU parameters per RBD across rolling windows.
    Useful for seeing regional differences in flood tail behaviour.
    """
    sr = pd.read_parquet(SCORES_DIR / "selection_rolling.parquet")
    sr = sr.loc[(sr["distribution"] == "johnson_su") & (sr["converged"] == True)]
    sr = sr.merge(cohort_rbd[["station_reference", "rbd_name"]],
                  on="station_reference", how="left")

    rows = []
    for (rbd, yr), grp in sr.groupby(["rbd_name", "window_start_year"]):
        rows.append({
            "rbd_name":         rbd,
            "window_start_year":yr,
            "window_mid_year":  yr + 5,
            "n_stations":       len(grp),
            "a_median":         grp["p1"].median(),
            "b_median":         grp["p2"].median(),
            "loc_median":       grp["p3"].median(),
            "scale_median":     grp["p4"].median(),
        })
    return pd.DataFrame(rows)


def print_report(cohort_rbd: pd.DataFrame,
                 summary: pd.DataFrame) -> None:
    print()
    print("=" * 65)
    print("  STATIONS PER RIVER BASIN DISTRICT")
    print("=" * 65)
    counts = (cohort_rbd.groupby("rbd_name")
              .size().rename("n_stations")
              .sort_values(ascending=False)
              .reset_index())
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    t0    = time.perf_counter()
    out_cohort  = OUT_DIR / "cohort_rbd.parquet"
    out_summary = OUT_DIR / "rbd_selection_summary.parquet"
    out_params  = OUT_DIR / "rbd_js_params.parquet"

    if all(p.exists() for p in (out_cohort, out_summary, out_params)) and not args.force:
        log.info("All outputs exist — skipping (use --force).")
        cohort_rbd = pd.read_parquet(out_cohort)
    else:
        cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
        rbd    = fetch_rbd_boundaries()
        cohort_rbd = assign_rbd(cohort, rbd)
        cohort_rbd.to_parquet(out_cohort, index=False)
        log.info("Written: %s", out_cohort.name)

    summary = build_rbd_selection_summary(cohort_rbd)
    summary.to_parquet(out_summary, index=False)
    log.info("Written: %s  (%d rows)", out_summary.name, len(summary))

    params = build_rbd_js_params(cohort_rbd)
    params.to_parquet(out_params, index=False)
    log.info("Written: %s  (%d rows)", out_params.name, len(params))

    print_report(cohort_rbd, summary)
    s = time.perf_counter() - t0
    log.info("subgroups.py complete  (%.1fs)", s)


if __name__ == "__main__":
    main()
