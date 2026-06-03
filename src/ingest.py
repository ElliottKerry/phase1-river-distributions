"""
ingest.py — Download EA National River Monitoring Archive (water levels) to data/raw/

Saves one Parquet file per station to  data/raw/<stationReference>.parquet.
A station manifest is written to        data/raw/stations.parquet.

Already-downloaded stations are skipped automatically (safe to re-run after
interruption).  Use --force to overwrite.

Usage
-----
    python src/ingest.py                         # all stations, 1980–today
    python src/ingest.py --start 1980-01-01 --end 2024-12-31
    python src/ingest.py --station 51107         # single station (smoke-test)
    python src/ingest.py --force                 # re-download everything
    python src/ingest.py --workers 5             # concurrent workers (default 4)
    python src/ingest.py --dry-run               # list stations, no download

Run from the project root (phase1_river_distributions/).
Logs to ingest.log and stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── EA Hydrology API constants ─────────────────────────────────────────────────
BASE               = "https://environment.data.gov.uk/hydrology"
STATION_PAGE_SIZE  = 500        # max per page for station listing
READINGS_LIMIT     = 90_000     # safe margin under the API's 100 k soft limit
CHUNK_YEARS        = 2          # years per readings request  ≈ 70 k rows max
DEFAULT_START      = "1980-01-01"
DEFAULT_END        = date.today().isoformat()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "ingest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── HTTP session ───────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    """Return a requests Session with exponential-backoff retry on 429/5xx."""
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=2,           # 2 s, 4 s, 8 s, 16 s, 32 s, 64 s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def _get(session: requests.Session, url: str, params: dict) -> dict:
    """GET with light rate-limiting; raises on non-200."""
    resp = session.get(url, params=params, timeout=60)
    if resp.status_code == 429:
        wait = int(resp.headers.get("Retry-After", 10))
        log.warning("Rate limited — sleeping %d s", wait)
        time.sleep(wait)
        resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Station discovery ──────────────────────────────────────────────────────────

def _scalar(v: object, default: str = "") -> str:
    """
    The EA API occasionally returns a field as a single-element list rather
    than a plain scalar (e.g. label: ["Thames at Kingston"]).
    Return the first element if it's a list, otherwise the value itself.
    """
    if isinstance(v, list):
        return str(v[0]) if v else default
    return str(v) if v is not None else default


def fetch_all_stations(session: requests.Session) -> pd.DataFrame:
    """
    Return a DataFrame of all EA stations that publish water-level data.

    Columns: notation, station_reference, label, lat, long, river_name,
             date_opened, wiski_id, nrfa_station_id, easting, northing
    """
    url    = f"{BASE}/id/stations.json"
    offset = 0
    rows: list[dict] = []

    with tqdm(desc="Fetching station list", unit=" stations") as pbar:
        while True:
            data = _get(session, url, {
                "observedProperty": "waterLevel",
                "_limit":           STATION_PAGE_SIZE,
                "_offset":          offset,
            })
            items = data.get("items", [])
            if not items:
                break

            for s in items:
                rows.append({
                    "notation":          _scalar(s.get("notation")),
                    "station_reference": _scalar(s.get("stationReference")),
                    "label":             _scalar(s.get("label")),
                    "lat":               s.get("lat"),
                    "long":              s.get("long"),
                    "easting":           s.get("easting"),
                    "northing":          s.get("northing"),
                    "river_name":        _scalar(s.get("riverName")),
                    "date_opened":       _scalar(s.get("dateOpened")),
                    "wiski_id":          _scalar(s.get("wiskiID")),
                    "nrfa_station_id":   _scalar(s.get("nrfaStationID")),
                })

            pbar.update(len(items))
            if len(items) < STATION_PAGE_SIZE:
                break
            offset += STATION_PAGE_SIZE
            time.sleep(0.1)   # be polite

    df = pd.DataFrame(rows)
    log.info("Found %d stations with water-level data", len(df))
    return df


# ── Measure discovery ──────────────────────────────────────────────────────────

def find_15min_level_measure(
    session: requests.Session, notation: str
) -> str | None:
    """
    Return the notation (URL path segment) of the station's 15-minute
    instantaneous water-level measure, or None if the station doesn't have one.

    The measure @id looks like:
        .../measures/{notation}-level-i-900-m-qualified
    900 seconds = 15 minutes.
    """
    url  = f"{BASE}/id/stations/{notation}/measures.json"
    data = _get(session, url, {})

    for m in data.get("items", []):
        measure_id = m.get("@id", "")
        # period=900 (15 min) + parameter contains 'level' + instantaneous
        period = m.get("period")
        param  = _scalar(m.get("parameterName") or m.get("parameter"))
        vtype  = _scalar(m.get("valueType"))

        is_15min       = (period == 900)
        is_level       = ("level" in param.lower() or "level" in measure_id.lower())
        is_instantaneous = ("instantaneous" in vtype.lower() or "-i-" in measure_id.lower())

        if is_15min and is_level and is_instantaneous:
            # Extract the notation = last path segment of the @id URL
            return measure_id.rstrip("/").split("/")[-1]

    return None


# ── Readings download ──────────────────────────────────────────────────────────

def _date_chunks(start: str, end: str, chunk_years: int) -> list[tuple[str, str]]:
    """
    Split [start, end] into consecutive chunks of `chunk_years` years.
    Returns a list of (chunk_start, chunk_end) ISO date strings.
    """
    chunks: list[tuple[str, str]] = []
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)

    while s < e:
        # advance by chunk_years years (approximate with 365*chunk_years days)
        chunk_end = min(
            date(s.year + chunk_years, s.month, s.day) - timedelta(days=1),
            e,
        )
        chunks.append((s.isoformat(), chunk_end.isoformat()))
        s = chunk_end + timedelta(days=1)

    return chunks


def fetch_readings(
    session: requests.Session,
    measure_notation: str,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    """
    Download all readings for one measure between start and end (inclusive).
    Handles pagination internally (offset-based).

    Returns a list of raw reading dicts: {dateTime, value, quality}.
    """
    url    = f"{BASE}/id/measures/{measure_notation}/readings.json"
    offset = 0
    rows: list[dict] = []

    while True:
        data = _get(session, url, {
            "mineq-date": start,
            "maxeq-date": end,
            "_limit":     READINGS_LIMIT,
            "_offset":    offset,
        })
        items = data.get("items", [])
        if not items:
            break

        rows.extend(items)

        if len(items) < READINGS_LIMIT:
            break
        offset += READINGS_LIMIT
        time.sleep(0.05)

    return rows


# ── Per-station orchestration ──────────────────────────────────────────────────

# PyArrow schema for raw readings — keeps file sizes small and types correct
_READINGS_SCHEMA = pa.schema([
    pa.field("station_reference", pa.string()),
    pa.field("datetime",          pa.string()),   # ISO-8601 string; cast in preprocess.py
    pa.field("value",             pa.float32()),
    pa.field("quality",           pa.string()),
])


def download_station(
    session:           requests.Session,
    station_reference: str,
    notation:          str,
    label:             str,
    start:             str,
    end:               str,
    force:             bool = False,
) -> tuple[str, str]:
    """
    Download all 15-minute level readings for one station and write a Parquet
    file to data/raw/<station_reference>.parquet.

    Returns (station_reference, status) where status is one of:
        'skipped'   — file exists and --force not set
        'no_measure'— station has no 15-min level measure
        'ok'        — downloaded successfully
        'error:<msg>'— exception was raised
    """
    out_path = RAW_DIR / f"{station_reference}.parquet"

    if out_path.exists() and not force:
        return station_reference, "skipped"

    try:
        measure_notation = find_15min_level_measure(session, notation)
        if measure_notation is None:
            return station_reference, "no_measure"

        all_rows: list[dict] = []
        for chunk_start, chunk_end in _date_chunks(start, end, CHUNK_YEARS):
            rows = fetch_readings(session, measure_notation, chunk_start, chunk_end)
            all_rows.extend(rows)
            time.sleep(0.05)

        if not all_rows:
            # Write empty file so the station is marked as processed
            pq.write_table(
                pa.table({col: [] for col in _READINGS_SCHEMA.names},
                         schema=_READINGS_SCHEMA),
                out_path,
            )
            return station_reference, "ok_empty"

        # Normalise to flat dicts and write Parquet
        records = [
            {
                "station_reference": station_reference,
                "datetime":          r.get("dateTime", ""),
                "value":             r.get("value"),
                "quality":           r.get("quality", ""),
            }
            for r in all_rows
        ]
        df = pd.DataFrame(records)
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float32")

        pq.write_table(
            pa.Table.from_pandas(df, schema=_READINGS_SCHEMA, preserve_index=False),
            out_path,
        )
        return station_reference, "ok"

    except Exception as exc:  # noqa: BLE001
        return station_reference, f"error:{exc}"


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start",   default=DEFAULT_START, help="First date (YYYY-MM-DD)")
    p.add_argument("--end",     default=DEFAULT_END,   help="Last date  (YYYY-MM-DD)")
    p.add_argument("--station", default=None,           help="Single stationReference to download")
    p.add_argument("--force",   action="store_true",    help="Re-download even if file exists")
    p.add_argument("--workers", type=int, default=4,    help="Parallel download threads")
    p.add_argument("--dry-run", action="store_true",    help="List stations only; no download")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    session = make_session()

    # ── 1. Station manifest ───────────────────────────────────────────────────
    manifest_path = RAW_DIR / "stations.parquet"

    if manifest_path.exists() and not args.force:
        log.info("Loading existing station manifest from %s", manifest_path)
        stations_df = pd.read_parquet(manifest_path)
    else:
        stations_df = fetch_all_stations(session)
        stations_df.to_parquet(manifest_path, index=False)
        log.info("Station manifest saved → %s  (%d stations)", manifest_path, len(stations_df))

    # ── 2. Optionally filter to a single station ──────────────────────────────
    if args.station:
        mask = stations_df["station_reference"] == str(args.station)
        if mask.sum() == 0:
            log.error("Station reference '%s' not found in manifest.", args.station)
            sys.exit(1)
        stations_df = stations_df[mask]
        log.info("Single-station mode: %s", args.station)

    if args.dry_run:
        print(stations_df[["station_reference", "label", "river_name", "lat", "long"]]
              .to_string(index=False))
        log.info("Dry run complete — %d stations listed.", len(stations_df))
        return

    # ── 3. Download readings ──────────────────────────────────────────────────
    log.info(
        "Downloading %d stations  [%s → %s]  workers=%d",
        len(stations_df), args.start, args.end, args.workers,
    )

    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_station,
                make_session(),     # each thread gets its own session
                row["station_reference"],
                row["notation"],
                row["label"],
                args.start,
                args.end,
                args.force,
            ): row["station_reference"]
            for _, row in stations_df.iterrows()
        }

        with tqdm(total=len(futures), desc="Stations", unit=" stn") as pbar:
            for future in as_completed(futures):
                ref, status = future.result()
                results[ref] = status
                if status.startswith("error"):
                    log.warning("%-12s  %s", ref, status)
                pbar.set_postfix_str(status)
                pbar.update(1)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(results.values())
    log.info(
        "Done. ok=%d  skipped=%d  no_measure=%d  empty=%d  errors=%d",
        counts["ok"],
        counts["skipped"],
        counts["no_measure"],
        counts["ok_empty"],
        sum(v for k, v in counts.items() if k.startswith("error")),
    )

    # Write a per-station status log alongside the manifest
    status_df = pd.DataFrame(
        [{"station_reference": k, "status": v} for k, v in results.items()]
    )
    status_df.to_parquet(RAW_DIR / "ingest_status.parquet", index=False)
    log.info("Status log → %s", RAW_DIR / "ingest_status.parquet")


if __name__ == "__main__":
    main()
