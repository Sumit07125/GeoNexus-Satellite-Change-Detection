"""Run isolated, read-only Sentinel-1 metadata queries against Earth Engine.

Run from Backend with its configured Python environment. This does not download
imagery or invoke the Geo-Nexus model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env", override=True)

import ee  # noqa: E402
import gee_service  # noqa: E402


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="Optional GEE project ID override; otherwise use Backend/.env")
    parser.add_argument("--t1", default="2021-03-15")
    parser.add_argument("--t2", default="2024-03-15")
    parser.add_argument("--window-days", type=int, default=15)
    parser.add_argument("--max-cloud", type=float, default=30.0, help="Printed for request parity; S1 queries do not use cloud filtering")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        default=[72.8432, 19.1113, 72.8527, 19.1203])
    return parser.parse_args()


def _dates(target: str, window_days: int):
    date = dt.date.fromisoformat(target)
    start = date - dt.timedelta(days=window_days)
    last_inclusive = date + dt.timedelta(days=window_days)
    end_exclusive = last_inclusive + dt.timedelta(days=1)
    return start.isoformat(), last_inclusive.isoformat(), end_exclusive.isoformat()


def _safe_failure(stage: str, error: Exception):
    print(f"FAIL [{stage}] {type(error).__name__}: {gee_service._safe_error_reason(error)}")


def _query_window(label: str, target: str, args, bbox):
    start, inclusive, end = _dates(target, args.window_days)
    print(f"\n{label}: target={target}; inclusive={start}..{inclusive}; filterDate end_exclusive={end}")
    result = {"ok": False, "orbits": None}

    try:
        base = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(bbox).filterDate(start, end)
        print(f"PASS [{label.lower()}_collection_build] COPERNICUS/S1_GRD bounds/date")
    except Exception as error:
        _safe_failure(f"s1_{label.lower()}_collection_build", error)
        return result

    try:
        base_count = base.size().getInfo()
        if isinstance(base_count, bool) or not isinstance(base_count, (int, float)) or int(base_count) != base_count or base_count < 0:
            raise ValueError("Invalid initial scene count response")
        base_count = int(base_count)
        print(f"PASS [s1_{label.lower()}_base_count] count={base_count}")
    except Exception as error:
        _safe_failure(f"s1_{label.lower()}_base_count (base.size().getInfo())", error)
        return result

    try:
        eligible = (
            base.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        )
        print(f"PASS [s1_{label.lower()}_filter_build] IW, descending, VV, VH")
    except Exception as error:
        _safe_failure(f"s1_{label.lower()}_filter_build", error)
        return result

    try:
        eligible_count = eligible.size().getInfo()
        if isinstance(eligible_count, bool) or not isinstance(eligible_count, (int, float)) or int(eligible_count) != eligible_count or eligible_count < 0:
            raise ValueError("Invalid eligible scene count response")
        eligible_count = int(eligible_count)
        print(f"PASS [s1_{label.lower()}_filtered_count] count={eligible_count}")
    except Exception as error:
        _safe_failure(f"s1_{label.lower()}_filtered_count (eligible.size().getInfo())", error)
        return result

    try:
        histogram = eligible.aggregate_histogram("relativeOrbitNumber_start").getInfo()
        print(f"PASS [s1_{label.lower()}_orbit_histogram] response={histogram!r}")
    except Exception as error:
        _safe_failure(f"s1_{label.lower()}_orbit_histogram (aggregate_histogram.getInfo())", error)
        # Diagnostic only: probe array retrieval to isolate histogram-specific failure.
        try:
            values = eligible.aggregate_array("relativeOrbitNumber_start").getInfo()
            print(f"PASS [s1_{label.lower()}_orbit_array_probe] response={values!r}; diagnostic only, production method unchanged")
        except Exception as array_error:
            _safe_failure(f"s1_{label.lower()}_orbit_array_probe (aggregate_array.getInfo())", array_error)
        return result

    try:
        if not isinstance(histogram, dict):
            raise ValueError("Orbit histogram is not a dictionary")
        orbit_counts = {}
        for raw_orbit, raw_count in histogram.items():
            orbit = gee_service._parse_integral_number(raw_orbit, "relative orbit ID")
            count = gee_service._parse_integral_number(raw_count, f"scene count for orbit {orbit}")
            if not 1 <= orbit <= 175:
                raise ValueError(f"Invalid orbit ID: {raw_orbit!r}")
            if orbit in orbit_counts:
                raise ValueError(f"Duplicate normalized orbit ID: {orbit}")
            if count <= 0:
                raise ValueError(f"Invalid orbit scene count: {raw_count!r}")
            orbit_counts[orbit] = count
        if eligible_count == 0 and orbit_counts:
            raise ValueError("Histogram has orbits while eligible count is zero")
        if eligible_count > 0 and not orbit_counts:
            raise ValueError("Histogram is empty despite eligible scenes")
        if sum(orbit_counts.values()) != eligible_count:
            raise ValueError("Histogram scene counts differ from eligible count")
        print(f"PASS [s1_{label.lower()}_orbit_metadata_parse] orbit_counts={orbit_counts}")
        result = {"ok": True, "orbits": orbit_counts}
    except Exception as error:
        _safe_failure(f"s1_{label.lower()}_orbit_metadata_parse", error)
    return result


def main():
    args = _args()
    if args.window_days < 0 or args.window_days > 90:
        raise SystemExit("--window-days must be between 0 and 90")
    if args.project:
        os.environ["GEE_PROJECT_ID"] = args.project.strip()
    lon_min, lat_min, lon_max, lat_max = args.bbox
    if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
        raise SystemExit("--bbox must be ordered longitude/latitude bounds in valid ranges")

    print(f"Project configuration source: {'command line override' if args.project else 'Backend/.env'}; project value redacted")
    print(f"Request parameters: window_days={args.window_days}, max_cloud_pct={args.max_cloud}, bbox={args.bbox}")
    try:
        gee_service.init_ee()
        print("PASS [gee_initialize] Earth Engine client initialized")
    except Exception as error:
        _safe_failure("gee_initialize", error)
        return 2
    try:
        value = ee.Number(1).getInfo()
        print(f"PASS [gee_smoke_request] ee.Number(1).getInfo()={value}")
    except Exception as error:
        _safe_failure("gee_smoke_request", error)
        return 2

    bbox = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    t1 = _query_window("T1", args.t1, args, bbox)
    t2 = _query_window("T2", args.t2, args, bbox)
    if t1["ok"] and t2["ok"]:
        common = sorted(set(t1["orbits"]).intersection(t2["orbits"]))
        print(f"\nPASS [s1_common_orbit_intersection] T1={t1['orbits']} T2={t2['orbits']} common={common}")
        return 0
    print("\nFAIL [s1_common_orbit_intersection] not computed because one or both metadata queries failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
