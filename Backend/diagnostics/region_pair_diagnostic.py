"""Reproducible, local-only operational diagnostic for configured Maharashtra pairs.

This CLI is intentionally separate from public Flask endpoints and prediction history.
It requires an admin token configured by the operator and a SHA-256 allow-list in the
diagnostic JSON configuration. It never substitutes S-CVA for failed AI inference.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import random
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def _validate_config(config: dict) -> tuple[dict, list[dict]]:
    if not isinstance(config, dict) or not isinstance(config.get("aoi_bbox"), dict):
        raise ValueError("Config must contain an aoi_bbox object.")
    aoi = config["aoi_bbox"]
    required = ("lon_min", "lat_min", "lon_max", "lat_max")
    try:
        aoi = {key: float(aoi[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("aoi_bbox must contain finite lon_min, lat_min, lon_max, lat_max values.") from exc
    if not all(math.isfinite(value) for value in aoi.values()):
        raise ValueError("aoi_bbox coordinates must be finite.")
    if not (-180 <= aoi["lon_min"] < aoi["lon_max"] <= 180 and -90 <= aoi["lat_min"] < aoi["lat_max"] <= 90):
        raise ValueError("aoi_bbox must be an ordered geographic bbox.")
    return aoi, []


def generate_candidates(config: dict, count: int, seed: int) -> list[dict]:
    """Generate reproducible region/date candidates inside the configured AOI bbox."""
    aoi, _ = _validate_config(config)
    size_km = float(config.get("region_size_km", 2.0))
    window_days = int(config.get("window_days", 15))
    max_cloud = int(config.get("max_cloud", 30))
    t1_dates = [dt.date.fromisoformat(value) for value in config.get("t1_dates", [])]
    t2_dates = [dt.date.fromisoformat(value) for value in config.get("t2_dates", [])]
    date_pairs = [(a.isoformat(), b.isoformat()) for a in t1_dates for b in t2_dates if a < b]
    if size_km <= 0 or not date_pairs:
        raise ValueError("Set positive region_size_km and ordered t1_dates/t2_dates in the config.")
    if not 1 <= window_days <= 90 or not 0 <= max_cloud <= 100:
        raise ValueError("window_days must be in [1,90] and max_cloud in [0,100].")

    rng = random.Random(seed)
    candidates = []
    half_lat = (size_km / 2) / 110.574
    for index in range(count):
        lat_lo, lat_hi = aoi["lat_min"] + half_lat, aoi["lat_max"] - half_lat
        if lat_lo >= lat_hi:
            raise ValueError("Configured AOI is too small for region_size_km.")
        lat = rng.uniform(lat_lo, lat_hi)
        half_lon = (size_km / 2) / (111.32 * max(0.1, math.cos(math.radians(lat))))
        lon_lo, lon_hi = aoi["lon_min"] + half_lon, aoi["lon_max"] - half_lon
        if lon_lo >= lon_hi:
            raise ValueError("Configured AOI is too narrow for region_size_km.")
        lon = rng.uniform(lon_lo, lon_hi)
        date_t1, date_t2 = rng.choice(date_pairs)
        candidates.append({
            "candidate_id": f"candidate_{index + 1:03d}",
            "bbox": {"lon_min": lon - half_lon, "lat_min": lat - half_lat, "lon_max": lon + half_lon, "lat_max": lat + half_lat},
            "date_t1": date_t1,
            "date_t2": date_t2,
            "window_days": window_days,
            "max_cloud": max_cloud,
        })
    return candidates


def _load_candidates(config: dict, count: int, seed: int) -> list[dict]:
    aoi, _ = _validate_config(config)
    configured = config.get("candidates")
    candidates = configured[:count] if isinstance(configured, list) else generate_candidates(config, count, seed)
    from app import _validate_payload

    validated = []
    for index, item in enumerate(candidates):
        candidate = dict(item)
        candidate["bbox"] = dict(candidate.get("bbox") or {})
        try:
            candidate["bbox"] = {key: float(candidate["bbox"][key]) for key in ("lon_min", "lat_min", "lon_max", "lat_max")}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Candidate {index + 1} has invalid bbox coordinates.") from exc
        if not all(math.isfinite(value) for value in candidate["bbox"].values()):
            raise ValueError(f"Candidate {index + 1} has non-finite bbox coordinates.")
        if not (aoi["lon_min"] <= float(candidate["bbox"]["lon_min"]) < float(candidate["bbox"]["lon_max"]) <= aoi["lon_max"] and aoi["lat_min"] <= float(candidate["bbox"]["lat_min"]) < float(candidate["bbox"]["lat_max"]) <= aoi["lat_max"]):
            raise ValueError(f"Candidate {index + 1} bbox falls outside the configured AOI.")
        payload = {
            "bbox": candidate["bbox"], "date_t1": candidate.get("date_t1"), "date_t2": candidate.get("date_t2"),
            "window_days": candidate.get("window_days", config.get("window_days", 15)),
            "max_cloud": candidate.get("max_cloud", config.get("max_cloud", 30)),
        }
        _validate_payload(payload, "geonexus_ai")
        bbox = payload["bbox"]
        lat_mid = (bbox["lat_min"] + bbox["lat_max"]) / 2
        width_m = (bbox["lon_max"] - bbox["lon_min"]) * 111320 * math.cos(math.radians(lat_mid))
        height_m = (bbox["lat_max"] - bbox["lat_min"]) * 110574
        if width_m < 1280 or height_m < 1280:
            raise ValueError(f"Candidate {index + 1} is smaller than the model's 128×128 10 m minimum grid.")
        if width_m * height_m / 100 <= 400_000:
            pass
        else:
            raise ValueError(f"Candidate {index + 1} exceeds the single-request 400,000-pixel model limit.")
        candidate.update(payload)
        candidate["candidate_id"] = candidate.get("candidate_id", f"candidate_{index + 1:03d}")
        validated.append(candidate)
    return validated


def _write_b64(value: str | None, path: Path) -> str | None:
    if not value:
        return None
    raw = value.split(",", 1)[-1]
    path.write_bytes(base64.b64decode(raw, validate=True))
    return str(path)


def _json_safe(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_diagnostic(config_path: Path, output_dir: Path, count: int = 30, seed: int = 314159, run_spectral: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed_hash = str(config.get("admin_token_sha256", "")).strip().lower()
    supplied_token = os.environ.get("GEONEXUS_DIAGNOSTIC_ADMIN_TOKEN", "")
    if not allowed_hash or len(allowed_hash) != 64 or not supplied_token:
        raise PermissionError("Configure GEONEXUS_DIAGNOSTIC_ADMIN_TOKEN and admin_token_sha256 before running this local admin diagnostic.")
    actual_hash = hashlib.sha256(supplied_token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(actual_hash, allowed_hash):
        raise PermissionError("Diagnostic admin authentication failed.")

    candidates = _load_candidates(config, count, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GEE_PROJECT_ID", "").strip():
        provider_available = False
        provider_message = "GEE_PROJECT_ID is not configured; this AI diagnostic requires real Sentinel-1 and Q inputs from Google Earth Engine."
    else:
        provider_available = True
        provider_message = None

    report = {
        "report_type": "operational_diagnostic_not_research_evaluation",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "random_seed": seed,
        "candidate_count": len(candidates),
        "aoi_bbox": config["aoi_bbox"],
        "model_version": "3.2.0",
        "provider_requirement": "Google Earth Engine with genuine Sentinel-1 and trained Q inputs",
        "spectral_requested": bool(run_spectral),
        "runs": [],
    }
    if not provider_available:
        for candidate in candidates:
            report["runs"].append({**candidate, "status": "acquisition_failed", "error_code": "provider_unavailable", "error": provider_message, "engine": "geonexus_ai"})
        report["summary"] = {"candidate_count": len(candidates), "ai_success": 0, "ai_failed": len(candidates), "spectral_success": 0, "spectral_failed": 0, "accuracy_metrics": None}
        (output_dir / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    from app import _validate_imagery, _validate_payload
    from gee_service import fetch_imagery
    from inference import run_inference

    spectral_runner = None
    if run_spectral:
        from spectral_cd import run_spectral_inference as spectral_runner

    for index, candidate in enumerate(candidates, 1):
        record = {**candidate, "engine": "geonexus_ai", "status": "pending", "timings_seconds": {}}
        run_dir = output_dir / candidate["candidate_id"]
        run_dir.mkdir(exist_ok=True)
        try:
            payload = _validate_payload({key: candidate[key] for key in ("bbox", "date_t1", "date_t2", "window_days", "max_cloud")}, "geonexus_ai")
            start = time.perf_counter()
            imagery = fetch_imagery(
                lon_min=payload["bbox"]["lon_min"], lat_min=payload["bbox"]["lat_min"],
                lon_max=payload["bbox"]["lon_max"], lat_max=payload["bbox"]["lat_max"],
                date_t1=payload["date_t1"], date_t2=payload["date_t2"],
                window_days=payload["window_days"], max_cloud=payload["max_cloud"],
            )
            record["timings_seconds"]["acquisition"] = round(time.perf_counter() - start, 3)
            record["provider"] = (imagery.get("meta") or {}).get("provider", "Google Earth Engine")
            record["acquisition_metadata"] = imagery.get("meta", {})
        except Exception as exc:
            record["status"] = "acquisition_failed"
            record["error_code"] = getattr(exc, "code", "acquisition_failed")
            record["error_type"] = type(exc).__name__
            from app import _safe_provider_detail
            record["error"] = _safe_provider_detail(exc)
            record["diagnostic"] = getattr(exc, "details", {})
            report["runs"].append(record)
            print(f"[{index}/{len(candidates)}] {candidate['candidate_id']}: {record['status']}")
            continue

        try:
            _validate_imagery(imagery, require_sar=True)
            start = time.perf_counter()
            ai = run_inference(imagery["t1_tensor"], imagery["t2_tensor"], imagery["t2_rgb_b64"])
            record["timings_seconds"]["ai_inference"] = round(time.perf_counter() - start, 3)
            record["ai"] = {"engine": "geonexus_ai", "model_version": ai.get("model_version"), "algorithm": "MH-DAPT-CD v3.2 Few-Shot k=30", "threshold": ai.get("threshold", ai.get("stats", {}).get("threshold")), "threshold_source": ai.get("threshold_source"), "stats": ai.get("stats")}
            for name, key in (("ai_mask", "binary_mask_b64"), ("ai_probability", "change_prob_b64"), ("ai_overlay", "overlay_b64")):
                record["ai"][f"{name}_path"] = _write_b64(ai.get(key), run_dir / f"{name}.png")
            record["status"] = "ai_success"
        except Exception as exc:
            code = getattr(exc, "code", "")
            record["status"] = "input_invalid" if code in ("invalid_quality_channel", "invalid_input_channels", "area_too_small", "invalid_input_grid", "missing_sar_channels", "missing_quality_channel") else "inference_failed"
            record["error_code"] = getattr(exc, "code", "inference_failed")
            record["error_type"] = type(exc).__name__
            from app import _safe_provider_detail
            record["error"] = _safe_provider_detail(exc)
            record["diagnostic"] = getattr(exc, "details", {})
        if spectral_runner:
            try:
                start = time.perf_counter()
                scva = spectral_runner(imagery["t1_tensor"], imagery["t2_tensor"], imagery["t2_rgb_b64"])
                record["timings_seconds"]["spectral_inference"] = round(time.perf_counter() - start, 3)
                record["spectral_scva"] = {"engine": "spectral_scva", "algorithm": "Spectral Change Vector Analysis (NDVI, NDWI, NDBI, BSI)", "threshold": scva.get("threshold", scva.get("stats", {}).get("threshold")), "threshold_source": scva.get("threshold_source"), "stats": scva.get("stats")}
                for name, key in (("spectral_mask", "binary_mask_b64"), ("spectral_score", "change_prob_b64"), ("spectral_overlay", "overlay_b64")):
                    record["spectral_scva"][f"{name}_path"] = _write_b64(scva.get(key), run_dir / f"{name}.png")
                record["spectral_status"] = "success"
            except Exception as exc:
                record["spectral_status"] = "failed"
                record["spectral_error_type"] = type(exc).__name__
                from app import _safe_provider_detail
                record["spectral_error"] = _safe_provider_detail(exc)
        report["runs"].append(record)
        print(f"[{index}/{len(candidates)}] {candidate['candidate_id']}: {record['status']}")

    ai_success = sum(row["status"] == "ai_success" for row in report["runs"])
    spectral_success = sum(row.get("spectral_status") == "success" for row in report["runs"])
    spectral_failed = sum(row.get("spectral_status") == "failed" for row in report["runs"])
    report["summary"] = {
        "candidate_count": len(candidates), "ai_success": ai_success, "ai_failed": len(candidates) - ai_success,
        "spectral_success": spectral_success, "spectral_failed": spectral_failed,
        "accuracy_metrics": None,
        "note": "Operational acquisition and inference diagnostics only; no ground-truth labels or research evaluation metrics were used.",
    }
    (output_dir / "diagnostic_report.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Admin-protected JSON configuration with AOI and dates/candidates")
    parser.add_argument("--output", type=Path, required=True, help="New or existing output directory for report and result PNGs")
    parser.add_argument("--count", type=int, default=30, help="Number of candidate pairs (default: 30)")
    parser.add_argument("--seed", type=int, default=314159, help="Reproducible candidate seed")
    parser.add_argument("--run-spectral", action="store_true", help="Also run independent S-CVA on AI-compatible acquired imagery")
    args = parser.parse_args()
    if args.count < 1 or args.count > 1000:
        parser.error("--count must be between 1 and 1000")
    try:
        report = run_diagnostic(args.config, args.output, args.count, args.seed, args.run_spectral)
    except (ValueError, PermissionError, OSError, json.JSONDecodeError) as exc:
        print(f"Diagnostic not run: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output / 'diagnostic_report.json'}; AI success {report['summary']['ai_success']}/{report['summary']['candidate_count']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
