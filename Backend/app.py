"""
app.py — Geo-Nexus Flask API
Uses a background job queue so long GEE + inference calls
don't drop the HTTP connection.
"""

import os
import time
import uuid
import datetime
import threading
import traceback
import math
import re
import json
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(override=True)


def _application_build_id():
    configured = os.environ.get("GEONEXUS_BUILD_ID", "").strip()
    if configured:
        return re.sub(r"[^A-Za-z0-9._-]", "", configured)[:80] or "unknown"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=1, check=True,
        )
        revision = completed.stdout.strip()[:40] or "unknown"
        state = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=1, check=True,
        )
        return f"{revision}-dirty" if state.stdout.strip() else revision
    except Exception:
        return "unknown"


BUILD_ID = _application_build_id()

from gee_config import get_gee_configuration

_startup_gee = get_gee_configuration()
print(f"[ENV] Google Earth Engine readiness: {_startup_gee['status']}")
print(f"[BOOT] Geo-Nexus build: {BUILD_ID}")

app = Flask(__name__)

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS(app, resources={r"/api/*": {
    "origins": [origin.strip() for origin in os.environ.get("CORS_ORIGINS", os.environ.get("FRONTEND_URL", "http://localhost:5173")).split(",") if origin.strip()],
    "methods": ["GET", "POST", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type"],
}})

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { "status": "pending|running|done|error", "progress": str, "result": dict, "error": str } }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class PipelineError(Exception):
    """Safe, structured pipeline failure suitable for job polling clients."""
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


INPUT_CHANNEL_ORDER = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "NDVI", "NDBI", "VV", "VH", "VH_VV", "Q"]
QUALITY_STD_MIN = 0.5  # training applies its heterogeneity acceptance test to raw n_clear, before Q clipping
AI_MIN_DIMENSION_M = 1280


def _bbox_dimensions_m(bbox):
    lat_mid = math.radians((bbox["lat_min"] + bbox["lat_max"]) / 2)
    lat_scale = 111132.92 - 559.82 * math.cos(2 * lat_mid) + 1.175 * math.cos(4 * lat_mid) - 0.0023 * math.cos(6 * lat_mid)
    lon_scale = 111412.84 * math.cos(lat_mid) - 93.5 * math.cos(3 * lat_mid) + 0.118 * math.cos(5 * lat_mid)
    return {
        "width": abs(bbox["lon_max"] - bbox["lon_min"]) * lon_scale,
        "height": abs(bbox["lat_max"] - bbox["lat_min"]) * lat_scale,
    }


def _input_channel_diagnostics(imagery):
    """Build bounded per-channel summaries; never include raster values or arrays."""
    import numpy as np

    meta = imagery.get("meta") or {}
    provider = meta.get("provider", "unknown")
    preprocessing = meta.get("preprocessing_id", "unknown")
    rows = []
    for date_key, tensor_key in (("t1", "t1_tensor"), ("t2", "t2_tensor")):
        image = imagery.get(tensor_key)
        date = meta.get(f"{date_key}_date") or meta.get(f"{date_key}_datetime") or "unknown"
        if image is None or getattr(image, "ndim", 0) != 3:
            rows.append({"date": date, "provider": provider, "preprocessing_id": preprocessing, "error": "missing_or_invalid_tensor"})
            continue
        for index, name in enumerate(INPUT_CHANNEL_ORDER[:image.shape[0]]):
            values = np.asarray(image[index])
            finite = np.isfinite(values)
            finite_values = values[finite]
            if index < 11:
                valid = finite & (values > 0)
            elif index < 13:
                valid = finite & (np.abs(values) <= 1.0)
            elif index < 16:
                valid = finite & (values != 0)
            else:
                valid = finite & (values >= 0) & (values <= 1) & (values > 0)
            if index in (13, 14, 15) and not meta.get("s1_used", False):
                feature_status = "placeholder" if meta.get("sar_channels_available") is False else "unavailable"
            elif index == 16 and meta.get("quality_channel_status") in ("placeholder", "unavailable"):
                feature_status = meta["quality_channel_status"]
            elif provider == "unknown":
                feature_status = "unavailable"
            else:
                feature_status = "real"
            row = {
                "date": date,
                "provider": provider,
                "preprocessing_id": preprocessing,
                "channel_index": index,
                "channel": name,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "finite_fraction": float(finite.mean()),
                "valid_pixel_fraction": float(valid.mean()),
                "feature_status": feature_status,
            }
            if index == 16:
                row["raw_n_clear_statistics"] = meta.get(f"{date_key}_quality_raw_stats")
            if finite_values.size:
                row.update({
                    "min": float(finite_values.min()),
                    "max": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "std": float(finite_values.std()),
                })
            else:
                row.update({"min": None, "max": None, "mean": None, "std": None})
            rows.append(row)
    return rows


def _validate_payload(body, engine="geonexus_ai"):
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    bbox = body.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError("bbox must contain lon_min, lat_min, lon_max, and lat_max.")
    try:
        coords = {key: float(bbox[key]) for key in ("lon_min", "lat_min", "lon_max", "lat_max")}
    except (KeyError, TypeError, ValueError):
        raise ValueError("Bounding box coordinates must be finite numbers.")
    if not all(math.isfinite(value) for value in coords.values()):
        raise ValueError("Bounding box coordinates must be finite numbers.")
    if not (-180 <= coords["lon_min"] < coords["lon_max"] <= 180 and -90 <= coords["lat_min"] < coords["lat_max"] <= 90):
        raise ValueError("Bounding box must be ordered and within longitude/latitude limits.")
    dates = []
    for key in ("date_t1", "date_t2"):
        value = body.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"{key} must be a valid YYYY-MM-DD date.")
        try:
            dates.append(datetime.date.fromisoformat(value))
        except ValueError:
            raise ValueError(f"{key} must be a valid YYYY-MM-DD date.")
    if dates[0] >= dates[1]:
        raise ValueError("date_t1 must be earlier than date_t2.")
    for key, default, low, high in (("max_cloud", 30, 0, 100), ("window_days", 15, 1, 90)):
        raw_value = body.get(key, default)
        try:
            value = int(raw_value)
        except (ValueError, TypeError):
            raise ValueError(f"{key} must be an integer from {low} to {high}.")
        if isinstance(raw_value, float) and not raw_value.is_integer():
            raise ValueError(f"{key} must be an integer from {low} to {high}.")
        if value < low or value > high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if body.get("threshold") is not None:
        try:
            threshold = float(body["threshold"])
        except (ValueError, TypeError):
            raise ValueError("threshold must be a finite number.")
        upper = 1 if engine == "geonexus_ai" else 10
        if not math.isfinite(threshold) or not 0 < threshold < upper:
            raise ValueError(f"threshold must be greater than 0 and less than {upper} for this analysis engine.")
    body["bbox"] = coords
    return body


def _estimated_area_stats(stats, bbox):
    """Estimate output pixel area from the requested bbox and returned raster shape."""
    shape = stats.get("output_size") or []
    if len(shape) != 2 or not all(isinstance(v, (int, float)) and v > 0 for v in shape):
        return
    h, w = shape
    lat_mid = (bbox["lat_min"] + bbox["lat_max"]) / 2
    width_m = (bbox["lon_max"] - bbox["lon_min"]) * 111320 * math.cos(math.radians(lat_mid))
    height_m = (bbox["lat_max"] - bbox["lat_min"]) * 110574
    px_ha = (width_m * height_m / (h * w)) / 10000
    stats["changed_area_ha"] = round(int(stats.get("changed_pixels", 0)) * px_ha, 2)
    stats["pixel_area_m2_estimate"] = round(px_ha * 10000, 2)
    stats["area_method"] = "Approximate bbox area divided by output raster pixels"


def _provider_error_kind(error):
    message = str(error).lower()
    if any(word in message for word in ("permission", "unauthorized", "authentication", "credentials", "oauth")):
        return "authentication_failed"
    if any(word in message for word in ("no sentinel", "no scene", "empty collection", "no imagery")):
        return "imagery_unavailable"
    return "provider_unavailable"


def _safe_provider_detail(error):
    """Keep useful provider diagnostics while stripping paths, URLs and secret-like values."""
    detail = str(error)
    detail = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", detail)
    detail = re.sub(r"(?i)(token|password|secret|credential|authorization)(\s*[=:]\s*)[^\s,;]+", r"\1\2[redacted]", detail)
    detail = re.sub(r"(?i)https?://\S+", "[provider-url]", detail)
    detail = re.sub(r"(?i)([a-z]:\\|/)(?:[^\s:]+[\\/])+[^\s:]*", "[path]", detail)
    return detail[:240]


def _validate_imagery(imagery, require_sar=False):
    import numpy as np
    t1, t2 = imagery.get("t1_tensor"), imagery.get("t2_tensor")
    if t1 is None or t2 is None or getattr(t1, "ndim", 0) != 3 or getattr(t2, "ndim", 0) != 3:
        raise PipelineError("invalid_input_channels", "The imagery provider did not return usable multi-band arrays.")
    if t1.shape[0] != 17 or t2.shape[0] != 17 or t1.shape != t2.shape:
        raise PipelineError("invalid_input_channels", "Imagery must provide matching 17-channel T1/T2 arrays on the same grid.")
    meta = imagery.get("meta") or {}
    diagnostics = _input_channel_diagnostics(imagery)
    app.logger.info("input_channel_diagnostics=%s", json.dumps(diagnostics, separators=(",", ":")))
    if require_sar and (not meta.get("s1_used", False) or meta.get("sar_channels_available") is False):
        provider = meta.get("provider", "unknown")
        dates = f"T1 {meta.get('t1_date', 'unknown')}, T2 {meta.get('t2_date', 'unknown')}"
        raise PipelineError(
            "missing_sar_channels",
            f"Geo-Nexus AI requires genuine Sentinel-1 VV, VH and VH_VV plus a valid Q feature. {provider} supplied unavailable or placeholder multimodal inputs for {dates}; use Spectral Index Analysis or configure a provider with real Sentinel-1 data.",
            {"provider": provider, "dates": {"t1": meta.get("t1_date"), "t2": meta.get("t2_date")}, "channels": ["VV", "VH", "VH_VV", "Q"]},
        )
    if require_sar:
        if meta.get("quality_channel_status") != "real" or "n_clear / 8" not in str(meta.get("quality_channel", "")):
            raise PipelineError("missing_quality_channel", f"{meta.get('provider', 'The selected provider')} did not supply the trained multi-date Q = clip(n_clear / 8, 0, 1) feature for both dates.", {"provider": meta.get("provider", "unknown"), "channel": "Q", "dates": {"t1": meta.get("t1_date"), "t2": meta.get("t2_date")}})
        if meta.get("pixel_resolution_m") != 10 or not str(meta.get("crs", "")).startswith("EPSG:326"):
            raise PipelineError("invalid_input_grid", "Geo-Nexus AI requires both dates on the same 10 m UTM input grid.", {"provider": meta.get("provider", "unknown"), "crs": meta.get("crs"), "pixel_resolution_m": meta.get("pixel_resolution_m")})
        for count_key in ("t1_scene_count", "t2_scene_count", "t1_s1_scene_count", "t2_s1_scene_count"):
            if not isinstance(meta.get(count_key), (int, float)) or meta[count_key] <= 0:
                date_key = "t1_date" if count_key.startswith("t1") else "t2_date"
                raise PipelineError("imagery_unavailable", f"{meta.get('provider', 'The selected provider')} found no usable {count_key.replace('_scene_count', '').upper()} scenes for {meta.get(date_key, 'the selected date')}.", {"provider": meta.get("provider", "unknown"), "date": meta.get(date_key), "scene_count_field": count_key})
    if require_sar:
        provider = meta.get("provider", "unknown")
        for label, image in (("T1", t1), ("T2", t2)):
            quality = image[16]
            date = meta.get(f"{label.lower()}_date", "unknown")
            raw_quality_stats = meta.get(f"{label.lower()}_quality_raw_stats") or {}
            finite_quality = quality[np.isfinite(quality)]
            quality_std = float(finite_quality.std()) if finite_quality.size else None
            details = {
                "provider": provider, "date": date, "channel": "Q",
                "label": label,
                "preprocessing_id": meta.get("preprocessing_id", "unknown"),
                "finite_fraction": float(np.isfinite(quality).mean()),
                "positive_quality_fraction": float((quality > 0).mean()),
                "min": float(finite_quality.min()) if finite_quality.size else None,
                "max": float(finite_quality.max()) if finite_quality.size else None,
                "std": quality_std, "raw_n_clear_std": raw_quality_stats.get("std"),
                "raw_n_clear_valid_pixel_fraction": raw_quality_stats.get("valid_pixel_fraction"),
                "raw_min": raw_quality_stats.get("min"),
                "raw_max": raw_quality_stats.get("max"),
                "valid_pixel_fraction": raw_quality_stats.get("valid_pixel_fraction"),
                "expected_range": [0.0, 1.0], "minimum_raw_n_clear_std": QUALITY_STD_MIN,
                # Enhanced diagnostics from _build_tensor
                "percentiles_p5_p25_p50_p75_p95": raw_quality_stats.get("percentiles_p5_p25_p50_p75_p95"),
                "value_counts": raw_quality_stats.get("value_counts"),
            }
            if not np.isfinite(quality).all():
                raise PipelineError("invalid_quality_channel", f"{label} Q input from {provider} on {date} contains NaN or infinite values; the trained Q channel must be finite and in [0, 1].", details)
            if float(quality.min()) < 0 or float(quality.max()) > 1:
                raise PipelineError("invalid_quality_channel", f"{label} Q input from {provider} on {date} is outside the trained [0, 1] range; check n_clear/8 encoding.", details)
            if float(quality.max()) == 0:
                raise PipelineError("invalid_quality_channel", f"{label} Q input from {provider} on {date} is all zero, indicating no valid clear observations or invalid masking.", details)
            if float(np.mean(quality > 0)) < 0.05:
                raise PipelineError("invalid_quality_channel", f"{label} Q input from {provider} on {date} has under 5% positive clear-observation coverage; check masking, nodata, and image availability.", details)
            raw_valid_fraction = raw_quality_stats.get("valid_pixel_fraction")
            if not isinstance(raw_valid_fraction, (int, float)) or not math.isfinite(raw_valid_fraction) or raw_valid_fraction < 1.0:
                raise PipelineError("invalid_quality_channel", f"{label} raw n_clear from {provider} on {date} contains non-finite or missing pixels before Q encoding.", details)
            raw_min = raw_quality_stats.get("min")
            raw_max = raw_quality_stats.get("max")
            if not isinstance(raw_min, (int, float)) or not isinstance(raw_max, (int, float)) or raw_min < 0 or raw_max > meta.get(f"{label.lower()}_scene_count", 0):
                raise PipelineError("invalid_quality_channel", f"{label} raw n_clear from {provider} on {date} is inconsistent with the acquired Sentinel-2 scene count.", details)
            raw_n_clear_std = raw_quality_stats.get("std")
            if not isinstance(raw_n_clear_std, (int, float)) or not math.isfinite(raw_n_clear_std):
                raise PipelineError("invalid_quality_channel", f"{label} Q input from {provider} on {date} is missing the raw n_clear diagnostic required by the trained preprocessing contract.", details)
            if raw_n_clear_std <= QUALITY_STD_MIN:
                raise PipelineError("invalid_quality_channel", f"{label} raw n_clear from {provider} on {date} does not meet the trained composite heterogeneity check (std(n_clear) must exceed {QUALITY_STD_MIN:.1f} observations). The model input Q is n_clear/8 clipped to [0,1]; check the selected date window and cloud-score matching.", details)
    if min(t1.shape[1:]) < 128:
        raise PipelineError("area_too_small", "The AI model needs at least 128×128 pixels at 10 m. Choose an area at least about 1.3 km in each dimension.")
    if not np.isfinite(t1).all() or not np.isfinite(t2).all():
        raise PipelineError("invalid_input_channels", "The imagery provider returned non-finite channel values.")
    for label, image in (("T1", t1), ("T2", t2)):
        optical_coverage = np.mean(image[:11] > 0, axis=(1, 2))
        if np.any(optical_coverage < 0.05):
            raise PipelineError("invalid_input_channels", f"{label} imagery is missing one or more Sentinel-2 optical bands required for analysis.")
        if np.any((image[:11] < 0) | (image[:11] > 32767)):
            raise PipelineError("invalid_input_channels", f"{label} Sentinel-2 bands are outside the trained reflectance-DN storage range.", {"provider": meta.get("provider", "unknown"), "date": meta.get(f"{label.lower()}_date"), "channels": INPUT_CHANNEL_ORDER[:11]})
        if require_sar and np.any(np.abs(image[11:13]) > 1.0001):
            raise PipelineError("invalid_input_channels", f"{label} NDVI/NDBI are outside their trained floating-point [-1, 1] range.", {"provider": meta.get("provider", "unknown"), "date": meta.get(f"{label.lower()}_date"), "channels": ["NDVI", "NDBI"]})
        if require_sar and np.any(np.mean(image[13:16] != 0, axis=(1, 2)) < 0.05):
            raise PipelineError("missing_sar_channels", f"{label} imagery is missing one or more real Sentinel-1 channels required by the model.")
        if require_sar and np.any((image[13:16] < -5000) | (image[13:16] > 1000)):
            raise PipelineError("invalid_input_channels", f"{label} Sentinel-1 dB × 100 channels are outside the trained storage range.", {"provider": meta.get("provider", "unknown"), "date": meta.get(f"{label.lower()}_date"), "channels": INPUT_CHANNEL_ORDER[13:16]})
    valid = (t1[2] > 100) & (t1[6] > 100) & (t2[2] > 100) & (t2[6] > 100)
    if float(valid.mean()) < 0.05:
        raise PipelineError("nodata", "Imagery contains too little valid optical coverage for analysis. Choose a smaller or clearer region/date window.")


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _trim_jobs():
    with _jobs_lock:
        if len(_jobs) <= 30:
            return
        terminal = sorted(
            ((job.get("created_at", 0), job_id) for job_id, job in _jobs.items() if job.get("status") in ("done", "error")),
        )
        while len(_jobs) > 20 and terminal:
            _, job_id = terminal.pop(0)
            _jobs.pop(job_id, None)


def _run_job(job_id: str, payload: dict):
    """Background thread for the trained AI engine. Only GEE supplies AI inputs."""
    try:
        try:
            from gee_service import fetch_imagery as gee_fetch
        except Exception as provider_import_error:
            raise PipelineError("gee_api_unavailable", "The Google Earth Engine client is unavailable in this backend environment.") from provider_import_error
        try:
            from inference import run_inference
        except Exception as model_import_error:
            raise PipelineError("model_unavailable", "The Geo-Nexus model runtime could not be loaded. Check the backend model dependencies and retry.") from model_import_error

        bbox        = payload["bbox"]
        date_t1     = payload["date_t1"]
        date_t2     = payload["date_t2"]
        loc_name    = payload.get("location_name", "Selected Area")
        window_days = int(payload.get("window_days", int(os.environ.get("DATE_SEARCH_WINDOW_DAYS", 15))))
        max_cloud   = int(payload.get("max_cloud",   int(os.environ.get("MAX_CLOUD_PCT", 30))))
        threshold   = payload.get("threshold", None)
        thumb_size  = int(os.environ.get("THUMB_DIM", 512))

        if threshold is not None:
            threshold = float(threshold)

        # ── Step 1: Acquire the real multimodal inputs from GEE ───────────────
        _update_job(job_id, status="running",
                    progress="Checking Earth Engine access, then acquiring Sentinel-2 and Sentinel-1 imagery…")
        app.logger.info(f"[{job_id}] Satellite fetch for '{loc_name}'")

        provider_attempts = []
        meta = {
            "provider": "Google Earth Engine",
            "t1_date": date_t1,
            "t2_date": date_t2,
            "provider_attempts": provider_attempts,
        }
        _update_job(job_id, progress="Acquiring T1/T2 Sentinel-2 and Sentinel-1 composites…")
        try:
            imagery = gee_fetch(
                lon_min=float(bbox["lon_min"]), lat_min=float(bbox["lat_min"]),
                lon_max=float(bbox["lon_max"]), lat_max=float(bbox["lat_max"]),
                date_t1=date_t1, date_t2=date_t2,
                window_days=window_days, max_cloud=max_cloud,
                thumb_size=thumb_size,
            )
            meta = imagery.get("meta") or meta
            meta.update({"provider": "Google Earth Engine", "t1_date": date_t1, "t2_date": date_t2})
            provider_attempts.append({"provider": "Google Earth Engine", "status": "success"})
            app.logger.info("provider_attempt=%s", json.dumps({"job_id": job_id, "provider": "Google Earth Engine", "status": "success"}))
        except Exception as gee_err:
            from gee_config import classify_gee_error
            if "single-request 10 m model grid limit" in str(gee_err):
                code, message = "area_too_large", "This area is too large for one 10 m AI request. Draw a smaller bbox and retry."
            elif getattr(gee_err, "code", None) and getattr(gee_err, "message", None):
                code, message = gee_err.code, gee_err.message
            else:
                code, message = classify_gee_error(gee_err)
            diagnostic = getattr(gee_err, "details", {}) or {}
            diagnostic = {
                **diagnostic,
                "request_id": job_id,
                "provider_readiness": get_gee_configuration().get("status", "unknown"),
                "request_parameters": {
                    "bbox": bbox,
                    "date_t1": date_t1,
                    "date_t2": date_t2,
                    "window_days": window_days,
                    "max_cloud_pct": max_cloud,
                },
            }
            attempt = {
                "provider": "Google Earth Engine", "status": code,
                "stage": diagnostic.get("stage", "acquisition"),
                "error_type": type(gee_err).__name__, "detail": message,
            }
            provider_attempts.append(attempt)
            meta["provider_attempts"] = provider_attempts
            app.logger.error("gee_acquisition_failure=%s", json.dumps({
                "job_id": job_id, "request_id": job_id, **attempt, "diagnostics": diagnostic,
            }, separators=(",", ":")))
            details = diagnostic or {
                "provider": "Google Earth Engine",
                "requested_dates": {"t1": date_t1, "t2": date_t2},
                "window_days": window_days,
                "max_cloud_pct": max_cloud,
            }
            details.setdefault("requested_dates", {"t1": date_t1, "t2": date_t2})
            raise PipelineError(code, message, details) from gee_err

        meta["provider_attempts"] = provider_attempts
        meta["input_channels"] = 17
        meta["input_channel_order"] = INPUT_CHANNEL_ORDER
        meta["ai_input_contract"] = "complete" if meta.get("s1_used", False) else "incomplete"
        _update_job(job_id, progress="Validating the trained 17-channel T1/T2 input contract…")
        _validate_imagery(imagery, require_sar=True)

        _update_job(job_id, progress="Running Geo-Nexus v3.2 checkpoint inference…")
        app.logger.info(f"[{job_id}] Imagery fetched. Running inference…")

        # ── Step 2: Model inference ───────────────────────────────────────────
        try:
            result = run_inference(
                t1_tensor=imagery["t1_tensor"],
                t2_tensor=imagery["t2_tensor"],
                t2_rgb_b64=imagery["t2_rgb_b64"],
                threshold=threshold,
            )
        except Exception as inference_err:
            app.logger.exception("[%s] Geo-Nexus model inference failed", job_id)
            raise PipelineError("inference_failed", "Geo-Nexus AI inference failed. Retry the AI analysis or explicitly choose Spectral Index Analysis.") from inference_err

        _estimated_area_stats(result["stats"], bbox)
        threshold_used = result.get("stats", {}).get("threshold")
        result.update({
            "engine": "geonexus_ai",
            "engine_label": "Geo-Nexus AI Model",
            "algorithm": "MH-DAPT-CD v3.2 Few-Shot k=30",
            "model_version": result.get("model_version", "unknown"),
            "threshold": threshold_used,
            "threshold_source": result.get("threshold_source", "unknown"),
        })

        app.logger.info(f"[{job_id}] Done. Change: {result['stats']['change_pct']}%")

        outcome_result = {
            "success": True,
            **{key: result[key] for key in ("engine", "engine_label", "algorithm", "model_version", "threshold", "threshold_source")},
            "location_name": loc_name,
            "t1_rgb":      f"data:image/png;base64,{imagery['t1_rgb_b64']}",
            "t2_rgb":      f"data:image/png;base64,{imagery['t2_rgb_b64']}",
            "change_mask": f"data:image/png;base64,{result['change_mask_b64']}",
            "binary_mask": f"data:image/png;base64,{result.get('binary_mask_b64', result['change_mask_b64'])}",
            "change_prob": f"data:image/png;base64,{result['change_prob_b64']}",
            "overlay":     f"data:image/png;base64,{result['overlay_b64']}",
            "dynamics":    result.get("dynamics", {}),
            "stats":       result["stats"],
            "meta":        imagery["meta"],
        }

        # ── Permanent SQLite storage persistence ──────────────────────
        try:
            import storage_manager
            saved_record = storage_manager.save_prediction({
                "id": f"pred_{job_id[:8]}",
                "locationName": loc_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "bbox": bbox,
                "dateT1": date_t1,
                "dateT2": date_t2,
                "result": outcome_result,
            })
            outcome_result["saved_id"] = saved_record.get("id")
            app.logger.info(f"[{job_id}] Permanently saved to database as {saved_record.get('id')}")
        except Exception as save_err:
            app.logger.warning(f"[{job_id}] Storage auto-save warning: {save_err}")
            outcome_result["persistence"] = {"saved": False, "error_code": "storage_failed", "message": "Analysis completed, but history could not be saved."}
        else:
            outcome_result["persistence"] = {"saved": True}

        _update_job(job_id, status="done", progress="✅ Complete", result=outcome_result)

    except Exception as e:
        app.logger.error("[%s] AI job failed (%s): %s", job_id, getattr(e, "code", "pipeline_failed"), type(e).__name__)
        code = e.code if isinstance(e, PipelineError) else "pipeline_failed"
        message = str(e) if isinstance(e, PipelineError) else "Geo-Nexus AI analysis failed. Check imagery availability and server logs, then retry."
        failure_meta = {key: meta[key] for key in ("provider", "provider_fallback", "input_channels", "s1_used", "preprocessing_id", "t1_date", "t2_date", "quality_channel", "quality_channel_status") if "meta" in locals() and key in meta}
        failure_meta["provider_attempts"] = locals().get("provider_attempts", [])
        if isinstance(e, PipelineError):
            diag = dict(e.details) if e.details else {}
            # Augment with acquisition context already known at exception time
            diag.setdefault("request_id", job_id)
            if "requested_dates" not in diag:
                diag["requested_dates"] = {"t1": locals().get("date_t1"), "t2": locals().get("date_t2")}
            if "window_days" not in diag:
                diag["window_days"] = locals().get("window_days")
            if "max_cloud_pct" not in diag:
                diag["max_cloud_pct"] = locals().get("max_cloud")
            if "bbox" not in diag:
                diag["bbox"] = locals().get("bbox")
            # Carry over search-window strings from imagery meta if present
            if "meta" in locals() and meta:
                for wk in ("t1_window", "t2_window", "search_windows"):
                    if wk in meta and wk not in diag:
                        diag[wk] = meta[wk]
                if "search_windows" not in diag and "t1_window" in diag and "t2_window" in diag:
                    diag["search_windows"] = {"t1": diag["t1_window"], "t2": diag["t2_window"]}
            failure_meta["diagnostic"] = diag
        _update_job(job_id, status="error", progress="❌ Failed", error=message, error_code=code, meta=failure_meta)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral-index job runner (no ML model required)
# ─────────────────────────────────────────────────────────────────────────────

def _run_spectral_job(job_id: str, payload: dict):
    """Background thread: STAC fetch + multi-index spectral change detection."""
    try:
        from spectral_cd import run_spectral_inference

        bbox        = payload["bbox"]
        date_t1     = payload["date_t1"]
        date_t2     = payload["date_t2"]
        loc_name    = payload.get("location_name", "Selected Area")
        window_days = int(payload.get("window_days", int(os.environ.get("DATE_SEARCH_WINDOW_DAYS", 15))))
        max_cloud   = int(payload.get("max_cloud",   int(os.environ.get("MAX_CLOUD_PCT", 30))))
        threshold   = payload.get("threshold", None)
        thumb_size  = int(os.environ.get("THUMB_DIM", 512))

        if threshold is not None:
            threshold = float(threshold)

        # ── Step 1: Fetch satellite imagery (same pipeline as neural model) ──
        _update_job(job_id, status="running",
                    progress="🛰 Fetching satellite imagery…")
        app.logger.info(f"[spectral/{job_id}] Fetch for '{loc_name}'")

        imagery = None
        provider = None
        gee_failure = None
        provider_attempts = []

        # GEE is preferred when configured; STAC fallback is specific to this
        # explicitly selected optical spectral engine.
        gee_configuration = get_gee_configuration()
        if gee_configuration.get("configured"):
            try:
                _update_job(job_id, progress="🛰 Attempting fetch via Google Earth Engine…")
                from gee_service import fetch_imagery as gee_fetch
                imagery = gee_fetch(
                    lon_min=float(bbox["lon_min"]), lat_min=float(bbox["lat_min"]),
                    lon_max=float(bbox["lon_max"]), lat_max=float(bbox["lat_max"]),
                    date_t1=date_t1, date_t2=date_t2,
                    window_days=window_days, max_cloud=max_cloud,
                    thumb_size=thumb_size,
                )
                provider = "Google Earth Engine"
                provider_attempts.append({"provider": provider, "status": "success"})
            except Exception as gee_err:
                gee_failure = gee_err
                from gee_config import classify_gee_error
                gee_code, gee_message = classify_gee_error(gee_err)
                provider_attempts.append({"provider": "Google Earth Engine", "status": gee_code, "detail": gee_message})
                app.logger.warning(f"[spectral/{job_id}] GEE unavailable; falling back to STAC")
                _update_job(job_id, progress="🛰 GEE unavailable — using Sentinel-2 Open Data…")
        else:
            provider_attempts.append({"provider": "Google Earth Engine", "status": gee_configuration["code"], "detail": gee_configuration["message"]})

        # Sentinel-2 STAC fallback (free, no auth)
        if imagery is None:
            from sentinel_stac_service import fetch_imagery as stac_fetch
            def _prog(msg):
                _update_job(job_id, progress=msg)
            try:
                imagery = stac_fetch(
                lon_min=float(bbox["lon_min"]), lat_min=float(bbox["lat_min"]),
                lon_max=float(bbox["lon_max"]), lat_max=float(bbox["lat_max"]),
                date_t1=date_t1, date_t2=date_t2,
                window_days=window_days, max_cloud=max_cloud,
                thumb_size=thumb_size,
                progress_cb=_prog,
                )
                provider = "Sentinel-2 STAC"
                provider_attempts.append({"provider": provider, "status": "success"})
                (imagery.get("meta") or {}).update({"t1_date": date_t1, "t2_date": date_t2})
            except Exception as stac_err:
                app.logger.exception("[spectral/%s] STAC acquisition failed", job_id)
                provider_attempts.append({"provider": "Sentinel-2 STAC", "status": _provider_error_kind(stac_err)})
                if gee_failure and _provider_error_kind(gee_failure) == "authentication_failed":
                    raise PipelineError("authentication_failed", "Google Earth Engine authentication failed, and Sentinel-2 STAC could not provide imagery for both dates.") from stac_err
                code = "providers_unavailable" if gee_failure else "imagery_unavailable"
                message = "Both imagery providers failed to return usable scenes." if gee_failure else "Sentinel-2 STAC could not find or retrieve usable imagery for both dates."
                raise PipelineError(code, message) from stac_err

        # ── Step 2: Spectral-index change detection (no neural network) ─────
        _update_job(job_id, progress="⚗️ Computing spectral change indices (NDVI · NDWI · NDBI · BSI)…")
        app.logger.info(f"[spectral/{job_id}] Running spectral engine…")

        result = run_spectral_inference(
            t1_tensor=imagery["t1_tensor"],
            t2_tensor=imagery["t2_tensor"],
            t2_rgb_b64=imagery["t2_rgb_b64"],
            threshold=threshold,
        )

        meta = imagery.get("meta") or {}
        meta["provider"] = provider or meta.get("provider", "unknown")
        meta["provider_fallback"] = bool(gee_failure and provider == "Sentinel-2 STAC")
        meta["provider_attempts"] = provider_attempts
        meta["input_channels"] = 17
        meta["input_channel_order"] = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "NDVI", "NDBI", "VV", "VH", "VH_VV", "Q"]
        meta["ai_input_contract"] = "incomplete" if not meta.get("s1_used", False) else "complete"
        _validate_imagery(imagery)
        _estimated_area_stats(result["stats"], bbox)
        result.update({
            "engine": "spectral_scva",
            "engine_label": "Spectral Index Analysis (S-CVA)",
            "algorithm": "Spectral Change Vector Analysis (NDVI, NDWI, NDBI, BSI)",
            "model_version": None,
            "threshold": result.get("stats", {}).get("threshold"),
            "threshold_source": "regional CVA magnitude statistics and minimum cutoff" if threshold is None else "request override",
        })

        app.logger.info(f"[spectral/{job_id}] Done. Change: {result['stats']['change_pct']}%")

        outcome_result = {
            "success":       True,
            "location_name": loc_name,
            "t1_rgb":        f"data:image/png;base64,{imagery['t1_rgb_b64']}",
            "t2_rgb":        f"data:image/png;base64,{imagery['t2_rgb_b64']}",
            "change_mask":   f"data:image/png;base64,{result['change_mask_b64']}",
            "binary_mask":   f"data:image/png;base64,{result.get('binary_mask_b64', result['change_mask_b64'])}",
            "change_prob":   f"data:image/png;base64,{result['change_prob_b64']}",
            "overlay":       f"data:image/png;base64,{result['overlay_b64']}",
            "dynamics":      result.get("dynamics", {}),
            "stats":         result["stats"],
            "meta":          meta,
            **{key: result[key] for key in ("engine", "engine_label", "algorithm", "model_version", "threshold", "threshold_source")},
        }

        # Auto-save to permanent SQLite storage
        try:
            import storage_manager
            saved = storage_manager.save_prediction({
                "id": f"spec_{job_id[:8]}",
                "locationName": loc_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "bbox": bbox,
                "dateT1": date_t1,
                "dateT2": date_t2,
                "result": outcome_result,
            })
            outcome_result["saved_id"] = saved.get("id")
        except Exception as save_err:
            app.logger.warning(f"[spectral/{job_id}] Storage warning: {save_err}")
            outcome_result["persistence"] = {"saved": False, "error_code": "storage_failed", "message": "Analysis completed, but history could not be saved."}
        else:
            outcome_result["persistence"] = {"saved": True}

        _update_job(job_id, status="done", progress="✅ Complete", result=outcome_result)

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error(f"[spectral/{job_id}] ERROR: {e}\n{tb}")
        code = e.code if isinstance(e, PipelineError) else "spectral_failed"
        message = str(e) if isinstance(e, PipelineError) else "Spectral Index Analysis failed. Check imagery availability and server logs, then retry."
        failure_meta = {key: meta[key] for key in ("provider", "provider_fallback") if "meta" in locals() and key in meta}
        failure_meta["provider_attempts"] = locals().get("provider_attempts", [])
        if isinstance(e, PipelineError):
            failure_meta["diagnostic"] = e.details
        _update_job(job_id, status="error", progress="❌ Failed",
                    error=message, error_code=code, meta=failure_meta)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    model_dir = os.path.join(os.path.dirname(__file__), os.environ.get("MODEL_DIR", "Model/geonexus_v3_2_k30"))
    checkpoint = os.path.join(model_dir, os.environ.get("MODEL_CHECKPOINT", "mh_fewshot_best_k30.pth"))
    gee = get_gee_configuration()
    checkpoint_available = os.path.isfile(checkpoint)
    return jsonify({
        "status": "ok",
        "service": "Geo-Nexus Change Detection API",
        "version": "3.2.0",
        "build_id": BUILD_ID,
        "model": os.environ.get("MODEL_CHECKPOINT", "mh_fewshot_best_k30.pth"),
        "model_checkpoint_available": checkpoint_available,
        "ai_readiness": {
            # File existence alone does not validate checkpoint compatibility;
            # EE access and strict checkpoint loading are verified per run.
            "ready": False,
            "status": gee["status"] if not gee.get("configured") else ("checkpoint_available_unverified" if checkpoint_available else "checkpoint_missing"),
            "gee": {key: value for key, value in gee.items() if key in ("status", "code", "configured", "message")},
            "checkpoint_available": checkpoint_available,
            "note": "Earth Engine authentication, project access, imagery, and strict checkpoint loading are verified during an analysis run.",
        },
        "engines": ["geonexus_ai", "spectral_scva"],
        "ai_requires": "Google Earth Engine imagery with real Sentinel-1 and trained quality inputs",
    })


@app.route("/api/detect/start", methods=["POST"])
def detect_start():
    """
    Start a change detection job asynchronously.
    Returns a job_id immediately — poll /api/detect/status/<job_id> for progress.
    """
    try:
        body = _validate_payload(request.get_json(silent=True), "geonexus_ai")

        dimensions = _bbox_dimensions_m(body["bbox"])
        if dimensions["width"] < AI_MIN_DIMENSION_M or dimensions["height"] < AI_MIN_DIMENSION_M:
            return jsonify({
                "success": False,
                "error": f"The selected area is about {dimensions['width'] / 1000:.2f} km × {dimensions['height'] / 1000:.2f} km. Geo-Nexus AI needs at least 1.28 km in both dimensions at 10 m resolution. Enlarge the area and retry.",
                "error_code": "area_too_small",
                "meta": {"estimated_bbox_dimensions_m": dimensions, "required_minimum_dimensions_m": {"width": AI_MIN_DIMENSION_M, "height": AI_MIN_DIMENSION_M}, "bbox": body["bbox"], "requested_dates": {"t1": body["date_t1"], "t2": body["date_t2"]}},
            }), 422

        readiness = get_gee_configuration()
        if not readiness.get("configured"):
            return jsonify({
                "success": False,
                "error": readiness["message"],
                "error_code": readiness["code"],
                "meta": {"provider": "Google Earth Engine", "provider_attempts": [{"provider": "Google Earth Engine", "status": readiness["code"]}], "requested_dates": {"t1": body["date_t1"], "t2": body["date_t2"]}},
            }), 503

        _trim_jobs()
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                "engine": "geonexus_ai",
                "status": "pending",
                "progress": "Job queued, starting…",
                "result": None,
                "error": None,
                "created_at": time.time(),
            }

        t = threading.Thread(target=_run_job, args=(job_id, body), daemon=True)
        t.start()

        return jsonify({"success": True, "job_id": job_id})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e), "error_code": "invalid_request"}), 400
    except Exception:
        app.logger.exception("Failed to start AI detection job")
        return jsonify({"success": False, "error": "Could not start the AI analysis job.", "error_code": "job_start_failed"}), 500


@app.route("/api/detect/status/<job_id>", methods=["GET"])
def detect_status(job_id: str):
    """Poll this endpoint every 3 seconds to check job progress."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"success": False, "error": "Job not found"}), 404

    resp = {
        "success": True,
        "job_id":  job_id,
        "request_id": job_id,
        "status":  job["status"],    # pending | running | done | error
        "progress": job["progress"],
        "engine": job.get("engine", "unknown"),
    }

    if job["status"] == "done":
        resp["result"] = job["result"]
        _trim_jobs()

    elif job["status"] == "error":
        resp["error"] = job.get("error")
        resp["error_code"] = job.get("error_code", "pipeline_failed")
        resp["meta"] = job.get("meta", {})

    return jsonify(resp)


@app.route("/api/detect/spectral/start", methods=["POST"])
def detect_spectral_start():
    """
    Start a spectral-index change detection job (no ML model).
    Uses the same multi-band Sentinel-2 data but computes change via
    NDVI / NDWI / NDBI / BSI ensemble differencing + Otsu thresholding.
    Returns a job_id — poll /api/detect/status/<job_id> for progress/result.
    """
    try:
        body = _validate_payload(request.get_json(silent=True), "spectral_scva")

        _trim_jobs()
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                "engine": "spectral_scva",
                "status": "pending",
                "progress": "Spectral job queued, starting…",
                "result": None,
                "error": None,
                "created_at": time.time(),
            }

        t = threading.Thread(target=_run_spectral_job, args=(job_id, body), daemon=True)
        t.start()

        return jsonify({"success": True, "job_id": job_id})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e), "error_code": "invalid_request"}), 400
    except Exception:
        app.logger.exception("Failed to start spectral detection job")
        return jsonify({"success": False, "error": "Could not start the spectral analysis job.", "error_code": "job_start_failed"}), 500


# ── Permanent Storage API Endpoints ──────────────────────────────────────────

@app.route("/api/predictions", methods=["GET"])
def get_all_saved_predictions():
    """
    Returns all permanently saved predictions from SQLite database.
    Survives all server restarts, reboots, and browser session clears!
    """
    try:
        import storage_manager
        predictions = storage_manager.get_all_predictions(include_full_result=True)
        return jsonify({
            "success": True,
            "predictions": predictions,
            "count": len(predictions),
        })
    except Exception:
        app.logger.exception("Error fetching saved predictions")
        return jsonify({"success": False, "error": "Could not retrieve saved prediction history.", "error_code": "storage_failed"}), 500


@app.route("/api/predictions/<pred_id>", methods=["GET"])
def get_saved_prediction_by_id(pred_id: str):
    """
    Returns a specific prediction record with full-res imagery by ID.
    """
    try:
        import storage_manager
        prediction = storage_manager.get_prediction_by_id(pred_id)
        if not prediction:
            return jsonify({"success": False, "error": "Prediction not found"}), 404
        return jsonify({"success": True, "prediction": prediction})
    except Exception:
        app.logger.exception("Error fetching saved prediction")
        return jsonify({"success": False, "error": "Could not retrieve the saved prediction.", "error_code": "storage_failed"}), 500


@app.route("/api/predictions", methods=["POST"])
def save_prediction_endpoint():
    """
    Explicitly save or update a prediction record in permanent SQLite storage.
    """
    try:
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"success": False, "error": "Empty body"}), 400
        import storage_manager
        saved = storage_manager.save_prediction(body)
        return jsonify({"success": True, "prediction": saved})
    except Exception:
        app.logger.exception("Error saving prediction")
        return jsonify({"success": False, "error": "Could not save this prediction.", "error_code": "storage_failed"}), 500


@app.route("/api/predictions/<pred_id>", methods=["DELETE"])
def delete_saved_prediction_endpoint(pred_id: str):
    """
    Permanently delete a prediction record from disk.
    """
    try:
        import storage_manager
        deleted = storage_manager.delete_prediction(pred_id)
        return jsonify({"success": True, "deleted": deleted})
    except Exception:
        app.logger.exception("Error deleting saved prediction")
        return jsonify({"success": False, "error": "Could not delete this prediction.", "error_code": "storage_failed"}), 500


@app.route("/api/predictions/match", methods=["POST"])
def match_saved_prediction_endpoint():
    """
    Finds if an exact or near-identical bounding box and date range was already analyzed.
    Allows 0s instant loading without re-running STAC/GEE or GPU inference!
    """
    try:
        body = request.get_json(silent=True) or {}
        bbox = body.get("bbox")
        date_t1 = body.get("date_t1") or body.get("dateT1")
        date_t2 = body.get("date_t2") or body.get("dateT2")
        import storage_manager
        engine = body.get("engine", "geonexus_ai")
        if engine not in ("geonexus_ai", "spectral_scva"):
            return jsonify({"success": False, "error": "engine must be geonexus_ai or spectral_scva", "error_code": "invalid_request"}), 400
        match = storage_manager.find_matching_prediction(bbox, date_t1, date_t2, engine=engine)
        return jsonify({
            "success": True,
            "match": match is not None,
            "prediction": match,
        })
    except Exception:
        app.logger.exception("Error matching saved prediction")
        return jsonify({"success": False, "error": "Could not check saved predictions.", "error_code": "storage_failed"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("FLASK_PORT", 5000))
    host  = os.environ.get("FLASK_HOST", "0.0.0.0")

    # Pre-load PyTorch model before server starts so worker threads don't trigger dynamic file loading
    try:
        from inference import _load_model
        _load_model()
        print("[INIT] GeoNexus model pre-loaded successfully.")
    except Exception as e:
        print(f"[INIT] Model pre-load note: {e}")

    # CRITICAL: use_reloader=False prevents Flask's watchdog from rebooting
    # when PyTorch lazy-loads modules in worker threads
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
