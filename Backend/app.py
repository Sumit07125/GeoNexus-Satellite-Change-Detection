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
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(override=True)

# Confirm .env loaded
_gee_proj = os.environ.get('GEE_PROJECT_ID', '').strip()
if not _gee_proj or _gee_proj in ('your-gcp-project-id', 'NOT SET', 'YOUR_GEE_PROJECT_ID'):
    print("[ENV] ℹ GEE_PROJECT_ID not set in .env — defaulting to Sentinel-2 AWS/Planetary Computer open data.")
else:
    print(f"[ENV] GEE_PROJECT_ID configured: '{_gee_proj}'")

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
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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


def _validate_imagery(imagery, require_sar=False):
    import numpy as np
    t1, t2 = imagery.get("t1_tensor"), imagery.get("t2_tensor")
    if t1 is None or t2 is None or getattr(t1, "ndim", 0) != 3 or getattr(t2, "ndim", 0) != 3:
        raise PipelineError("invalid_input_channels", "The imagery provider did not return usable multi-band arrays.")
    if t1.shape[0] != 17 or t2.shape[0] != 17 or t1.shape != t2.shape:
        raise PipelineError("invalid_input_channels", "Imagery must provide matching 17-channel T1/T2 arrays on the same grid.")
    if min(t1.shape[1:]) < 128:
        raise PipelineError("area_too_small", "The AI model needs at least 128×128 pixels at 10 m. Choose an area at least about 1.3 km in each dimension.")
    if not np.isfinite(t1).all() or not np.isfinite(t2).all():
        raise PipelineError("invalid_input_channels", "The imagery provider returned non-finite channel values.")
    for label, image in (("T1", t1), ("T2", t2)):
        optical_coverage = np.mean(image[:11] > 0, axis=(1, 2))
        if np.any(optical_coverage < 0.05):
            raise PipelineError("invalid_input_channels", f"{label} imagery is missing one or more Sentinel-2 optical bands required for analysis.")
        if require_sar and np.any(np.mean(image[13:16] != 0, axis=(1, 2)) < 0.05):
            raise PipelineError("missing_sar_channels", f"{label} imagery is missing one or more real Sentinel-1 channels required by the model.")
        if require_sar and float(np.ptp(image[16])) <= 0:
            raise PipelineError("invalid_quality_channel", f"{label} imagery has no spatial variation in the trained clear-observation quality channel.")
    valid = (t1[2] > 100) & (t1[6] > 100) & (t2[2] > 100) & (t2[6] > 100)
    if float(valid.mean()) < 0.05:
        raise PipelineError("nodata", "Imagery contains too little valid optical coverage for analysis. Choose a smaller or clearer region/date window.")
    if require_sar and not (imagery.get("meta") or {}).get("s1_used", False):
        raise PipelineError("missing_sar_channels", "Geo-Nexus AI requires real Sentinel-1 VV, VH, VH_VV and spatial quality inputs. The selected provider supplied Sentinel-2 only; run Spectral Index Analysis or configure working Google Earth Engine access.")


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
    """Background thread that runs STAC fetch + inference."""
    try:
        # Use STAC (free, no billing) — GEE as optional fallback
        from sentinel_stac_service import fetch_imagery as stac_fetch
        from inference import run_inference

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

        # ── Step 1: Fetch satellite imagery (GEE first, STAC fallback) ───────
        _update_job(job_id, status="running",
                    progress="🛰 Fetching satellite imagery…")
        app.logger.info(f"[{job_id}] Satellite fetch for '{loc_name}'")

        imagery = None
        provider = None
        gee_failure = None
        provider_attempts = []

        # Strategy 1: Try Google Earth Engine first
        gee_proj = os.environ.get("GEE_PROJECT_ID", "").strip()
        if gee_proj and gee_proj not in ('', 'NOT SET', 'YOUR_GEE_PROJECT_ID'):
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
                app.logger.info(f"[{job_id}] Google Earth Engine fetch succeeded!")
            except Exception as gee_err:
                if "single-request 10 m model grid limit" in str(gee_err):
                    raise PipelineError("area_too_large", "This area is too large for one 10 m AI request. Draw a smaller bbox and retry.") from gee_err
                gee_failure = gee_err
                provider_attempts.append({"provider": "Google Earth Engine", "status": _provider_error_kind(gee_err)})
                app.logger.warning(f"[{job_id}] GEE unavailable; falling back to Sentinel-2 Open Data.")
                _update_job(job_id, progress="🛰 GEE permission error — falling back to Sentinel-2 Open Data…")

        # Strategy 2: High-speed Sentinel-2 STAC (free, no auth, no billing required)
        if imagery is None:
            from sentinel_stac_service import fetch_imagery as stac_fetch
            def update_progress(msg):
                _update_job(job_id, progress=msg)
            try:
                imagery = stac_fetch(
                lon_min=float(bbox["lon_min"]), lat_min=float(bbox["lat_min"]),
                lon_max=float(bbox["lon_max"]), lat_max=float(bbox["lat_max"]),
                date_t1=date_t1, date_t2=date_t2,
                window_days=window_days, max_cloud=max_cloud,
                thumb_size=thumb_size,
                progress_cb=update_progress,
                )
                provider = "Sentinel-2 STAC"
                provider_attempts.append({"provider": provider, "status": "success"})
            except Exception as stac_err:
                app.logger.exception("[%s] Sentinel-2 STAC acquisition failed", job_id)
                provider_attempts.append({"provider": "Sentinel-2 STAC", "status": _provider_error_kind(stac_err)})
                if gee_failure:
                    if _provider_error_kind(gee_failure) == "authentication_failed":
                        raise PipelineError("authentication_failed", "Google Earth Engine authentication failed, and Sentinel-2 STAC could not provide imagery for both dates.") from stac_err
                    raise PipelineError("providers_unavailable", "Imagery acquisition failed for both Google Earth Engine and Sentinel-2 STAC. Check provider access and scene availability.") from stac_err
                raise PipelineError("imagery_unavailable", "Sentinel-2 STAC could not find or retrieve usable imagery for both dates.") from stac_err

        meta = imagery.get("meta") or {}
        meta["provider"] = provider or meta.get("provider", "unknown")
        meta["provider_fallback"] = bool(gee_failure and provider == "Sentinel-2 STAC")
        meta["provider_attempts"] = provider_attempts
        meta["input_channels"] = 17
        meta["input_channel_order"] = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "NDVI", "NDBI", "VV", "VH", "VH_VV", "Q"]
        meta["ai_input_contract"] = "complete" if meta.get("s1_used", False) else "incomplete"
        _validate_imagery(imagery, require_sar=True)

        _update_job(job_id, progress="🤖 Running GeoNexus v3.2 change detection model…")
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
        tb = traceback.format_exc()
        app.logger.error(f"[{job_id}] ERROR: {e}\n{tb}")
        code = e.code if isinstance(e, PipelineError) else "pipeline_failed"
        message = str(e) if isinstance(e, PipelineError) else "Geo-Nexus AI analysis failed. Check imagery availability and server logs, then retry."
        failure_meta = {key: meta[key] for key in ("provider", "provider_fallback", "input_channels", "s1_used") if "meta" in locals() and key in meta}
        failure_meta["provider_attempts"] = locals().get("provider_attempts", [])
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

        # Try GEE first if configured
        gee_proj = os.environ.get("GEE_PROJECT_ID", "").strip()
        if gee_proj and gee_proj not in ('', 'NOT SET', 'YOUR_GEE_PROJECT_ID'):
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
                provider_attempts.append({"provider": "Google Earth Engine", "status": _provider_error_kind(gee_err)})
                app.logger.warning(f"[spectral/{job_id}] GEE unavailable; falling back to STAC")
                _update_job(job_id, progress="🛰 GEE unavailable — using Sentinel-2 Open Data…")

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
        _update_job(job_id, status="error", progress="❌ Failed",
                    error=message, error_code=code, meta=failure_meta)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    model_dir = os.path.join(os.path.dirname(__file__), os.environ.get("MODEL_DIR", "Model/geonexus_v3_2_k30"))
    checkpoint = os.path.join(model_dir, os.environ.get("MODEL_CHECKPOINT", "mh_fewshot_best_k30.pth"))
    return jsonify({
        "status": "ok",
        "service": "Geo-Nexus Change Detection API",
        "version": "3.2.0",
        "model": os.environ.get("MODEL_CHECKPOINT", "mh_fewshot_best_k30.pth"),
        "model_checkpoint_available": os.path.isfile(checkpoint),
        "engines": ["geonexus_ai", "spectral_scva"],
        "ai_requires": "Google Earth Engine imagery with real Sentinel-1 and spatial quality channels",
    })


@app.route("/api/detect/start", methods=["POST"])
def detect_start():
    """
    Start a change detection job asynchronously.
    Returns a job_id immediately — poll /api/detect/status/<job_id> for progress.
    """
    try:
        body = _validate_payload(request.get_json(silent=True), "geonexus_ai")

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
