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
    "origins": "*",
    "methods": ["GET", "POST", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type"],
}})

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { "status": "pending|running|done|error", "progress": str, "result": dict, "error": str } }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


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
                app.logger.info(f"[{job_id}] Google Earth Engine fetch succeeded!")
            except Exception as gee_err:
                app.logger.warning(f"[{job_id}] GEE unavailable ({gee_err}). Falling back to Sentinel-2 Open Data…")
                _update_job(job_id, progress="🛰 GEE permission error — falling back to Sentinel-2 Open Data…")

        # Strategy 2: High-speed Sentinel-2 STAC (free, no auth, no billing required)
        if imagery is None:
            from sentinel_stac_service import fetch_imagery as stac_fetch
            def update_progress(msg):
                _update_job(job_id, progress=msg)
            imagery = stac_fetch(
                lon_min=float(bbox["lon_min"]), lat_min=float(bbox["lat_min"]),
                lon_max=float(bbox["lon_max"]), lat_max=float(bbox["lat_max"]),
                date_t1=date_t1, date_t2=date_t2,
                window_days=window_days, max_cloud=max_cloud,
                thumb_size=thumb_size,
                progress_cb=update_progress,
            )

        _update_job(job_id, progress="🤖 Running GeoNexus v3.2 change detection model…")
        app.logger.info(f"[{job_id}] Imagery fetched. Running inference…")

        # ── Step 2: Model inference ───────────────────────────────────────────
        result = run_inference(
            t1_tensor=imagery["t1_tensor"],
            t2_tensor=imagery["t2_tensor"],
            t2_rgb_b64=imagery["t2_rgb_b64"],
            threshold=threshold,
        )

        app.logger.info(f"[{job_id}] Done. Change: {result['stats']['change_pct']}%")

        outcome_result = {
            "success": True,
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

        _update_job(job_id, status="done", progress="✅ Complete", result=outcome_result)

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error(f"[{job_id}] ERROR: {e}\n{tb}")
        _update_job(job_id, status="error", progress="❌ Failed",
                    error=str(e),
                    detail=tb if app.debug else "Enable debug mode for details.")


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
            except Exception as gee_err:
                app.logger.warning(f"[spectral/{job_id}] GEE unavailable ({gee_err})")
                _update_job(job_id, progress="🛰 GEE unavailable — using Sentinel-2 Open Data…")

        # Sentinel-2 STAC fallback (free, no auth)
        if imagery is None:
            from sentinel_stac_service import fetch_imagery as stac_fetch
            def _prog(msg):
                _update_job(job_id, progress=msg)
            imagery = stac_fetch(
                lon_min=float(bbox["lon_min"]), lat_min=float(bbox["lat_min"]),
                lon_max=float(bbox["lon_max"]), lat_max=float(bbox["lat_max"]),
                date_t1=date_t1, date_t2=date_t2,
                window_days=window_days, max_cloud=max_cloud,
                thumb_size=thumb_size,
                progress_cb=_prog,
            )

        # ── Step 2: Spectral-index change detection (no neural network) ─────
        _update_job(job_id, progress="⚗️ Computing spectral change indices (NDVI · NDWI · NDBI · BSI)…")
        app.logger.info(f"[spectral/{job_id}] Running spectral engine…")

        result = run_spectral_inference(
            t1_tensor=imagery["t1_tensor"],
            t2_tensor=imagery["t2_tensor"],
            t2_rgb_b64=imagery["t2_rgb_b64"],
            threshold=threshold,
        )

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
            "meta":          imagery["meta"],
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

        _update_job(job_id, status="done", progress="✅ Complete", result=outcome_result)

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error(f"[spectral/{job_id}] ERROR: {e}\n{tb}")
        _update_job(job_id, status="error", progress="❌ Failed",
                    error=str(e),
                    detail=tb if app.debug else "Enable debug mode for details.")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Geo-Nexus Change Detection API",
        "version": "3.2.0",
        "model": os.environ.get("MODEL_CHECKPOINT", "mh_fewshot_best_k30.pth"),
    })


@app.route("/api/detect/start", methods=["POST"])
def detect_start():
    """
    Start a change detection job asynchronously.
    Returns a job_id immediately — poll /api/detect/status/<job_id> for progress.
    """
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"success": False, "error": "Empty request body"}), 400

        for field in ["bbox", "date_t1", "date_t2"]:
            if field not in body:
                return jsonify({"success": False, "error": f"Missing field: {field}"}), 400

        bbox = body["bbox"]
        for coord in ["lon_min", "lat_min", "lon_max", "lat_max"]:
            if coord not in bbox:
                return jsonify({"success": False, "error": f"Missing bbox.{coord}"}), 400

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "pending",
                "progress": "Job queued, starting…",
                "result": None,
                "error": None,
            }

        t = threading.Thread(target=_run_job, args=(job_id, body), daemon=True)
        t.start()

        return jsonify({"success": True, "job_id": job_id})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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
    }

    if job["status"] == "done":
        resp["result"] = job["result"]
        # Keep completed job in cache for multi-tab/re-poll safety, trim old jobs if too large
        with _jobs_lock:
            if len(_jobs) > 30:
                # remove oldest keys
                for k in list(_jobs.keys())[:-20]:
                    _jobs.pop(k, None)

    elif job["status"] == "error":
        resp["error"]  = job.get("error")
        resp["detail"] = job.get("detail")
        with _jobs_lock:
            _jobs.pop(job_id, None)

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
        body = request.get_json(force=True)
        if not body:
            return jsonify({"success": False, "error": "Empty request body"}), 400

        for field in ["bbox", "date_t1", "date_t2"]:
            if field not in body:
                return jsonify({"success": False, "error": f"Missing field: {field}"}), 400

        bbox = body["bbox"]
        for coord in ["lon_min", "lat_min", "lon_max", "lat_max"]:
            if coord not in bbox:
                return jsonify({"success": False, "error": f"Missing bbox.{coord}"}), 400

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "pending",
                "progress": "Spectral job queued, starting…",
                "result": None,
                "error": None,
            }

        t = threading.Thread(target=_run_spectral_job, args=(job_id, body), daemon=True)
        t.start()

        return jsonify({"success": True, "job_id": job_id})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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
    except Exception as e:
        app.logger.error(f"Error fetching saved predictions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


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
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predictions", methods=["POST"])
def save_prediction_endpoint():
    """
    Explicitly save or update a prediction record in permanent SQLite storage.
    """
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"success": False, "error": "Empty body"}), 400
        import storage_manager
        saved = storage_manager.save_prediction(body)
        return jsonify({"success": True, "prediction": saved})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predictions/<pred_id>", methods=["DELETE"])
def delete_saved_prediction_endpoint(pred_id: str):
    """
    Permanently delete a prediction record from disk.
    """
    try:
        import storage_manager
        deleted = storage_manager.delete_prediction(pred_id)
        return jsonify({"success": True, "deleted": deleted})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predictions/match", methods=["POST"])
def match_saved_prediction_endpoint():
    """
    Finds if an exact or near-identical bounding box and date range was already analyzed.
    Allows 0s instant loading without re-running STAC/GEE or GPU inference!
    """
    try:
        body = request.get_json(force=True)
        bbox = body.get("bbox")
        date_t1 = body.get("date_t1") or body.get("dateT1")
        date_t2 = body.get("date_t2") or body.get("dateT2")
        import storage_manager
        match = storage_manager.find_matching_prediction(bbox, date_t1, date_t2)
        return jsonify({
            "success": True,
            "match": match is not None,
            "prediction": match,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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
