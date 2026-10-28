import os
import io
import math
import datetime
import requests
import numpy as np
from PIL import Image

import ee

# Ensure .env is loaded even when this module is imported standalone
try:
    from dotenv import load_dotenv as _load_dotenv
    import pathlib
    _env_path = pathlib.Path(__file__).parent / ".env"
    _load_dotenv(dotenv_path=_env_path, override=True)
except Exception:
    pass  # dotenv not installed or .env missing — rely on os.environ

_EE_INITIALIZED = False


def init_ee():
    """
    Initialize Earth Engine once per process.
    Tries multiple strategies in order:
      1. Service account key (if GEE_SERVICE_ACCOUNT_KEY is set)
      2. User credentials + specified GEE_PROJECT_ID
      3. User credentials without project (legacy EE accounts)
    """
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    project = os.environ.get("GEE_PROJECT_ID", "").strip()
    key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "").strip()

    print(f"[GEE] init_ee() called — project='{project}' key='{key_path or 'none'}'")

    last_error = None

    # Strategy 1: Service account JSON key
    if key_path and os.path.isfile(key_path):
        try:
            credentials = ee.ServiceAccountCredentials(email=None, key_file=key_path)
            ee.Initialize(credentials=credentials, project=project or None)
            _EE_INITIALIZED = True
            print(f"[GEE] Initialized via service account. Project: {project or 'default'}")
            return
        except Exception as e:
            last_error = e
            print(f"[GEE] Service account init failed: {e}")

    # Strategy 2: User credentials + project ID
    if project:
        try:
            ee.Initialize(project=project)
            _EE_INITIALIZED = True
            print(f"[GEE] Initialized with project: {project}")
            return
        except Exception as e:
            last_error = e
            print(f"[GEE] Project-based init failed: {e}")

    # Strategy 3: Legacy user credentials (reads project from credentials file)
    try:
        print("[GEE] Trying legacy initialization (no project arg)...")
        ee.Initialize()
        _EE_INITIALIZED = True
        print("[GEE] Initialized in legacy mode.")
        return
    except Exception as e:
        last_error = e
        print(f"[GEE] Legacy init also failed: {e}")

    # Strategy 4: High-volume endpoint (sometimes bypasses permission issues)
    try:
        print("[GEE] Trying high-volume endpoint...")
        ee.Initialize(
            project=project,
            opt_url='https://earthengine-highvolume.googleapis.com'
        )
        _EE_INITIALIZED = True
        print("[GEE] Initialized via high-volume endpoint.")
        return
    except Exception as e:
        last_error = e
        print(f"[GEE] High-volume init also failed: {e}")

    raise RuntimeError(
        f"All Google Earth Engine initialization strategies failed.\n"
        f"Last error: {last_error}\n\n"
        f"To fix:\n"
        f"  1. Run: earthengine authenticate\n"
        f"  2. Enable EE API: https://console.cloud.google.com/apis/library/earthengine.googleapis.com?project={project}\n"
        f"  3. Grant IAM role: https://console.cloud.google.com/iam-admin/iam?project={project}\n"
        f"     Add your Google account → Role: 'Earth Engine Resource User'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Band helpers
# ─────────────────────────────────────────────────────────────────────────────

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
# Channel order: 11 optical raw + NDVI + NDBI + VV + VH + VH_VV + Q
CHANNEL_ORDER = [*S2_BANDS, "NDVI", "NDBI", "VV", "VH", "VH_VV", "Q"]


def _get_s2_composite(bbox: ee.Geometry, date_str: str, window_days: int, max_cloud: int):
    """
    Return the least-cloudy Sentinel-2 SR HARMONIZED composite within
    ±window_days of date_str. Also return the n_clear observation count.
    """
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start = (target - datetime.timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (target + datetime.timedelta(days=window_days + 1)).strftime("%Y-%m-%d")

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    # Cloud Score+ masking
    # Match training preprocessing: link Cloud Score+ by source scene index.
    # Unmatched Cloud Score+ bands remain masked, so they cannot count as clear.
    cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    linked = col.linkCollection(cloud_score, ["cs_cdf"])
    masked = linked.map(lambda img: img.updateMask(img.select("cs_cdf").gte(0.60)))

    # n_clear: how many valid observations per pixel
    n_clear = masked.select("B4").count().rename("n_clear").unmask(0)

    composite = masked.select(S2_BANDS).median()

    # Add NDVI and NDBI
    eps = 1e-6
    b8, b4, b11 = composite.select("B8"), composite.select("B4"), composite.select("B11")
    ndvi = b8.subtract(b4).divide(b8.add(b4).add(eps)).rename("NDVI")
    ndbi = b11.subtract(b8).divide(b11.add(b8).add(eps)).rename("NDBI")

    return composite.addBands([ndvi, ndbi, n_clear]), int(col.size().getInfo())


def _get_s1_composite(bbox: ee.Geometry, date_str: str, window_days: int, relative_orbit: int):
    """
    Return a Sentinel-1 GRD median composite (VV, VH, VH_VV) for the period.
    Applies linear-power averaging: composite in linear → convert back to dB.
    """
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start = (target - datetime.timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (target + datetime.timedelta(days=window_days + 1)).strftime("%Y-%m-%d")

    def to_natural(img):
        return ee.Image(10).pow(img.divide(10))

    def to_db(img):
        return img.log10().multiply(10)

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox)
        .filterDate(start, end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .select(["VV", "VH"])
    )

    scene_count = int(col.size().getInfo())
    if scene_count == 0:
        raise RuntimeError(f"No Sentinel-1 scenes matched orbit {relative_orbit} for {date_str} ±{window_days} days.")

    composite = to_db(col.map(to_natural).median())
    vv = composite.select("VV")
    vh = composite.select("VH")
    vhvv = vh.subtract(vv).rename("VH_VV")

    return composite.addBands(vhvv), scene_count


# ─────────────────────────────────────────────────────────────────────────────
# Public fetch function
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_optimal_dimensions(lon_min, lat_min, lon_max, lat_max, base_dim=512, min_dim=256, max_dim=768):
    lon_span = abs(float(lon_max) - float(lon_min))
    lat_span = abs(float(lat_max) - float(lat_min))
    lat_mid = (float(lat_min) + float(lat_max)) / 2.0
    lat_cos = max(0.1, math.cos(math.radians(lat_mid)))
    aspect = (lon_span * lat_cos) / (lat_span + 1e-9)

    base = max(base_dim, 512)
    if aspect >= 1.0:
        w = base
        h = int(round(base / aspect))
    else:
        h = base
        w = int(round(base * aspect))

    w = max(min_dim, min(max_dim, (w // 32) * 32))
    h = max(min_dim, min(max_dim, (h // 32) * 32))
    return int(h), int(w)


def fetch_imagery(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float,
    date_t1: str, date_t2: str,
    window_days: int = 15,
    max_cloud: int = 30,
    thumb_size: int = 512,
) -> dict:
    init_ee()

    bbox = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    center_lon = (float(lon_min) + float(lon_max)) / 2
    output_crs = "EPSG:32643" if center_lon < 78 else "EPSG:32644"
    relative_orbit = 136 if center_lon < 77 else 165
    out_h, out_w = _calculate_optimal_dimensions(lon_min, lat_min, lon_max, lat_max, base_dim=thumb_size)
    thumb_dim = f"{out_w}x{out_h}"

    # ── T1 composite ────────────────────────────────────────────────────────
    s2_t1, t1_count = _get_s2_composite(bbox, date_t1, window_days, max_cloud)
    if t1_count == 0:
        raise RuntimeError(f"No Sentinel-2 scenes matched the T1 window for {date_t1}.")
    s1_t1, t1_s1_count = _get_s1_composite(bbox, date_t1, window_days, relative_orbit)

    # ── T2 composite ────────────────────────────────────────────────────────
    s2_t2, t2_count = _get_s2_composite(bbox, date_t2, window_days, max_cloud)
    if t2_count == 0:
        raise RuntimeError(f"No Sentinel-2 scenes matched the T2 window for {date_t2}.")
    s1_t2, t2_s1_count = _get_s1_composite(bbox, date_t2, window_days, relative_orbit)

    # ── RGB thumbnails (B4=Red, B3=Green, B2=Blue) ──────────────────────────
    vis_rgb = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.4}

    t1_thumb_url = s2_t1.getThumbURL({
        **vis_rgb, "region": bbox, "dimensions": thumb_dim, "format": "png"
    })
    t2_thumb_url = s2_t2.getThumbURL({
        **vis_rgb, "region": bbox, "dimensions": thumb_dim, "format": "png"
    })

    t1_rgb_b64 = _url_to_b64(t1_thumb_url)
    t2_rgb_b64 = _url_to_b64(t2_thumb_url)

    # ── 17-channel numpy tensors ─────────────────────────────────────────────
    t1_tensor, t1_quality_stats = _build_tensor(s2_t1, s1_t1, bbox, output_crs)
    t2_tensor, t2_quality_stats = _build_tensor(s2_t2, s1_t2, bbox, output_crs)
    if t1_tensor.shape != t2_tensor.shape:
        raise RuntimeError("T1 and T2 acquisitions produced different output grids.")

    target_t1 = datetime.datetime.strptime(date_t1, "%Y-%m-%d")
    target_t2 = datetime.datetime.strptime(date_t2, "%Y-%m-%d")
    meta = {
        "provider": "Google Earth Engine",
        "preprocessing_id": "gee_s2_s1_median_composite_v2",
        "s1_used": True,
        "sar_channels_available": True,
        "s1_note": "Sentinel-1 GRD VV/VH composite and VH_VV difference",
        "crs": output_crs,
        "relative_orbit": relative_orbit,
        "orbit_pass": "DESCENDING",
        "output_shape": list(t1_tensor.shape[1:]),
        "pixel_resolution_m": 10,
        "resolution_note": "GEE model inputs use a 10 m UTM grid; area values are approximate at bbox edges.",
        "quality_channel": "clip(n_clear / 8, 0, 1)",
        "quality_channel_status": "real",
        "quality_channel_source": "per-pixel count of Sentinel-2 B4 observations after linked Cloud Score+ cs_cdf >= 0.60 masking, divided by 8 and clipped to [0,1]",
        "t1_quality_raw_stats": t1_quality_stats,
        "t2_quality_raw_stats": t2_quality_stats,
        "cloud_score_join": "Cloud Score+ linked by source system:index; unmatched quality bands remain masked",
        "s2_collection": "COPERNICUS/S2_SR_HARMONIZED",
        "s1_collection": "COPERNICUS/S1_GRD",
        "t1_date": date_t1,
        "t2_date": date_t2,
        "t1_window": f"{(target_t1 - datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')} → "
                     f"{(target_t1 + datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')}",
        "t2_window": f"{(target_t2 - datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')} → "
                     f"{(target_t2 + datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')}",
        "t1_scene_count": t1_count,
        "t2_scene_count": t2_count,
        "t1_s1_scene_count": t1_s1_count,
        "t2_s1_scene_count": t2_s1_count,
        "window_days": window_days,
        "max_cloud_pct": max_cloud,
        "clear_score_threshold": 0.60,
        "input_channel_order": CHANNEL_ORDER,
        "bbox": [lon_min, lat_min, lon_max, lat_max],
    }

    return {
        "t1_rgb_b64": t1_rgb_b64,
        "t2_rgb_b64": t2_rgb_b64,
        "t1_tensor": t1_tensor,   # float32 [17, H, W], training scales retained
        "t2_tensor": t2_tensor,
        "meta": meta,
    }


def _build_tensor(s2_img, s1_img, bbox, output_crs: str) -> tuple[np.ndarray, dict]:
    """
    Download the trained [17, H, W] contract and raw n_clear stats on a 10 m UTM grid.
    Channel order: B2..B12(11) + NDVI + NDBI + VV + VH + VH_VV + Q(n_clear)
    Returns (encoded_tensor, raw_n_clear_statistics).
    """
    bands = S2_BANDS + ["NDVI", "NDBI", "VV", "VH", "VH_VV", "n_clear"]

    area_m2 = float(bbox.area(maxError=1).getInfo())
    if area_m2 / 100.0 > 400_000:
        raise RuntimeError("Selected area exceeds the single-request 10 m model grid limit; select a smaller area.")

    # Combine S2 and S1 into one image
    combined = s2_img.addBands(s1_img)

    # Use getDownloadURL for numpy-compatible download
    url = combined.select(bands).getDownloadURL({
        "scale": 10,
        "crs": output_crs,
        "region": bbox,
        "format": "NPY",
    })

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    arr = np.load(io.BytesIO(resp.content))  # structured numpy array

    channels = []
    for b in bands:
        ch = arr[b].astype(np.float32)
        channels.append(ch)

    stacked = np.stack(channels, axis=0)  # [17, H, W]

    quality = stacked[16]
    finite_quality = quality[np.isfinite(quality)]
    quality_stats = {
        "source_feature": "n_clear before Q scaling/clipping",
        "valid_pixel_fraction": float(np.isfinite(quality).mean()),
        "positive_fraction": float((quality > 0).mean()),
        "min": float(finite_quality.min()) if finite_quality.size else None,
        "max": float(finite_quality.max()) if finite_quality.size else None,
        "mean": float(finite_quality.mean()) if finite_quality.size else None,
        "std": float(finite_quality.std()) if finite_quality.size else None,
    }
    return _encode_model_tensor(stacked), quality_stats


def _encode_model_tensor(raw: np.ndarray) -> np.ndarray:
    """Convert raw GEE bands to training storage scales without quantizing floats."""
    if raw.ndim != 3 or raw.shape[0] != 17:
        raise ValueError("Expected 17 channels in manifest order.")
    encoded = raw.astype(np.float32, copy=True)
    # Earth Engine Sentinel-2 SR bands are already DN scaled by 10,000.
    encoded[:11] = np.clip(encoded[:11], -32768, 32767)
    encoded[13:16] = np.clip(encoded[13:16] * 100, -32768, 32767)
    encoded[16] = np.clip(encoded[16] / 8.0, 0.0, 1.0)
    return encoded


def _url_to_b64(url: str) -> str:
    """Download image from URL and return as base64-encoded PNG string."""
    import base64
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
