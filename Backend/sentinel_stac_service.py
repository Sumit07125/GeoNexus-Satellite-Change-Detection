"""
sentinel_stac_service.py
=======================
Free, no-auth, no-billing satellite imagery fetcher.
Uses Microsoft Planetary Computer / Element84 STAC endpoints
to fetch Sentinel-2 L2A imagery directly from AWS open data.

No GEE account. No billing. No API key.
"""

import os
import io
import math
import datetime
import json
import requests
import numpy as np
from PIL import Image

# STAC endpoint for Sentinel-2 (Element84 / AWS Open Data)
STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

# S2 band keys in STAC assets → our channel order (matching GEE B2-B12 order)
S2_ASSET_KEYS = [
    "blue",        # B02  ch0
    "green",       # B03  ch1
    "red",         # B04  ch2
    "rededge1",    # B05  ch3
    "rededge2",    # B06  ch4
    "rededge3",    # B07  ch5
    "nir",         # B08  ch6
    "nir08",       # B8A  ch7
    "nir09",       # B09  ch8
    "swir16",      # B11  ch9
    "swir22",      # B12  ch10
]

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

class STACItem:
    """Lightweight representation of a STAC Item GeoJSON feature."""
    def __init__(self, feat: dict):
        self.id = feat.get("id", "")
        self.properties = feat.get("properties", {})
        self.assets = feat.get("assets", {})
        self.datetime = self.properties.get("datetime", "")


def _probe_scene_coverage(feat, bbox):
    """
    Check if a candidate scene covers the bbox without swath/nodata edge cutoffs.
    Returns (zero_frac, is_enclosing).
    """
    search_bbox = [bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]]
    fb = feat.get("bbox", [])
    is_enclosing = False
    if len(fb) >= 4:
        is_enclosing = (
            fb[0] <= search_bbox[0] + 0.005 and
            fb[1] <= search_bbox[1] + 0.005 and
            fb[2] >= search_bbox[2] - 0.005 and
            fb[3] >= search_bbox[3] - 0.005
        )

    zero_frac = 0.0
    red_asset = feat.get("assets", {}).get("red", {})
    href = red_asset.get("href", "")
    if href:
        try:
            import rasterio
            import rasterio.warp
            from rasterio.crs import CRS
            from rasterio.windows import from_bounds
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_TIMEOUT=15,
            ):
                with rasterio.open(href) as src:
                    dst_crs = CRS.from_epsg(4326)
                    if src.crs != dst_crs:
                        bounds_src = rasterio.warp.transform_bounds(
                            dst_crs, src.crs,
                            bbox["lon_min"], bbox["lat_min"],
                            bbox["lon_max"], bbox["lat_max"]
                        )
                    else:
                        bounds_src = (
                            bbox["lon_min"], bbox["lat_min"],
                            bbox["lon_max"], bbox["lat_max"]
                        )
                    win = from_bounds(*bounds_src, src.transform)
                    probe = src.read(1, window=win, out_shape=(32, 32), resampling=rasterio.enums.Resampling.nearest)
                    zero_frac = float(np.mean(probe == 0))
        except Exception:
            zero_frac = 0.5

    return zero_frac, is_enclosing


def _search_best_scene(bbox, date_str, window_days=15, max_cloud=30):
    """
    Search STAC and select the cleanest Sentinel-2 scene:
    1. Zero nodata / swath cuts (probes pixel coverage to guarantee complete images).
    2. Enclosing bounding box coverage.
    3. Minimum cloud cover.
    Returns (primary_item, fallback_items) for seamless mosaicing.
    """
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start = (d - datetime.timedelta(days=window_days)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (d + datetime.timedelta(days=window_days)).strftime("%Y-%m-%dT23:59:59Z")

    search_url = f"{STAC_URL}/search"
    search_bbox = [bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]]

    # Query candidate scenes
    payload = {
        "collections": [S2_COLLECTION],
        "bbox": search_bbox,
        "datetime": f"{start}/{end}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 10,
    }
    feats = []
    try:
        r = requests.post(search_url, json=payload, timeout=20)
        feats = r.json().get("features", []) if r.status_code == 200 else []
    except Exception as e:
        print(f"  [STAC] Search request error: {e}")

    if not feats:
        payload2 = {
            "collections": [S2_COLLECTION],
            "bbox": search_bbox,
            "datetime": f"{start}/{end}",
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            "limit": 6,
        }
        try:
            r2 = requests.post(search_url, json=payload2, timeout=20)
            feats = r2.json().get("features", []) if r2.status_code == 200 else []
        except Exception as e:
            print(f"  [STAC] Search fallback error: {e}")

    if not feats:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for bbox={bbox} "
            f"around {date_str} ±{window_days} days."
        )

    # Probe candidates for swath edge cutoffs & enclosing coverage
    evaluated = []
    for f in feats:
        zero_frac, is_enclosing = _probe_scene_coverage(f, bbox)
        cloud = float(f.get("properties", {}).get("eo:cloud_cover", 50.0))
        # Perfect scene has 0% zeros
        is_clean = (zero_frac < 0.01)
        evaluated.append({
            "item": STACItem(f),
            "zero_frac": zero_frac,
            "is_enclosing": is_enclosing,
            "is_clean": is_clean,
            "cloud": cloud,
        })
        # If we found an enclosing scene with 0% nodata and low clouds, stop early
        if is_clean and is_enclosing and cloud < 10.0:
            break

    # Sort evaluated: clean scenes first, then enclosing scenes, then lowest zero_frac, then lowest cloud
    evaluated.sort(key=lambda x: (
        not x["is_clean"],
        not x["is_enclosing"],
        x["zero_frac"],
        x["cloud"]
    ))

    best = evaluated[0]["item"]
    fallbacks = [x["item"] for x in evaluated[1:]]
    print(f"  [STAC] Selected best scene {best.id} (clean={evaluated[0]['is_clean']}, zeros={evaluated[0]['zero_frac']:.1%}, cloud={evaluated[0]['cloud']}%)")
    return best, fallbacks


def _calculate_optimal_dimensions(bbox, base_dim=512, min_dim=384, max_dim=768):
    """
    Compute (height, width) matching the true ground aspect ratio of the bounding box.
    Eliminates squashed / distorted / blurry rendering when drawing random rectangles.
    Ensures dimensions are multiples of 32 for optimal tiled inference.
    """
    lon_span = abs(float(bbox["lon_max"]) - float(bbox["lon_min"]))
    lat_span = abs(float(bbox["lat_max"]) - float(bbox["lat_min"]))
    lat_mid = (float(bbox["lat_min"]) + float(bbox["lat_max"])) / 2.0
    lat_cos = max(0.1, math.cos(math.radians(lat_mid)))

    # Real-world aspect ratio: width / height
    aspect = (lon_span * lat_cos) / (lat_span + 1e-9)

    base = max(base_dim, 512)
    if aspect >= 1.0:
        w = base
        h = int(round(base / aspect))
    else:
        h = base
        w = int(round(base * aspect))

    # Clamp bounds and round to multiples of 32
    w = max(min_dim, min(max_dim, (w // 32) * 32))
    h = max(min_dim, min(max_dim, (h // 32) * 32))
    return int(h), int(w)


def _read_band_rasterio(href, bbox, size):
    """
    Read a single band COG tile using rasterio (fast, range-request).
    Preserves exact geographic bounds with cubic resampling for sharp details.
    """
    import rasterio
    import rasterio.warp
    from rasterio.crs import CRS
    from rasterio.windows import from_bounds

    if isinstance(size, (tuple, list)):
        out_h, out_w = int(size[0]), int(size[1])
    else:
        out_h, out_w = int(size), int(size)

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_TIMEOUT=30,
        GDAL_HTTP_MAX_RETRY=3,
        GDAL_HTTP_RETRY_DELAY=1,
    ):
        with rasterio.open(href) as src:
            dst_crs = CRS.from_epsg(4326)
            if src.crs != dst_crs:
                bounds_src = rasterio.warp.transform_bounds(
                    dst_crs, src.crs,
                    bbox["lon_min"], bbox["lat_min"],
                    bbox["lon_max"], bbox["lat_max"]
                )
            else:
                bounds_src = (
                    bbox["lon_min"], bbox["lat_min"],
                    bbox["lon_max"], bbox["lat_max"]
                )

            win = from_bounds(*bounds_src, src.transform)
            # Use cubic resampling to maintain razor-sharp terrain & field boundaries
            data = src.read(
                1,
                window=win,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.cubic,
            ).astype(np.float32)
    return data


def _read_band_requests(href, bbox, size):
    """
    Fallback: download the whole GeoTIFF and let PIL crop/resize with Lanczos.
    """
    if isinstance(size, (tuple, list)):
        out_h, out_w = int(size[0]), int(size[1])
    else:
        out_h, out_w = int(size), int(size)

    resp = requests.get(href, timeout=30, stream=True)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    img = img.resize((out_w, out_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def _get_band(item, asset_key, bbox, size, fallback_items=None):
    """
    Fetch a single band using best available method with graceful fallback.
    If the primary scene contains nodata/zeros, mosaics with fallback items.
    """
    if isinstance(size, (tuple, list)):
        out_h, out_w = int(size[0]), int(size[1])
    else:
        out_h, out_w = int(size), int(size)

    def _read_item_band(target_item):
        asset = target_item.assets.get(asset_key)
        if asset is None:
            return None
        href = asset.get("href") if isinstance(asset, dict) else getattr(asset, "href", "")
        if not href:
            return None
        if "blob.core.windows.net" in href or "planetarycomputer" in href:
            try:
                import planetary_computer
                href = planetary_computer.sign(href)
            except ImportError:
                pass
        try:
            return _read_band_rasterio(href, bbox, (out_h, out_w))
        except Exception as e:
            try:
                return _read_band_requests(href, bbox, (out_h, out_w))
            except Exception:
                return None

    data = _read_item_band(item)
    if data is None:
        data = np.zeros((out_h, out_w), dtype=np.float32)

    # Mosaic / fill nodata if primary item had missing/zero pixels
    if np.any(data == 0) and fallback_items:
        for alt_item in fallback_items:
            if np.mean(data == 0) < 0.001:
                break
            alt_data = _read_item_band(alt_item)
            if alt_data is not None:
                zero_mask = (data == 0) & (alt_data > 0)
                data[zero_mask] = alt_data[zero_mask]

    return data


def _stretch_channel(ch, p_low=2.0, p_high=98.0, gamma=1.25):
    """
    Adaptive percentile contrast stretch for satellite optical imagery.
    Removes atmospheric noise and stretches 2% to 98% non-zero pixels.
    """
    valid = ch[ch > 0]
    if len(valid) == 0:
        return np.zeros_like(ch, dtype=np.uint8)
    vmin = float(np.percentile(valid, p_low))
    vmax = float(np.percentile(valid, p_high))
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((ch - vmin) / (vmax - vmin), 0.0, 1.0)
    scaled = np.power(scaled, 1.0 / gamma)
    return (scaled * 255.0).astype(np.uint8)


def _rgb_from_ch_arrays(r, g, b):
    """
    Build high-definition, sharp, crystal-clear base64 RGB PNG:
    - Balanced 2%-98% percentile dynamic range stretch
    - Unsharp mask filter to eliminate atmospheric haze & sensor blur
    - Micro-contrast sharpness boost for field boundaries, roads, rivers
    """
    from PIL import ImageEnhance, ImageFilter

    r_u8 = _stretch_channel(r)
    g_u8 = _stretch_channel(g)
    b_u8 = _stretch_channel(b)
    rgb = np.stack([r_u8, g_u8, b_u8], axis=2)
    img = Image.fromarray(rgb, mode="RGB")

    # 1. Natural color vibrance boost (1.20)
    img = ImageEnhance.Color(img).enhance(1.20)

    # 2. Atmospheric haze removal / Unsharp mask to make edges crisp
    img = img.filter(ImageFilter.UnsharpMask(radius=1.3, percent=125, threshold=2))

    # 3. Micro-contrast crispness
    img = ImageEnhance.Sharpness(img).enhance(1.20)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    return base64.b64encode(buf.getvalue()).decode()



def _load_s1_means():
    """
    Load S1 channel means from norm_stats_trainonly.json.
    Used to fill S1 channels when Sentinel-1 is unavailable.
    """
    norm_path = os.environ.get(
        "NORM_STATS_PATH",
        os.path.join(os.path.dirname(__file__),
                     "Model", "geonexus_v3_2_k30", "norm_stats_trainonly.json"),
    )
    try:
        with open(norm_path) as f:
            stats = json.load(f)
        means = stats.get("mean", [])
        # channels 13, 14, 15 = S1 VV, VH, VH_VV
        s1_means = [float(means[i]) if i < len(means) else 0.0
                    for i in [13, 14, 15]]
        n_clear_mean = float(means[16]) if len(means) > 16 else 5.0
        return s1_means, n_clear_mean
    except Exception as e:
        print(f"  [STAC] Could not load norm_stats for S1 fill: {e}")
        return [0.0, 0.0, 0.0], 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API — same interface as gee_service.fetch_imagery
# ─────────────────────────────────────────────────────────────────────────────

def fetch_imagery(
    lon_min, lat_min, lon_max, lat_max,
    date_t1, date_t2,
    window_days=15, max_cloud=30,
    thumb_size=512,
    progress_cb=None,
):
    """
    Fetch Sentinel-2 imagery for T1 and T2 using the STAC API.
    Fetches all 11 bands concurrently via ThreadPoolExecutor for high speed.
    Returns same dict as gee_service.fetch_imagery:
      {
        t1_tensor: np.ndarray [17, H, W] float32,
        t2_tensor: np.ndarray [17, H, W] float32,
        t1_rgb_b64: str (base64 PNG),
        t2_rgb_b64: str (base64 PNG),
        meta: dict,
      }
    """
    from concurrent.futures import ThreadPoolExecutor

    bbox = dict(lon_min=lon_min, lat_min=lat_min,
                lon_max=lon_max, lat_max=lat_max)

    if progress_cb:
        progress_cb("🛰 Searching Sentinel-2 scenes for T1 & T2…")

    print(f"[STAC] Searching T1 scene around {date_t1} ±{window_days}d …")
    item_t1, fallbacks_t1 = _search_best_scene(bbox, date_t1, window_days, max_cloud)
    print(f"[STAC] T1 → {item_t1.id}  cloud={item_t1.properties.get('eo:cloud_cover','-')}%")

    print(f"[STAC] Searching T2 scene around {date_t2} ±{window_days}d …")
    item_t2, fallbacks_t2 = _search_best_scene(bbox, date_t2, window_days, max_cloud)
    print(f"[STAC] T2 → {item_t2.id}  cloud={item_t2.properties.get('eo:cloud_cover','-')}%")

    s1_means, n_clear_mean = _load_s1_means()
    target_shape = _calculate_optimal_dimensions(bbox, base_dim=thumb_size)
    print(f"[STAC] Target spatial shape for bbox: {target_shape[0]}×{target_shape[1]} (HxW)")

    def build_tensor_and_rgb(item, fallbacks, scene_label):
        if progress_cb:
            progress_cb(f"🛰 Downloading 11 spectral bands for {scene_label} in parallel…")

        def fetch_band(key):
            return key, _get_band(item, key, bbox, target_shape, fallback_items=fallbacks)

        with ThreadPoolExecutor(max_workers=4) as executor:
            fetched = dict(executor.map(fetch_band, S2_ASSET_KEYS))

        bands = [fetched[key] for key in S2_ASSET_KEYS]
        stacked = np.stack(bands, axis=0).astype(np.float32)  # [11, H, W]

        # ch2=red (B04), ch1=green (B03), ch0=blue (B02)
        rgb_b64 = _rgb_from_ch_arrays(stacked[2], stacked[1], stacked[0])

        # Channels 11–12: NDVI, NDBI
        nir  = stacked[6]  # B08
        red  = stacked[2]  # B04
        swir = stacked[9]  # B11

        denom_ndvi = nir + red + 1e-6
        ndvi = (nir - red) / denom_ndvi * 10000

        denom_ndbi = swir + nir + 1e-6
        ndbi = (swir - nir) / denom_ndbi * 10000

        # Channels 13–15: S1 (fill with distribution means)
        H, W = stacked.shape[1], stacked.shape[2]
        s1_vv  = np.full((H, W), s1_means[0] * 100, dtype=np.float32)
        s1_vh  = np.full((H, W), s1_means[1] * 100, dtype=np.float32)
        s1_rat = np.full((H, W), s1_means[2] * 100, dtype=np.float32)

        # Channel 16: n_clear (observation count)
        n_clear = np.full((H, W), n_clear_mean, dtype=np.float32)

        tensor = np.concatenate([
            stacked,                       # ch 0-10
            ndvi[np.newaxis],              # ch 11
            ndbi[np.newaxis],              # ch 12
            s1_vv[np.newaxis],             # ch 13
            s1_vh[np.newaxis],             # ch 14
            s1_rat[np.newaxis],            # ch 15
            n_clear[np.newaxis],           # ch 16
        ], axis=0)  # [17, H, W]

        return tensor.astype(np.float32), rgb_b64

    print("[STAC] Building T1 tensor & RGB …")
    t1_tensor, t1_rgb = build_tensor_and_rgb(item_t1, fallbacks_t1, "T1 (Before)")

    print("[STAC] Building T2 tensor & RGB …")
    t2_tensor, t2_rgb = build_tensor_and_rgb(item_t2, fallbacks_t2, "T2 (After)")

    meta = {
        "source": "Sentinel-2 L2A (European Space Agency / Open Data)",
        "t1_scene_id": item_t1.id,
        "t1_cloud_pct": item_t1.properties.get("eo:cloud_cover", "N/A"),
        "t1_datetime": str(item_t1.datetime or item_t1.properties.get("datetime", "")),
        "t2_scene_id": item_t2.id,
        "t2_cloud_pct": item_t2.properties.get("eo:cloud_cover", "N/A"),
        "t2_datetime": str(item_t2.datetime or item_t2.properties.get("datetime", "")),
        "s1_used": False,
        "s1_note": "S1 channels filled with training-distribution mean",
        "thumb_size": thumb_size,
    }

    return dict(
        t1_tensor=t1_tensor,
        t2_tensor=t2_tensor,
        t1_rgb_b64=t1_rgb,
        t2_rgb_b64=t2_rgb,
        meta=meta,
    )
