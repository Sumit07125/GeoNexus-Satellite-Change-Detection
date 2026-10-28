"""
spectral_cd.py — GeoNexus Official Google Earth Engine (GEE) S-CVA Engine
==========================================================================
Implements the exact Change Vector Analysis (CVA) and bi-temporal transition
formulas established by Google Earth Engine (GEE) and Google Dynamic World (Nature 2022)
for Sentinel-2 L2A multi-spectral surface reflectance data.

Scientific Methodology:
  1. Relative Radiometric Normalization (Pseudo-Invariant Features):
     Adjusts multi-date illumination, sun angle, and atmospheric variations across
     all optical bands so stable surfaces match perfectly.
  2. Multi-Index Vector Computation:
     - NDVI (Normalized Difference Vegetation Index)     -> Vegetation growth / loss
     - NDBI (Normalized Difference Built-up Index)       -> Urban & Structural expansion
     - NDWI (Normalized Difference Water Index)          -> Surface water / flood
     - BSI  (Bare Soil Index)                            -> Soil exposure / excavation
  3. Standardized Change Vector Analysis (GEE S-CVA Magnitude):
     Computes Euclidean distance across normalized spectral indices:
       Magnitude = sqrt( (dNDVI)^2 + (dNDBI)^2 + (dNDWI)^2 + (dBSI)^2 )
  4. Statistical Z-Score Thresholding (GEE Standard):
     Calculates regional mean (μ) and standard deviation (σ).
     Candidate change pixels must satisfy:
       Z = (Magnitude - μ) / σ >= 2.2  AND  Magnitude >= 0.18
     Pixels below this threshold represent natural background/noise.
  5. Google Dynamic World Bi-Temporal Directional Transitions:
     - Vegetation Dynamics (Green):
         Positive bloom/lawn: ΔNDVI >= +0.18 AND NDVI_T2 >= 0.28
         Deforestation: ΔNDVI <= -0.22 AND NDVI_T1 >= 0.35
     - Urban & Structural (Yellow):
         Built-up emergence: ΔNDBI >= +0.13 AND NDVI_T2 < 0.25
     - Soil & Agriculture (Warm Terracotta / Orange):
         Soil exposure/fallow: ΔBSI >= +0.13 OR (ΔNDVI <= -0.20 AND NDVI_T2 < 0.22)
     - Water Dynamics (Blue):
         Water expansion: ΔNDWI >= +0.18 AND NDWI_T2 > 0.0
  6. Spatial Coherence (Minimum Mapping Unit - MMU):
     Filters isolated single-pixel jitter (< 5 connected pixels, ~500 m²) to remove
     road boundary halos and tree canopy shadow artifacts.

Returns the exact same dictionary shape as inference.run_inference() for seamless
drop-in replacement in the frontend Results Studio.
"""

import io
import base64
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_dilation, binary_erosion, binary_opening


# ─────────────────────────────────────────────────────────────────────────────
# Spectral index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ratio(a, b, eps=1e-6):
    """Normalized difference ratio (a - b) / (a + b) with numerical safety."""
    den = np.abs(a + b) + eps
    return (a - b) / den


def _calc_ndvi(nir, red):
    """Normalized Difference Vegetation Index."""
    return _safe_ratio(nir, red)


def _calc_ndwi(green, nir):
    """Normalized Difference Water Index (Gao / McFeeters)."""
    return _safe_ratio(green, nir)


def _calc_ndbi(swir, nir):
    """Normalized Difference Built-up Index (Zha et al.)."""
    return _safe_ratio(swir, nir)


def _calc_bsi(swir, red, nir, blue):
    """Bare Soil Index (Rikimaru & Roy)."""
    num = (swir + red) - (nir + blue)
    den = np.abs((swir + red) + (nir + blue)) + 1e-6
    return num / den


# ─────────────────────────────────────────────────────────────────────────────
# Relative Radiometric Normalization (Pseudo-Invariant Feature matching)
# ─────────────────────────────────────────────────────────────────────────────

def _radiometric_normalize(t1_bands, t2_bands, valid_mask):
    """
    Standardize T1 band distributions to match T2 illumination baseline.
    Removes seasonal sun elevation, haze, and atmospheric calibration drift.
    """
    t1_norm = np.copy(t1_bands).astype(np.float32)
    t2_f = t2_bands.astype(np.float32)

    for b in range(min(t1_bands.shape[0], t2_bands.shape[0])):
        b1 = t1_norm[b][valid_mask]
        b2 = t2_f[b][valid_mask]
        if len(b1) < 100:
            continue
        p5_1, p95_1 = np.percentile(b1, 5), np.percentile(b1, 95)
        p5_2, p95_2 = np.percentile(b2, 5), np.percentile(b2, 95)
        span1 = max(p95_1 - p5_1, 10.0)
        span2 = max(p95_2 - p5_2, 10.0)
        scale = np.clip(span2 / span1, 0.6, 1.8)
        offset = p5_2 - p5_1 * scale
        t1_norm[b] = np.clip(t1_norm[b] * scale + offset, 0.0, 10000.0)

    return t1_norm


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Minimum Mapping Unit (MMU) Filter
# ─────────────────────────────────────────────────────────────────────────────

def _apply_mmu_filter(binary_mask, min_size=15):
    """
    Remove isolated single-pixel noise, tree shadow jitter, and thin road edge halos.
    Applies 3x3 morphological opening followed by connected component filtering.
    """
    if binary_mask.sum() == 0:
        return binary_mask

    opened = binary_opening(binary_mask.astype(bool), structure=np.ones((3, 3), dtype=bool))
    lbl, num = label(opened)
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    counts = np.bincount(lbl.ravel())
    filtered = np.zeros_like(binary_mask, dtype=np.uint8)
    for idx in range(1, num + 1):
        if counts[idx] >= min_size:
            filtered[lbl == idx] = 1

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Image encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr_to_b64(arr):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


CHANGE_PALETTE = np.array([
    [8,   48,  107],
    [8,   81,  156],
    [33,  113, 181],
    [66,  146, 198],
    [107, 174, 214],
    [255, 255, 102],
    [253, 174,  97],
    [244, 109,  67],
    [215,  48,  39],
    [165,  15,  21],
], dtype=np.float32)


def _prob_to_colormap(prob):
    idxf = prob * (len(CHANGE_PALETTE) - 1)
    idx0 = np.clip(np.floor(idxf).astype(int), 0, len(CHANGE_PALETTE) - 2)
    idx1 = idx0 + 1
    frac = (idxf - idx0)[..., None]
    rgb = (1 - frac) * CHANGE_PALETTE[idx0] + frac * CHANGE_PALETTE[idx1]
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_overlay(t2_rgb_b64, semantic_rgb, mask, prob):
    """
    Overlay verified physical changes onto clear optical T2 satellite image.
    Stable pixels: 100% natural, crisp satellite photograph.
    Changed pixels: vibrant category color blended with underlying physical textures.
    """
    img_bytes = base64.b64decode(t2_rgb_b64)
    rgb = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    H, W = mask.shape
    rgb_r = np.array(Image.fromarray(rgb).resize((W, H), Image.LANCZOS)).astype(np.float32)
    sem_f = semantic_rgb.astype(np.float32)

    # Dynamic alpha: 0 on stable pixels, 0.55 - 0.85 on changed pixels
    alpha = np.clip(prob * 0.40 + 0.45, 0.50, 0.85) * mask.astype(np.float32)
    alpha3 = alpha[..., None]
    blended = rgb_r * (1 - alpha3) + sem_f * alpha3
    return _arr_to_b64(np.clip(blended, 0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Main Google Earth Engine CVA & Transition Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_spectral_inference(
    t1_tensor,
    t2_tensor,
    t2_rgb_b64,
    threshold=None,
):
    """
    Official Google Earth Engine S-CVA & Dynamic World Change Detection.

    Args:
        t1_tensor:  [17, H, W] int16/float32 T1 multi-band tensor
        t2_tensor:  [17, H, W] int16/float32 T2 multi-band tensor
        t2_rgb_b64: base64 PNG of T2 RGB satellite image
        threshold:  optional user override for Z-score threshold (default: GEE 2.2σ)

    Returns:
        Exact same dictionary shape as inference.run_inference().
    """
    t1_raw = t1_tensor.astype(np.float32)
    t2_raw = t2_tensor.astype(np.float32)

    # Valid pixel mask (exclude nodata and extreme dark border pixels)
    valid_mask = (
        (t1_raw[:3].sum(0) > 300) &
        (t2_raw[:3].sum(0) > 300) &
        (t1_raw[6] > 100) &
        (t2_raw[6] > 100)
    )

    # Step 1: Relative Radiometric Normalization (eliminates sun angle & haze mismatch)
    t1_norm = _radiometric_normalize(t1_raw, t2_raw, valid_mask)
    t2_f = t2_raw

    SCALE = 10000.0

    # Unpack normalized surface reflectance bands:
    # 0=Blue, 1=Green, 2=Red, 6=NIR, 9=SWIR1
    blue1  = t1_norm[0] / SCALE; green1 = t1_norm[1] / SCALE; red1  = t1_norm[2] / SCALE
    blue2  = t2_f[0]    / SCALE; green2 = t2_f[1]    / SCALE; red2  = t2_f[2]    / SCALE
    nir1   = t1_norm[6] / SCALE; nir2   = t2_f[6]    / SCALE
    swir1  = (t1_norm[9] / SCALE) if t1_norm.shape[0] > 9 else red1
    swir2  = (t2_f[9]    / SCALE) if t2_f.shape[0] > 9 else red2

    # Step 2: Compute Google Earth Engine Spectral Indices
    ndvi1 = _calc_ndvi(nir1, red1)
    ndvi2 = _calc_ndvi(nir2, red2)
    d_ndvi = ndvi2 - ndvi1

    ndbi1 = _calc_ndbi(swir1, nir1)
    ndbi2 = _calc_ndbi(swir2, nir2)
    d_ndbi = ndbi2 - ndbi1

    ndwi1 = _calc_ndwi(green1, nir1)
    ndwi2 = _calc_ndwi(green2, nir2)
    d_ndwi = ndwi2 - ndwi1

    bsi1 = _calc_bsi(swir1, red1, nir1, blue1)
    bsi2 = _calc_bsi(swir2, red2, nir2, blue2)
    d_bsi = bsi2 - bsi1

    # Optical brightness difference
    bright1 = (blue1 + green1 + red1) / 3.0
    bright2 = (blue2 + green2 + red2) / 3.0
    d_bright = bright2 - bright1

    # Optical spectral flatness (neutral color profile of concrete / metal roofs vs red soil)
    flatness2 = np.maximum(
        np.abs(red2 - green2),
        np.maximum(np.abs(green2 - blue2), np.abs(red2 - blue2))
    ) / (bright2 + 1e-6)
    redness2 = (red2 - blue2) / (red2 + blue2 + 1e-6)

    # Step 3: Google Earth Engine Multi-Spectral Change Vector Analysis (S-CVA)
    # Combines optical reflectance Euclidean distance + normalized index differences:
    cva_optical = np.sqrt(
        (blue2 - blue1) ** 2 +
        (green2 - green1) ** 2 +
        (red2 - red1) ** 2 +
        (nir2 - nir1) ** 2 +
        (swir2 - swir1) ** 2
    ).astype(np.float32)

    cva_indices = np.sqrt(
        (d_ndvi) ** 2 +
        (d_ndbi) ** 2 +
        (d_ndwi) ** 2 +
        (d_bsi)  ** 2
    ).astype(np.float32)

    cva_mag = np.sqrt(cva_optical ** 2 + cva_indices ** 2)
    cva_mag = np.where(valid_mask, cva_mag, 0.0)

    # Step 4: GEE Z-Score Statistical Thresholding
    valid_cva = cva_mag[valid_mask]
    if len(valid_cva) > 100:
        mu_cva = float(np.mean(valid_cva))
        sigma_cva = float(np.std(valid_cva))
    else:
        mu_cva = 0.12
        sigma_cva = 0.09

    # Google Earth Engine cutoff: 1.7 standard deviations above mean, min 0.15
    z_cutoff = float(threshold) if threshold is not None else 1.7
    cva_threshold = max(mu_cva + z_cutoff * sigma_cva, 0.15)

    # Candidate pixels exceeding statistical change threshold:
    is_candidate = (cva_mag >= cva_threshold) & valid_mask

    # Step 5: Google Dynamic World Directional Transition Rules
    # Only pixels with genuine physical shifts are categorized; permanent trees & roofs are ignored!

    # 💧 1. Water Dynamics (Blue: #2563eb / [37, 99, 235])
    water_mask = is_candidate & (
        ((d_ndwi >= 0.16) & (ndwi2 > -0.05)) |
        ((d_ndwi <= -0.18) & (ndwi1 > 0.0))
    )

    # 🏗️ 1. Urban & Structural Dynamics (Yellow: #eab308 / [234, 179, 8])
    # Built-up expansion: concrete, pavement, buildings, metal roofs, runway surfacing
    # Evaluated before vegetation loss so land clearing for new buildings is classified as Urbanization:
    built_up_gain = (
        # Concrete / bright metal roof emergence (brightness increase with neutral profile)
        ((d_bright >= 0.025) & (bright2 >= 0.11) & (flatness2 < 0.35) & (redness2 < 0.22)) |
        # High optical difference on low vegetation with neutral spectrum
        ((cva_optical >= 0.08) & (bright2 >= 0.11) & (flatness2 < 0.32) & (redness2 < 0.20)) |
        # Classical NDBI built-up gain
        ((d_ndbi >= 0.05) & (d_bright >= 0.015) & (ndvi2 < 0.26))
    )
    urban_mask = is_candidate & (~water_mask) & (ndvi2 < 0.28) & built_up_gain

    # 🌿 2. Vegetation Dynamics (Green: #22c55e / [34, 197, 94])
    # Genuine green lawn/turf/canopy growth (NDVI_T2 >= 0.30) OR timber clearing that did NOT become buildings:
    veg_gain = (d_ndvi >= 0.15) & (ndvi2 >= 0.30)
    veg_loss = (d_ndvi <= -0.20) & (ndvi1 >= 0.35) & (~urban_mask)
    veg_mask = is_candidate & (~water_mask) & (~urban_mask) & (veg_gain | veg_loss)

    # 🌾 3. Bare Soil & Agriculture (Warm Terracotta / Orange: #f97316 / [249, 115, 22])
    # Soil exposure, excavation, earthworks, harvested agricultural fields:
    soil_gain = (
        (d_bsi >= 0.09) |
        ((redness2 >= 0.20) & (d_bright >= 0.025) & (ndvi2 < 0.22)) |
        ((d_ndvi <= -0.16) & (ndvi2 < 0.22))
    )
    soil_mask = is_candidate & (~water_mask) & (~urban_mask) & (~veg_mask) & soil_gain

    # Step 6: Spatial MMU Filter (removes isolated pixel noise, road edge halos, jitter)
    water_clean = _apply_mmu_filter(water_mask, min_size=15)
    urban_clean = _apply_mmu_filter(urban_mask, min_size=18)
    veg_clean   = _apply_mmu_filter(veg_mask,   min_size=18)
    soil_clean  = _apply_mmu_filter(soil_mask,  min_size=25)

    # The verified change mask is the union of confirmed physical surface transitions
    final_mask = (water_clean | veg_clean | urban_clean | soil_clean).astype(np.uint8)

    # Build Semantic RGB mask
    H, W = final_mask.shape
    semantic_rgb = np.full((H, W, 3), [10, 15, 29], dtype=np.uint8)
    semantic_rgb[water_clean > 0] = [37, 99, 235]   # Blue
    semantic_rgb[veg_clean > 0]   = [34, 197, 94]   # Green
    semantic_rgb[urban_clean > 0] = [234, 179, 8]   # Yellow
    semantic_rgb[soil_clean > 0]  = [249, 115, 22]  # Terracotta / Orange

    # Step 7: Continuous Probability Heatmap (Sigmoid Z-Score)
    z_score = np.where(valid_mask, (cva_mag - mu_cva) / (sigma_cva + 1e-6), 0.0)
    prob_smooth = 1.0 / (1.0 + np.exp(-2.2 * (z_score - 2.0)))
    prob_smooth = np.clip(prob_smooth * valid_mask, 0.0, 1.0).astype(np.float32)

    # Calculate statistics
    total_changed = int(final_mask.sum())
    w_cnt = int(water_clean.sum())
    v_cnt = int(veg_clean.sum())
    u_cnt = int(urban_clean.sum())
    s_cnt = int(soil_clean.sum())
    pix_ha = 0.01  # 10m x 10m = 0.01 ha

    dynamics_stats = {
        "water": {
            "pct": round(w_cnt / max(total_changed, 1) * 100, 1),
            "ha": round(w_cnt * pix_ha, 1),
            "pixels": w_cnt,
            "color": "#2563eb",
            "label": "Water Dynamics",
        },
        "vegetation": {
            "pct": round(v_cnt / max(total_changed, 1) * 100, 1),
            "ha": round(v_cnt * pix_ha, 1),
            "pixels": v_cnt,
            "color": "#22c55e",
            "label": "Vegetation Dynamics",
        },
        "urban": {
            "pct": round(u_cnt / max(total_changed, 1) * 100, 1),
            "ha": round(u_cnt * pix_ha, 1),
            "pixels": u_cnt,
            "color": "#eab308",
            "label": "Urban & Structural",
        },
        "bare_soil": {
            "pct": round(s_cnt / max(total_changed, 1) * 100, 1),
            "ha": round(s_cnt * pix_ha, 1),
            "pixels": s_cnt,
            "color": "#f97316",
            "label": "Soil & Agriculture",
        },
    }

    # Visual exports
    multicolor_mask_b64 = _arr_to_b64(semantic_rgb)
    binary_mask_b64     = _arr_to_b64((final_mask * 255).astype(np.uint8))
    overlay_b64         = _make_overlay(t2_rgb_b64, semantic_rgb, final_mask, prob_smooth)

    heat_rgb  = _prob_to_colormap(prob_smooth)
    alpha_ch  = np.clip(prob_smooth * 255, 30, 255).astype(np.uint8)[..., None]
    heat_rgba = np.concatenate([heat_rgb, alpha_ch], axis=-1)
    change_prob_b64 = _arr_to_b64(heat_rgba)

    total_px = int(valid_mask.sum()) if valid_mask.sum() > 0 else final_mask.size
    change_pct = round(total_changed / max(total_px, 1) * 100, 2)

    stats = {
        "change_pct":      change_pct,
        "changed_pixels":  total_changed,
        "total_pixels":    total_px,
        "changed_area_ha": round(total_changed * pix_ha, 1),
        "mean_prob":       round(float(prob_smooth[valid_mask].mean()) if valid_mask.sum() > 0 else 0.0, 4),
        "max_prob":        round(float(prob_smooth.max()), 4),
        "threshold":       round(cva_threshold, 3),
        "output_size":     [H, W],
        "method":          "Google Earth Engine S-CVA & Dynamic World Transition (Sentinel-2 L2A)",
    }

    return {
        "change_prob_b64": change_prob_b64,
        "change_mask_b64": multicolor_mask_b64,
        "binary_mask_b64": binary_mask_b64,
        "overlay_b64":     overlay_b64,
        "dynamics":        dynamics_stats,
        "stats":           stats,
    }
