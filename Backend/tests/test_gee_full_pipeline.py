import os
import json
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_opening

from sentinel_stac_service import fetch_imagery
from spectral_cd import _radiometric_normalize

bbox = {
    "lon_min": 74.76297167861242,
    "lat_min": 20.86412777761325,
    "lon_max": 74.77258545692956,
    "lat_max": 20.87311088936316
}
date_t1 = "2021-03-15"
date_t2 = "2026-03-15"

imagery = fetch_imagery(
    lon_min=bbox["lon_min"], lat_min=bbox["lat_min"],
    lon_max=bbox["lon_max"], lat_max=bbox["lat_max"],
    date_t1=date_t1, date_t2=date_t2,
    window_days=15, max_cloud=30, thumb_size=512
)

t1_raw = imagery["t1_tensor"].astype(np.float32)
t2_raw = imagery["t2_tensor"].astype(np.float32)

valid_mask = (
    (t1_raw[:3].sum(0) > 300) &
    (t2_raw[:3].sum(0) > 300) &
    (t1_raw[6] > 100) &
    (t2_raw[6] > 100)
)

t1_norm = _radiometric_normalize(t1_raw, t2_raw, valid_mask)
SCALE = 10000.0

blue1  = t1_norm[0] / SCALE; green1 = t1_norm[1] / SCALE; red1  = t1_norm[2] / SCALE
blue2  = t2_raw[0]  / SCALE; green2 = t2_raw[1]  / SCALE; red2  = t2_raw[2]  / SCALE
nir1   = t1_norm[6] / SCALE; nir2   = t2_raw[6]  / SCALE
swir1_1= t1_norm[9] / SCALE; swir1_2= t2_raw[9]  / SCALE
swir2_1= t1_norm[10]/ SCALE; swir2_2= t2_raw[10] / SCALE

bright1 = (blue1 + green1 + red1) / 3.0
bright2 = (blue2 + green2 + red2) / 3.0
d_bright = bright2 - bright1

print("Bright1 range:", bright1.min(), bright1.max(), "mean:", bright1.mean())
print("Bright2 range:", bright2.min(), bright2.max(), "mean:", bright2.mean())
print("d_bright range:", d_bright.min(), d_bright.max(), "mean:", d_bright.mean())

def safe_diff(a, b):
    return (a - b) / (np.abs(a + b) + 1e-6)

ndvi1 = safe_diff(nir1, red1)
ndvi2 = safe_diff(nir2, red2)
d_ndvi = ndvi2 - ndvi1

ndbi1 = safe_diff(swir1_1, nir1)
ndbi2 = safe_diff(swir1_2, nir2)
d_ndbi = ndbi2 - ndbi1

mndwi1 = safe_diff(green1, swir1_1)
mndwi2 = safe_diff(green2, swir1_2)

# Index-based Built-up Index (IBI)
ibi1 = (ndbi1 - (ndvi1 + mndwi1) / 2.0) / (ndbi1 + (ndvi1 + mndwi1) / 2.0 + 1e-6)
ibi2 = (ndbi2 - (ndvi2 + mndwi2) / 2.0) / (ndbi2 + (ndvi2 + mndwi2) / 2.0 + 1e-6)
d_ibi = ibi2 - ibi1

# Flatness & Redness
flatness2 = np.maximum(
    np.abs(red2 - green2),
    np.maximum(np.abs(green2 - blue2), np.abs(red2 - blue2))
) / (bright2 + 1e-6)
redness1 = (red1 - blue1) / (red1 + blue1 + 1e-6)
redness2 = (red2 - blue2) / (red2 + blue2 + 1e-6)

# Multi-spectral CVA Magnitude
cva_optical = np.sqrt(
    (blue2 - blue1)**2 +
    (green2 - green1)**2 +
    (red2 - red1)**2 +
    (nir2 - nir1)**2 +
    (swir1_2 - swir1_1)**2
)

cva_indices = np.sqrt(
    d_ndvi**2 +
    d_ndbi**2 +
    (mndwi2 - mndwi1)**2
)

cva_total = np.sqrt(cva_optical**2 + cva_indices**2)

mu = float(cva_total[valid_mask].mean())
sigma = float(cva_total[valid_mask].std())
z_score = (cva_total - mu) / (sigma + 1e-6)
is_change_cand = (z_score >= 1.5) & (cva_total >= 0.14) & valid_mask

# 1. Vegetation Dynamics (Green: #22c55e)
veg_gain = (d_ndvi >= 0.16) & (ndvi2 >= 0.25)
veg_loss = (d_ndvi <= -0.18) & (ndvi1 >= 0.32)
veg_mask = is_change_cand & (veg_gain | veg_loss)

# 2. Urban / Structural Dynamics (Yellow: #eab308)
# Brightness in reflectance units: bright2 >= 0.14 (1400 DN), d_bright >= 0.05 (500 DN)
urban_cand = (
    (~veg_mask) &
    (ndvi2 < 0.28) &
    (
        ((d_bright >= 0.05) & (bright2 >= 0.14) & (flatness2 < 0.30) & (redness2 < 0.22)) |
        ((d_ibi >= 0.10) & (bright2 >= 0.12)) |
        ((d_ndbi >= 0.08) & (d_bright >= 0.04) & (ndvi2 < 0.24))
    )
)
urban_mask = is_change_cand & urban_cand

# 3. Soil / Agriculture (Terracotta: #f97316)
soil_cand = (
    (~veg_mask) &
    (~urban_mask) &
    (
        ((redness2 >= 0.22) & (d_bright >= 0.05) & (ndvi2 < 0.22)) |
        (d_ndvi <= -0.16)
    )
)
soil_mask = is_change_cand & soil_cand

def clean_mask(m, min_size=15):
    opened = binary_opening(m, structure=np.ones((3, 3)))
    lbl, num = label(opened)
    if num == 0:
        return np.zeros_like(m, dtype=np.uint8)
    cnt = np.bincount(lbl.ravel())
    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num + 1):
        if cnt[i] >= min_size:
            out[lbl == i] = 1
    return out

veg_clean = clean_mask(veg_mask, min_size=20)
urban_clean = clean_mask(urban_mask, min_size=20)
soil_clean = clean_mask(soil_mask, min_size=25)

tot = veg_clean.sum() + urban_clean.sum() + soil_clean.sum()
print("=== RESULTS WITH CORRECT S2 BOA REFLECTANCE THRESHOLDS ===")
print("Vegetation pixels:", veg_clean.sum(), f"({veg_clean.sum()/max(tot,1)*100:.1f}%)")
print("Urban pixels:     ", urban_clean.sum(), f"({urban_clean.sum()/max(tot,1)*100:.1f}%)")
print("Soil pixels:      ", soil_clean.sum(), f"({soil_clean.sum()/max(tot,1)*100:.1f}%)")

# Save overlay
t2_b64 = imagery["t2_rgb_b64"]
import base64, io
t2_img = np.array(Image.open(io.BytesIO(base64.b64decode(t2_b64))).convert("RGB")).astype(np.float32) / 255.0

overlay = np.copy(t2_img)
alpha = 0.70
overlay[veg_clean > 0] = overlay[veg_clean > 0] * (1 - alpha) + np.array([34/255, 197/255, 94/255]) * alpha
overlay[urban_clean > 0] = overlay[urban_clean > 0] * (1 - alpha) + np.array([234/255, 179/255, 8/255]) * alpha
overlay[soil_clean > 0] = overlay[soil_clean > 0] * (1 - alpha) + np.array([249/255, 115/255, 22/255]) * alpha

Image.fromarray((overlay * 255).astype(np.uint8)).save("scratch_pipeline_overlay_v2.png")
print("Saved scratch_pipeline_overlay_v2.png successfully!")
