import numpy as np
from PIL import Image

data = np.load("scratch_tensors.npz")
t1_raw = data["t1"]
t2_raw = data["t2"]

from spectral_cd import _radiometric_normalize, _apply_mmu_filter

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
swir1  = t1_norm[9] / SCALE; swir2  = t2_raw[9]  / SCALE

bright1 = (blue1 + green1 + red1) / 3.0
bright2 = (blue2 + green2 + red2) / 3.0
d_bright = bright2 - bright1

def safe_diff(a, b):
    return (a - b) / (np.abs(a + b) + 1e-6)

ndvi1 = safe_diff(nir1, red1)
ndvi2 = safe_diff(nir2, red2)
d_ndvi = ndvi2 - ndvi1

ndbi1 = safe_diff(swir1, nir1)
ndbi2 = safe_diff(swir2, nir2)
d_ndbi = ndbi2 - ndbi1

ndwi1 = safe_diff(green1, nir1)
ndwi2 = safe_diff(green2, nir2)
d_ndwi = ndwi2 - ndwi1

bsi1 = (swir1 + red1 - (nir1 + blue1)) / (swir1 + red1 + nir1 + blue1 + 1e-6)
bsi2 = (swir2 + red2 - (nir2 + blue2)) / (swir2 + red2 + nir2 + blue2 + 1e-6)
d_bsi = bsi2 - bsi1

flatness2 = np.maximum(
    np.abs(red2 - green2),
    np.maximum(np.abs(green2 - blue2), np.abs(red2 - blue2))
) / (bright2 + 1e-6)
redness2 = (red2 - blue2) / (red2 + blue2 + 1e-6)

cva_optical = np.sqrt(
    (blue2 - blue1)**2 + (green2 - green1)**2 + (red2 - red1)**2 + (nir2 - nir1)**2 + (swir2 - swir1)**2
)
cva_indices = np.sqrt(d_ndvi**2 + d_ndbi**2 + d_ndwi**2 + d_bsi**2)
cva_mag = np.sqrt(cva_optical**2 + cva_indices**2)

mu = float(cva_mag[valid_mask].mean())
sigma = float(cva_mag[valid_mask].std())
cva_threshold = max(mu + 1.4 * sigma, 0.14)
is_candidate = (cva_mag >= cva_threshold) & valid_mask

# 1. Water
water_mask = is_candidate & (((d_ndwi >= 0.16) & (ndwi2 > -0.05)) | ((d_ndwi <= -0.18) & (ndwi1 > 0.0)))

# 2. Urban / Structural (Takes precedence over veg loss)
built_up_gain = (
    # Direct optical brightness increase of concrete/roofing
    ((d_bright >= 0.025) & (bright2 >= 0.11) & (flatness2 < 0.35) & (redness2 < 0.22)) |
    # High optical change on low vegetation with neutral profile
    ((cva_optical >= 0.08) & (bright2 >= 0.11) & (flatness2 < 0.32) & (redness2 < 0.20)) |
    # Classic NDBI
    ((d_ndbi >= 0.05) & (d_bright >= 0.015) & (ndvi2 < 0.26))
)
urban_mask = is_candidate & (~water_mask) & (ndvi2 < 0.28) & built_up_gain

# 3. Vegetation
# Genuine green growth (NDVI_T2 >= 0.30)
veg_gain = (d_ndvi >= 0.15) & (ndvi2 >= 0.30)
veg_loss = (d_ndvi <= -0.20) & (ndvi1 >= 0.35) & (~urban_mask)
veg_mask = is_candidate & (~water_mask) & (~urban_mask) & (veg_gain | veg_loss)

# 4. Soil
soil_gain = (
    (d_bsi >= 0.09) |
    ((redness2 >= 0.20) & (d_bright >= 0.025) & (ndvi2 < 0.22)) |
    ((d_ndvi <= -0.16) & (ndvi2 < 0.22))
)
soil_mask = is_candidate & (~water_mask) & (~urban_mask) & (~veg_mask) & soil_gain

water_clean = _apply_mmu_filter(water_mask, min_size=15)
urban_clean = _apply_mmu_filter(urban_mask, min_size=18)
veg_clean   = _apply_mmu_filter(veg_mask,   min_size=18)
soil_clean  = _apply_mmu_filter(soil_mask,  min_size=25)

tot = water_clean.sum() + urban_clean.sum() + veg_clean.sum() + soil_clean.sum()
print("=== CALIBRATED RESULTS ===")
print("Total changed pixels:", tot)
print("Urban (Yellow):     ", urban_clean.sum(), f"({urban_clean.sum()/max(tot,1)*100:.1f}%)")
print("Vegetation (Green): ", veg_clean.sum(), f"({veg_clean.sum()/max(tot,1)*100:.1f}%)")
print("Soil (Terracotta):  ", soil_clean.sum(), f"({soil_clean.sum()/max(tot,1)*100:.1f}%)")
print("Water (Blue):       ", water_clean.sum(), f"({water_clean.sum()/max(tot,1)*100:.1f}%)")

# Make overlay
img2 = Image.open("scratch_t2.png")
rgb_f = np.array(img2).astype(np.float32) / 255.0
overlay = np.copy(rgb_f)
alpha = 0.70
overlay[urban_clean > 0] = overlay[urban_clean > 0] * (1 - alpha) + np.array([234/255, 179/255, 8/255]) * alpha
overlay[veg_clean > 0]   = overlay[veg_clean > 0] * (1 - alpha) + np.array([34/255, 197/255, 94/255]) * alpha
overlay[soil_clean > 0]  = overlay[soil_clean > 0] * (1 - alpha) + np.array([249/255, 115/255, 22/255]) * alpha
overlay[water_clean > 0] = overlay[water_clean > 0] * (1 - alpha) + np.array([37/255, 99/255, 235/255]) * alpha

Image.fromarray((overlay * 255).astype(np.uint8)).save("scratch_calibrated_overlay.png")
print("Saved scratch_calibrated_overlay.png successfully!")
