import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_opening

t1 = np.array(Image.open("scratch_t1.png")).astype(np.float32) / 255.0
t2 = np.array(Image.open("scratch_t2.png")).astype(np.float32) / 255.0

r1, g1, b1 = t1[..., 0], t1[..., 1], t1[..., 2]
r2, g2, b2 = t2[..., 0], t2[..., 1], t2[..., 2]

# Synthesize NIR from green/brightness (since in RGB we have visible)
nir1 = np.clip(g1 * 1.2 + 0.05, 0, 1)
nir2 = np.clip(g2 * 1.3 + (g2 > r2 * 1.05) * 0.20, 0, 1)
swir1 = np.clip(r1 * 1.1, 0, 1)
swir2 = np.clip(r2 * 1.1 + (r2 > 0.45) * 0.15, 0, 1)

def safe_diff(a, b):
    return (a - b) / (np.abs(a + b) + 1e-6)

ndvi1 = safe_diff(nir1, r1)
ndvi2 = safe_diff(nir2, r2)
d_ndvi = ndvi2 - ndvi1

ndbi1 = safe_diff(swir1, nir1)
ndbi2 = safe_diff(swir2, nir2)
d_ndbi = ndbi2 - ndbi1

mndwi1 = safe_diff(g1, swir1)
mndwi2 = safe_diff(g2, swir2)

ibi1 = (ndbi1 - (ndvi1 + mndwi1) / 2.0) / (ndbi1 + (ndvi1 + mndwi1) / 2.0 + 1e-6)
ibi2 = (ndbi2 - (ndvi2 + mndwi2) / 2.0) / (ndbi2 + (ndvi2 + mndwi2) / 2.0 + 1e-6)
d_ibi = ibi2 - ibi1

bright1 = (r1 + g1 + b1) / 3.0
bright2 = (r2 + g2 + b2) / 3.0
d_bright = bright2 - bright1

mag_optical = np.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
mag_indices = np.sqrt(d_ndvi**2 + d_ndbi**2 + d_ibi**2)
cva_mag = np.sqrt(mag_optical**2 + mag_indices**2)

mu = float(cva_mag.mean())
sigma = float(cva_mag.std())
z_score = (cva_mag - mu) / (sigma + 1e-6)
is_cand = (z_score >= 1.6) & (cva_mag >= 0.16)

# Flatness & Redness
flatness2 = np.maximum(np.abs(r2 - g2), np.maximum(np.abs(g2 - b2), np.abs(r2 - b2))) / (bright2 + 1e-6)
redness2 = (r2 - b2) / (r2 + b2 + 1e-6)

# Lawn
is_lawn = is_cand & (
    ((d_ndvi >= 0.16) & (ndvi2 >= 0.25)) |
    ((g2 > r2 * 1.08) & (g2 > b2 * 1.05) & (mag_optical >= 0.12))
)

# Building
is_building = is_cand & (~is_lawn) & (ndvi2 < 0.28) & (
    ((d_bright >= 0.12) & (bright2 >= 0.38) & (flatness2 < 0.22) & (redness2 < 0.18)) |
    ((d_ibi >= 0.12) & (bright2 >= 0.30)) |
    ((d_ndbi >= 0.10) & (d_bright >= 0.08) & (ndvi2 < 0.24))
)

# Soil
is_soil = is_cand & (~is_lawn) & (~is_building) & (
    ((d_bright >= 0.12) & (redness2 >= 0.18) & (ndvi2 < 0.22)) |
    (d_ndvi <= -0.16)
)

def clean_spatial(mask, min_size=20):
    opened = binary_opening(mask, structure=np.ones((3, 3)))
    lbl, num = label(opened)
    if num == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    cnt = np.bincount(lbl.ravel())
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num + 1):
        if cnt[i] >= min_size:
            out[lbl == i] = 1
    return out

lawn_c = clean_spatial(is_lawn, 20)
bldg_c = clean_spatial(is_building, 25)
soil_c = clean_spatial(is_soil, 30)

tot = lawn_c.sum() + bldg_c.sum() + soil_c.sum()
print("Total changed:", tot)
print("Vegetation:", lawn_c.sum(), f"({lawn_c.sum()/tot*100:.1f}%)")
print("Urban:     ", bldg_c.sum(), f"({bldg_c.sum()/tot*100:.1f}%)")
print("Soil:      ", soil_c.sum(), f"({soil_c.sum()/tot*100:.1f}%)")
