"""
data/export_qgis.py — QGIS Verification Package Exporter for Geo-Nexus v3.2

Writes one GeoTIFF triplet per patch to verify, plus a QGIS style file (.qml)
so the auto-label renders in standard project colors immediately on load.

Specified in FINAL_ARCH_3.md Part 27.5.
"""
import numpy as np
import rasterio
from rasterio.windows import Window, transform as win_transform
from pathlib import Path

CLASS_LUT = np.array([
    [0, 0, 0],        # 0: no change
    [30, 100, 220],   # 1: water gain
    [80, 200, 220],   # 2: water loss
    [255, 200, 0],    # 3: construction
    [220, 50, 50],    # 4: veg loss
    [50, 180, 50],    # 5: veg gain
    [180, 180, 180],  # 6: other
], dtype=np.uint8)

QML = '''<qgis><pipe><rasterrenderer type="paletted" band="1" opacity="0.5">
<colorPalette>
<paletteEntry value="0" color="#000000" label="no change" alpha="0"/>
<paletteEntry value="1" color="#1E64DC" label="water gain"/>
<paletteEntry value="2" color="#50C8DC" label="water loss"/>
<paletteEntry value="3" color="#FFC800" label="construction"/>
<paletteEntry value="4" color="#DC3232" label="vegetation loss"/>
<paletteEntry value="5" color="#32B432" label="vegetation gain"/>
<paletteEntry value="6" color="#B4B4B4" label="other"/>
<paletteEntry value="255" color="#FF00FF" label="UNCERTAIN"/>
</colorPalette></rasterrenderer></pipe></qgis>'''


def export_patch(zone: str, r: int, c: int, x1: np.ndarray, x2: np.ndarray, 
                 lab: np.ndarray, ae: np.ndarray, base_tf, crs, out_dir: Path,
                 blind: bool = False, patch: int = 128):
    """
    Export an 8-band false/true colour raster stack, AlphaEarth change map, 
    and auto-label raster with .qml styling for rapid QGIS verification.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = win_transform(Window(c, r, patch, patch), base_tf)
    stem = f'{zone}_{r}_{c}'

    # 8-band image stack: T1 RGB + NIR, T2 RGB + NIR
    # Channel index in x1: B2=0, B3=1, B4=2, B8=6 (in 11-band S2 order)
    img = np.stack([
        x1[2], x1[1], x1[0], x1[6],
        x2[2], x2[1], x2[0], x2[6]
    ])[:, r:r+patch, c:c+patch]

    with rasterio.open(
        out_dir / f'{stem}.tif', 'w', driver='GTiff',
        height=patch, width=patch, count=8, dtype='float32',
        crs=crs, transform=tf
    ) as d:
        d.write(img.astype(np.float32))
        d.descriptions = ('T1_R', 'T1_G', 'T1_B', 'T1_NIR',
                          'T2_R', 'T2_G', 'T2_B', 'T2_NIR')

    # BLIND patches get NO auto-label AND NO AlphaEarth change map for anchoring bias control (Part 22.5)
    if blind:
        return

    with rasterio.open(
        out_dir / f'{stem}_ae.tif', 'w', driver='GTiff',
        height=patch, width=patch, count=1, dtype='float32',
        crs=crs, transform=tf
    ) as d:
        d.write(ae[None, r:r+patch, c:c+patch].astype(np.float32))

    with rasterio.open(
        out_dir / f'{stem}_autolabel.tif', 'w', driver='GTiff',
        height=patch, width=patch, count=1, dtype='uint8',
        crs=crs, transform=tf, nodata=None
    ) as d:
        d.write(lab[None, r:r+patch, c:c+patch])

    (out_dir / f'{stem}_autolabel.qml').write_text(QML, encoding='utf-8')
