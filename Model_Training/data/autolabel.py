"""
data/autolabel.py — Multi-Modal Evidence Fusion for Geo-Nexus v3.2

Fuses Open Buildings / Hansen Global Forest Change / Dynamic World / 
spectral indices (NDVI, NDBI, MNDWI) / JRC Surface Water into a 7-class 
training label with an explicit 255 = UNCERTAIN class, arbitrated by AlphaEarth.

Formulas and fusion table are specified in FINAL_ARCH_3.md Part 23 & Part 27.3.
"""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening

IGNORE = 255
CLASS_NAMES = [
    'no_change',      # 0
    'water_gain',     # 1
    'water_loss',     # 2
    'construction',   # 3
    'veg_loss',       # 4
    'veg_gain',       # 5
    'other'           # 6
]

# Evidence band order written by 03_export_autolabels.js (10 bands Int16)
EV = {
    'ob_p2020':    0,  # Open Buildings confidence (2020), scaled 0-10000
    'ob_p2023':    1,  # Open Buildings confidence (2023), scaled 0-10000
    'ob_h2023':    2,  # Open Buildings height (2023), scaled cm (0-10000)
    'hansen_ly':   3,  # Hansen lossyear (0=none, 20-23=loss 2020-2023)
    'hansen_tc00': 4,  # Hansen treecover2000 (0-100)
    'dw_built_t1': 5,  # Dynamic World built probability T1 (0-10000)
    'dw_built_t2': 6,  # Dynamic World built probability T2 (0-10000)
    'dw_trees_t1': 7,  # Dynamic World trees probability T1 (0-10000)
    'dw_trees_t2': 8,  # Dynamic World trees probability T2 (0-10000)
    'jrc_occ':     9   # JRC surface water occurrence (0-100)
}

CFG = dict(
    # Construction thresholds
    ob_hi=0.60, ob_lo=0.20, ob_jump=0.45,
    tau_ndbi=0.15, ndvi_drop_min=0.05, dw_margin=0.35,
    # Vegetation thresholds
    tau_ndvi_loss=0.20, ndvi_t1_min=0.40,
    tau_ndvi_gain=0.20, ndvi_t1_max=0.30, ndvi_t2_min=0.45,
    # Water thresholds
    mndwi_thresh=0.0, jrc_occ_min=5, jrc_buffer_px=20,
    # AlphaEarth arbiter -- PERCENTILES, not fixed values
    ae_pct_hi=92, ae_pct_lo=55,
    # Morphology cleanup
    min_blob_px=4,
)


def load_evidence(zone: str, root, load_merged_fn=None):
    """
    Returns a dict of float arrays on the 10 m grid, unscaled to [0, 1] or true units.
    """
    if load_merged_fn is None:
        from data.preprocess import load_merged
        load_merged_fn = load_merged

    e = load_merged_fn(f'{zone}_autolabel_evidence', root).astype(np.float32)
    return {
        'ob_p2020':    e[EV['ob_p2020']]    / 10000.0,
        'ob_p2023':    e[EV['ob_p2023']]    / 10000.0,
        'ob_h2023':    e[EV['ob_h2023']]    / 100.0,
        'hansen_ly':   e[EV['hansen_ly']],
        'hansen_tc00': e[EV['hansen_tc00']],
        'dw_built_t1': e[EV['dw_built_t1']] / 10000.0,
        'dw_built_t2': e[EV['dw_built_t2']] / 10000.0,
        'dw_trees_t1': e[EV['dw_trees_t1']] / 10000.0,
        'dw_trees_t2': e[EV['dw_trees_t2']] / 10000.0,
        'jrc_occ':     e[EV['jrc_occ']],
    }


def build_evidence_masks(ev, ndvi1, ndvi2, ndbi1, ndbi2, mndwi1, mndwi2, c=CFG):
    """
    Build binary evidence candidate masks using multi-sensor physics and product logic.
    """
    d_ndvi = ndvi2 - ndvi1
    d_ndbi = ndbi2 - ndbi1

    # ---- E1: CONSTRUCTION -------------------------------------------------
    # Triple condition defends against Open Buildings' uncalibrated cross-year
    # confidence drift.
    e_ob = ((ev['ob_p2023'] >= c['ob_hi']) &
            (ev['ob_p2020'] <= c['ob_lo']) &
            ((ev['ob_p2023'] - ev['ob_p2020']) >= c['ob_jump']))

    # NDBI rises both when vegetation becomes concrete AND when a field dries.
    # Requiring NDVI to fall simultaneously removes most drying false positives.
    e_ndbi = (d_ndbi >= c['tau_ndbi']) & (d_ndvi <= -c['ndvi_drop_min'])
    e_dw_b = (ev['dw_built_t2'] - ev['dw_built_t1']) >= c['dw_margin']
    construction = e_ob | (e_ndbi & e_dw_b)

    # ---- E2: VEGETATION LOSS ----------------------------------------------
    # Hansen is 30.92 m resampled to 10 m. Erode by 1 so a single misregistered
    # Hansen pixel cannot stamp a 3x3 block of labels into the data.
    e_hansen = (ev['hansen_ly'] >= 20) & (ev['hansen_ly'] <= 23)
    e_hansen = binary_erosion(e_hansen, np.ones((3, 3)))
    
    # NDVI can only meaningfully FALL from a vegetated start.
    e_ndvi_l = (d_ndvi <= -c['tau_ndvi_loss']) & (ndvi1 >= c['ndvi_t1_min'])
    e_dw_t   = (ev['dw_trees_t1'] - ev['dw_trees_t2']) >= c['dw_margin']
    veg_loss = e_hansen | (e_ndvi_l & e_dw_t)

    # ---- E3: VEGETATION GAIN (spectral only -- conservative) --------------
    veg_gain = ((d_ndvi >= c['tau_ndvi_gain']) &
                (ndvi1 <= c['ndvi_t1_max']) & 
                (ndvi2 >= c['ndvi_t2_min']))

    # ---- E4: WATER --------------------------------------------------------
    w1 = mndwi1 >= c['mndwi_thresh']
    w2 = mndwi2 >= c['mndwi_thresh']
    # JRC prior removes the biggest MNDWI false positive: terrain and cloud
    # shadow on hillslopes (the Zone B failure mode).
    prior = binary_dilation(ev['jrc_occ'] >= c['jrc_occ_min'],
                            np.ones((c['jrc_buffer_px'] * 2 + 1,) * 2))
    water_gain = (~w1) & w2 & prior
    water_loss = w1 & (~w2) & prior

    return {
        'water_gain': water_gain,
        'water_loss': water_loss,
        'construction': construction,
        'veg_loss': veg_loss,
        'veg_gain': veg_gain
    }


from scipy.ndimage import label as nd_label


def remove_small_blobs(mask: np.ndarray, min_size: int = 4) -> np.ndarray:
    """Remove connected components with area smaller than min_size pixels (MMU filter)."""
    if min_size <= 1:
        return mask
    labeled, num_features = nd_label(mask)
    if num_features == 0:
        return mask
    counts = np.bincount(labeled.flat)
    remove = counts < min_size
    remove[0] = False  # Keep background
    out = mask.copy()
    out[remove[labeled]] = False
    return out


def fuse(masks, ae_score, c=CFG):
    """
    Apply the Part 23.2 fusion table arbitrated by AlphaEarth embeddings.
    ae_score: AlphaEarth 1 - cosine, in [0, 2].
    Returns (label uint8 with 255 = uncertain, confidence float32).
    """
    H, W = ae_score.shape
    label = np.zeros((H, W), np.uint8)
    conf  = np.zeros((H, W), np.float32)

    # Priority order matching Step 6: water -> construction -> veg
    claimed = np.zeros((H, W), bool)
    for name, cid in [('water_gain', 1), ('water_loss', 2), ('construction', 3),
                      ('veg_loss', 4), ('veg_gain', 5)]:
        m = masks[name] & ~claimed
        m = binary_opening(m, np.ones((3, 3)))  # drop 1-px isolated speckle
        if c.get('min_blob_px', 0) > 1:
            m = remove_small_blobs(m, c['min_blob_px'])  # MMU filter (<4 px)
        label[m] = cid
        claimed |= m

    # Percentile thresholds per zone -- NOT fixed absolute values
    tau_hi = np.percentile(ae_score, c['ae_pct_hi'])
    tau_lo = np.percentile(ae_score, c['ae_pct_lo'])
    ae_chg  = ae_score >= tau_hi
    ae_stab = ae_score <= tau_lo

    # Case 1: Product claimed + AlphaEarth agrees change -> High Confidence
    conf[claimed & ae_chg] = 1.00

    # Case 2: Product claimed + AlphaEarth unsure -> Moderate Confidence
    conf[claimed & ~ae_chg & ~ae_stab] = 0.60

    # Case 3: CONFLICT: Product claims change but AlphaEarth detects high stability -> DROP
    label[claimed & ae_stab] = IGNORE

    # Case 4: Unclaimed + AlphaEarth detects change -> 'other' (Class 6)
    unclaimed = ~claimed
    label[unclaimed & ae_chg] = 6
    conf[unclaimed & ae_chg]  = 0.50

    # Case 5: Unclaimed + AlphaEarth detects stability -> 'no change' (Class 0)
    label[unclaimed & ae_stab] = 0
    conf[unclaimed & ae_stab]  = 1.00

    # Case 6: UNRESOLVED: Nothing claims it and AlphaEarth is ambiguous -> UNCERTAIN (255)
    label[unclaimed & ~ae_chg & ~ae_stab] = IGNORE

    return label, conf


def report(label, zone):
    """Class balance and uncertain audit. Print for each zone."""
    tot = label.size
    out = {'zone': zone, 'tau_note': 'percentile-based'}
    for i, n in enumerate(CLASS_NAMES):
        out[n] = round(100.0 * float((label == i).sum()) / tot, 3)
    out['uncertain'] = round(100.0 * float((label == IGNORE).sum()) / tot, 3)
    
    summary_str = f"[{zone}] " + "  ".join(
        f"{k}={v}%" for k, v in out.items() if k not in ('zone', 'tau_note')
    )
    print(summary_str)
    
    u = out['uncertain']
    if u > 40:
        print(f'  WARNING: [{zone}] uncertain ({u:.1f}%) > 40% -- raise ae_pct_lo')
    if u < 8:
        print(f'  WARNING: [{zone}] uncertain ({u:.1f}%) < 8% -- thresholds too loose, noise leaking in')
        
    return out
