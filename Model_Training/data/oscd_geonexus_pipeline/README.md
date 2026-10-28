# Geo-Nexus / MH-DAPT-CD v3.2 — OSCD release pipeline

Copy this directory into:
`C:\Users\sumit_mali\Desktop\Geo_Watch\data\oscd_geonexus_pipeline`

## Project contract
- OSCD: 24 bi-temporal Sentinel-2 city pairs with 13 bands per date.
- Frozen Geo-Nexus supervised split: 11 train / 3 validation / 10 test.
- Train patch: 128x128 with stride 64.
- Validation/test patch: 128x128 with stride 128.
- 17 channels per timestamp: 11 raw optical + NDVI + NDBI + 3 SAR placeholders + q=1.
- OSCD supervised transfer is optical-only; placeholder SAR channels are not measured SAR.
- Stored tensors: int16 after x10000; labels: uint8 0/1.
- Train-only change-patch oversampling metadata: change fraction >1% is replicated 3x.

## Why your latest 33758 error happened
The previous version treated any x10000 result above signed int16 as fatal. Your verified run reached a raw scaled maximum of 33758. The project output contract is nevertheless int16 x10000, and the earlier project implementation explicitly used final `np.clip(..., -32768, 32767).astype(np.int16)` storage. This rebuilt package therefore clips **only at the final serialization boundary** and records the exact saturation count/fraction in the split manifest. No pre-feature clipping is performed and non-finite values still fail.

Run `audit_oscd_values.py` before the write stage to obtain the raw-range report.

## Safety
- Raw OSCD is never modified.
- Preparation writes into a staging directory and promotes only after validation.
- Partial output is deleted after an exception.
- SHA256 hashes are generated for the final release.
- Kaggle staging copies only approved prepared files; raw `Images/`, `Train Labels/`, and `Test Labels/` are not uploaded.

## Output
`data/oscd/prepared/` contains the three arrays, three label arrays, split metadata, sampler metadata, preprocessing manifest and SHA256 manifest.
