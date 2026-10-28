# Research / source notes used for the rebuild

## OSCD source
Official OSCD project page:
https://rcdaudt.github.io/oscd/

The source describes 24 registered Sentinel-2 image pairs, 13 bands per date, mixed spatial resolutions, and openly available pixel-level change labels.

## Dataset layout
TorchGeo's current OSCD documentation/source:
https://docs.torchgeo.org/en/stable/api/datasets/oscd.html

The loader uses the official OSCD directory naming and 13 Sentinel-2 bands. The Geo-Nexus pipeline keeps the project's frozen `imgs_*_rect` preference and falls back to native `imgs_*` with explicit alignment to the B4 reference grid.

## Geo-Nexus project architecture
Primary local source of truth:
`FINAL_ARCH_3(8).md`

Relevant frozen decisions:
- OSCD is the supervised optical-only transfer stage.
- 11 train / 3 validation / 10 test project split.
- 128x128 patches.
- Train stride 64; evaluation stride 128.
- 17-channel tensor contract: 11 raw optical + NDVI + NDBI + 3 SAR placeholders + q.
- int16 storage after x10000.
- train-only 3x change-patch oversampling.

## Why final int16 saturation is audited
The previous run failed at final serialization because the scaled range reached 33758. The architecture requires int16 x10000 storage, and earlier project implementation materials used final `np.clip(..., -32768, 32767)` serialization. This rebuild retains that storage contract while making saturation explicit in `oscd_preprocess_manifest.json` and `oscd_value_audit.json`.

No clipping occurs before NDVI/NDBI feature construction.

## Kaggle CLI
Current Kaggle CLI documentation:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md

The current CLI uses:
- `kaggle datasets create -p <folder> --dir-mode zip`
- `kaggle datasets version -p <folder> -m "..." --dir-mode zip`
- required `dataset-metadata.json` in the upload folder.

The pipeline generates that metadata automatically and stages only the validated prepared release.
