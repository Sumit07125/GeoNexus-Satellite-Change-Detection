# Geo-Nexus v3.2 — P4b V6 Model Artifacts

Kaggle Dataset:
`sumit07125/geonexus-p4b-v6-model-artifacts`

Title:
`Geo-Nexus P4b V6 Model Artifacts`

## Included P4b checkpoints

- `mh_fewshot_k2.pth`
- `mh_fewshot_k3.pth`
- `mh_fewshot_k5.pth`
- `mh_fewshot_k10.pth`
- `mh_fewshot_k20.pth`
- `mh_fewshot_k30.pth`

Best-k alias:

- `mh_fewshot_best_k30.pth`

## Included model/runtime artifacts

- `p3_geonexus_model.py`
- `geonexus_v3_2_p3_oscd_model.pth`
- `norm_stats_trainonly.json`
- `geonexus_model_registry.json`
- P4b results/configuration/diagnostics
- SHA256 manifest
- dataset metadata

## Main model for GeoWatch-Nexus

The production model is:

`mh_fewshot_best_k30.pth`

Input contract:

- 2 timestamps
- 17 channels per timestamp
- 128x128 patches

The application must use the stored train-only normalization file and the
provided P3 model source. Do not change the channel order or normalization.

## Protocol

MH-ADAPT = 30
MH-VAL = 30
MH-TEST = 110

The best k is selected from MH-VAL only.

Label 255 is ignore.
