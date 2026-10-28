# Geo-Nexus — Model Training & Research Pipeline

This directory contains the machine learning research, data preparation pipelines, and Jupyter training notebooks for the **MH-DAPT-CD v3.2** (Maharashtra Domain-Adaptive Pretraining for Change Detection) model.

## Overview

- **Architecture**: Siamese ResNet trunk with SplitStem (multi-spectral optical + SAR), cross-temporal difference fusion, and attention decoders.
- **Protocol**:
  - **Phase 1 (Pre-training)**: SSL4EO-S12 foundation weights.
  - **Phase 2 (DAPT)**: Domain-Adaptive Pre-Training on Sentinel-2 optical and Sentinel-1 SAR imagery over Maharashtra geographical zones.
  - **Phase 3 (Transfer)**: Supervised transfer learning on the OSCD (Onera Satellite Change Detection) benchmark.
  - **Phase 4 (Few-Shot Adaptation)**: Few-shot adaptation on verified Maharashtra ground-truth regions ($k \in \{2, 3, 5, 10, 20, 30\}$), with $k=30$ selected as the production model (`mh_fewshot_best_k30.pth`).

## Directory Layout

```
Model_Training/
├── data/
│   ├── gee/                      # GEE export scripts (JavaScript)
│   ├── oscd_geonexus_pipeline/   # OSCD preprocessing and validation suite
│   ├── download_pretrained_weights_v3_2_amended.py
│   └── stage_kaggle.py
├── models/
│   └── stem.py                   # SplitStem multi-spectral input module
├── notebooks/                    # End-to-end research & training notebooks
│   ├── 01_preprocess_*.ipynb
│   ├── geonexus-dapt-final-6664-persistent-artifacts.ipynb
│   ├── geonexus-p3-oscd-supervised-transfer-final-kaggle.ipynb
│   ├── geonexus-p4a-mh-zeroshot-final.ipynb
│   └── geonexus-p4b-mh-fewshot-v6-final.ipynb
├── FINAL_ARCH_3.md               # Complete architecture & protocol specification
└── progress.md                   # Chronological research milestones and audit logs
```

## Dataset & Model Weights

Due to size constraints, raw datasets and large model checkpoints are managed outside Git:

1. **Production Model Weights**:
   - Checkpoint: `mh_fewshot_best_k30.pth` (96.5 MB)
   - Kaggle Artifacts: [`sumit07125/geonexus-p4b-v6-model-artifacts`](https://www.kaggle.com/datasets/sumit07125/geonexus-p4b-v6-model-artifacts)
   - Runtime Location: Placed in `Backend/Model/geonexus_v3_2_k30/`

2. **OSCD Benchmark Dataset**:
   - Download the raw OSCD dataset from IEEE DataPort / IAPR.
   - Run the preprocessing pipeline:
     ```bash
     cd data/oscd_geonexus_pipeline
     python prepare_oscd.py --src ../OSCD --out ../prepared
     ```

3. **Pretrained SSL Weights**:
   - Download script:
     ```bash
     python data/download_pretrained_weights_v3_2_amended.py --out data/weights
     ```

