# Kaggle upload

Dataset slug: `oscd-onera-v1`.

Create private dataset:
```powershell
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --create
```

Update existing dataset:
```powershell
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --version
```

Dry-run:
```powershell
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --dry-run
```

The current Kaggle CLI requires a `dataset-metadata.json` in the upload directory and supports `kaggle datasets create`, `kaggle datasets version`, and `--dir-mode zip`. This package generates the metadata automatically and uploads only validated prepared files.
