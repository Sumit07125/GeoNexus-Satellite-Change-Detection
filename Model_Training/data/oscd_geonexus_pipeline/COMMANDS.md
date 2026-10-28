# Geo-Nexus OSCD v3.2 — simple execution

Open PowerShell in:
`C:\Users\sumit_mali\Desktop\Geo_Watch`

## 1. Install once
```powershell
python -m pip install -U numpy rasterio tqdm kaggle
```

## 2. Run all local stages
```powershell
python "data\oscd_geonexus_pipeline\run_pipeline.py"
```

The terminal shows a live overall progress line like:
`LOCAL PIPELINE: |########........| 3/6 [time<ETA]`

## 3. Create the private Kaggle dataset
First configure Kaggle credentials at `%USERPROFILE%\.kaggle\kaggle.json`.

```powershell
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --dry-run
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --create
```

## 4. Update an existing Kaggle dataset later
```powershell
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --version
```

## Manual sequence
```powershell
cd "C:\Users\sumit_mali\Desktop\Geo_Watch"
python "data\oscd_geonexus_pipeline\build_oscd_manifest.py"
python "data\oscd_geonexus_pipeline\verify_oscd.py" "C:\Users\sumit_mali\Desktop\Geo_Watch\data\OSCD"
python "data\oscd_geonexus_pipeline\audit_oscd_values.py" "C:\Users\sumit_mali\Desktop\Geo_Watch\data\OSCD"
python -m py_compile "data\oscd_geonexus_pipeline\prepare_oscd.py"
python "data\oscd_geonexus_pipeline\prepare_oscd.py" "C:\Users\sumit_mali\Desktop\Geo_Watch\data\OSCD" --split-manifest "data\oscd\splits\oscd_splits.json" --out "data\oscd\prepared" --dry-run
python "data\oscd_geonexus_pipeline\prepare_oscd.py" "C:\Users\sumit_mali\Desktop\Geo_Watch\data\OSCD" --split-manifest "data\oscd\splits\oscd_splits.json" --out "data\oscd\prepared" --overwrite
python "data\oscd_geonexus_pipeline\validate_prepared.py" "data\oscd\prepared"
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --dry-run
python "data\oscd_geonexus_pipeline\stage_kaggle_oscd.py" --create
```
