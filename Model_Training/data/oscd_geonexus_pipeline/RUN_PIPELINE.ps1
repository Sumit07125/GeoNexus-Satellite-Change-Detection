# Run from C:\Users\sumit_mali\Desktop\Geo_Watch
$ErrorActionPreference = "Stop"

python -m pip install -U numpy rasterio tqdm kaggle
python "data\oscd_geonexus_pipeline\self_test.py"
python "data\oscd_geonexus_pipeline\run_pipeline.py"

Write-Host "Local Geo-Nexus OSCD pipeline finished." -ForegroundColor Green
Write-Host "Review data\oscd\prepared\oscd_preprocess_manifest.json before Kaggle upload."
