# Rebuild changelog

## v3.2-rebuilt

- Removed the hard-coded long archive directory assumption.
- Supports `Images/`, `Train Labels/`, `Test Labels/` directly under `data/OSCD`.
- Supports the long official archive names as a fallback.
- Handles B1/B01/B8/B08/B8A/B08A naming variants.
- Prefers complete `imgs_*_rect` directories.
- Falls back to `imgs_*` and reprojects/resamples onto the B4 reference grid.
- Validates the exact Geo-Nexus 11/3/10 split before heavy processing.
- Converts 1/2 and 0/255 label conventions to binary 0/1.
- Keeps all 17 channels per timestamp in the required order.
- Writes 128x128 patches with the frozen 64/128 stride protocol.
- Uses memory-mapped output files to avoid holding all patches in RAM.
- Uses staging directories and removes partial output after failure.
- Replaces the previous fatal int16 overflow guard with final-storage clipping plus an explicit saturation audit.
- Generates SHA256 hashes for the final prepared release.
- Adds a raw-value audit stage.
- Adds a complete Kaggle staging/upload script.
- Adds a six-stage local pipeline runner with a live overall progress bar.
- Adds a synthetic self-test covering the 33758 overflow condition.
