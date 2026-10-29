# Local 30-pair operational diagnostic

This optional CLI is separate from the public API and normal prediction history. It acquires data through Google Earth Engine only, validates the trained 17-channel contract, runs the production Geo-Nexus checkpoint, and optionally runs S-CVA independently on the same acquisition. It records failed acquisitions and invalid inputs; it does not replace them or report research accuracy.

## Private configuration and access

Set a long random secret in `Backend/.env` as `GEONEXUS_DIAGNOSTIC_ADMIN_TOKEN`. Put only its SHA-256 digest in a private JSON config ignored by Git. Generate the digest locally without putting the secret in shell history, for example in a local Python prompt:

```python
import getpass, hashlib
hashlib.sha256(getpass.getpass("Diagnostic admin token: ").encode()).hexdigest()
```

Example `Backend/diagnostics/private_config.json`:

```json
{
  "admin_token_sha256": "<64-character digest>",
  "aoi_bbox": {"lon_min": 73.5, "lat_min": 17.5, "lon_max": 74.5, "lat_max": 19.0},
  "region_size_km": 2.0,
  "t1_dates": ["2021-01-01", "2022-01-01"],
  "t2_dates": ["2024-01-01", "2025-01-01"],
  "window_days": 15,
  "max_cloud": 30
}
```

The configured AOI is a geographic bounding box supplied by the operator; it is not a legal or exact Maharashtra state boundary. For exact candidate areas and date pairs, provide a `candidates` array with `candidate_id`, `bbox`, `date_t1`, and `date_t2`; every bbox must stay within `aoi_bbox`.

## Run

From `Backend/`, with Earth Engine credentials and `GEE_PROJECT_ID` configured:

```powershell
python diagnostics/region_pair_diagnostic.py --config diagnostics/private_config.json --output diagnostics/output/run-001 --count 30 --seed 314159 --run-spectral
```

The JSON report records all generated inputs, provider/acquisition metadata, validation/acquisition failures, timing, model/preprocessing version, threshold provenance, and separate AI/S-CVA output paths. The same seed and config reproduce generated regions and date pairs. A failed acquisition remains a failed row; the CLI does not replace it to force 30 successes. Result files are operational diagnostics only and have no ground-truth accuracy interpretation.

## Isolated Sentinel-1 orbit metadata query

To identify which read-only Earth Engine metadata operation fails, run from `Backend/` using the same Python environment and `.env` as Flask:

```powershell
python diagnostics/test_s1_orbit_query.py --t1 2021-03-15 --t2 2024-03-15 --window-days 15 --max-cloud 30 --bbox 72.8432 19.1113 72.8527 19.1203
```

This probe initializes EE, makes one trivial request, then runs base collection count, descending IW VV/VH filtered count, orbit histogram, and response parsing independently for T1 and T2. It prints inclusive and exclusive date bounds, computes the common orbit set only when both queries succeed, and never downloads imagery or runs inference. If histogram evaluation fails, it tries `aggregate_array` as a diagnostic probe only; production still uses the histogram query. Project credentials are read locally and never printed.
