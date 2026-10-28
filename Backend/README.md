# Geo-Nexus — Backend API & Inference Server

Flask-based asynchronous API server providing dual-engine satellite change detection:
1. **Primary — Geo-Nexus AI Model** (`MH-DAPT-CD v3.2`, Siamese ResNet encoder, trained on Sentinel-1 + Sentinel-2 inputs).
2. **Explicit alternative — Spectral Index Analysis (S-CVA)** (optical NDVI, NDWI, NDBI, BSI and change-vector analysis; it does not run the neural model).

The AI endpoint requires genuine Sentinel-1 VV/VH/VH_VV and the trained Q quality feature. Google Earth Engine constructs these inputs on the documented 10 m UTM grid. The runtime accepts areas at least 128×128 pixels (about 1.3 km per side) and caps a single download at 400,000 pixels. The public STAC provider supplies Sentinel-2 only and is used by the explicit spectral endpoint. The AI endpoint never falls back to STAC: it returns a structured readiness, authentication, project-access, acquisition, or input-validation error when GEE cannot provide the trained multimodal inputs. STAC's constant SAR/Q placeholders are never passed to the model.

For the production multi-date composite path, training defines `Q = clip(n_clear / 8, 0, 1)`, where `n_clear` counts B4 observations after linked Cloud Score+ `cs_cdf >= 0.60` masking. The training heterogeneity gate applies to raw `n_clear` (`std(n_clear) > 0.5`) before Q clipping, so a constant Q raster can be valid when the raw count varied above the clipping ceiling. GEE records raw count statistics; runtime checks source provenance, finite values, encoded range and all-zero/low-coverage nodata before applying the criterion to raw counts. Missing raw diagnostics or failed heterogeneity returns provider/date/channel details; no variation is fabricated. The single-date monsoon benchmark uses `cs_cdf` directly, but the runtime does not currently acquire that product.

## Architecture

- **`app.py`**: Asynchronous Flask REST API server with non-blocking threaded job queues.
- **`inference.py`**: Model loading, int16 multi-band tensor normalization, patch inference (128x128), and semantic RGB change mask generation.
- **`spectral_cd.py`**: Change Vector Analysis (S-CVA) computing surface dynamics without requiring ML weights.
- **`sentinel_stac_service.py`**: Sentinel-2 L2A STAC client (Element84 / AWS Open Data); does not provide real SAR inputs.
- **`gee_service.py`**: Google Earth Engine client for multi-spectral scene extraction and cloud filtering.
- **`storage_manager.py`**: Thread-safe SQLite persistence for predicted outcomes.
- **`Model/`**: Model architecture (`p3_geonexus_model.py`), manifest, normalization stats, and weights.
- **`tests/`**: Integration and validation test suites.
- **`diagnostics/region_pair_diagnostic.py`**: Optional local admin-only 30-candidate operational diagnostic using GEE, the production checkpoint, reproducible candidates, separate S-CVA output, and honest failure records. It does not generate research accuracy metrics.

## Setup Instructions

### 1. Python Environment

Python 3.10 or 3.11 is recommended.

```bash
cd Backend
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:
- `GEE_PROJECT_ID`: Required for AI. Replace `your-gcp-project-id` with a real Google Cloud project that has the Earth Engine API enabled. Missing, blank, and placeholder values are rejected; spectral analysis remains usable.
- Authentication: run `earthengine authenticate` for local user credentials, or set `GEE_SERVICE_ACCOUNT_KEY` to a valid service-account JSON file whose account has Earth Engine access to the configured project. Secret contents are never returned by health checks.
- `FRONTEND_URL`: `http://localhost:5173`
- `FLASK_PORT`: `5000`
- `CORS_ORIGINS`: comma-separated browser origins; defaults to `FRONTEND_URL`.

### 3. Running the Server

```bash
python app.py
```

Server starts at: **http://localhost:5000**
Health check endpoint: `GET /api/health`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | API status, checkpoint file presence, and sanitized AI/GEE readiness (remote access is verified during a run) |
| `POST` | `/api/detect/start` | Submit an AI change detection job |
| `POST` | `/api/detect/spectral/start` | Submit an enhanced multi-spectral change detection job |
| `GET` | `/api/detect/status/<job_id>` | Poll job progress and retrieve result or structured error |
| `GET` | `/api/predictions` | List all saved predictions from SQLite database |
| `POST` | `/api/predictions` | Save/update a prediction record |
| `DELETE` | `/api/predictions/<id>` | Delete a saved prediction record |

Both start endpoints validate bbox coordinates, ordered ISO dates, cloud percentage, temporal search window, and engine-specific thresholds. The AI start endpoint checks project and credential configuration before queueing; authentication, project access, imagery availability, input validity, and strict model loading are checked during the job. GEE-to-STAC fallback is not used for AI. Successful responses include an engine ID, display label, algorithm/model version, threshold provenance, and imagery provider/acquisition metadata. Legacy database rows are retained with source `unknown`; cache matching is engine-specific.

### Sentinel-1 orbit and area policy

For the exact Pune, Satara, and Vidarbha research benchmark extents, the documented frozen relative orbits (136, 136, and 165) remain in force. Other AOIs query `COPERNICUS/S1_GRD` independently in each requested date window, retain descending IW scenes with VV and VH, and choose a single shared relative orbit. The selection is deterministic: maximize the lower of the T1/T2 scene counts, then total count, then prefer the lower orbit ID. If there is no common orbit, the AI job returns `gee_no_common_s1_orbit` with per-date orbit counts; it does not alter the requested window. Users may explicitly widen the temporal search window and retry.

AI requests smaller than 1.28 km in either bbox dimension are rejected before a satellite job is queued. The downloaded T1/T2 grids are checked again for matching shapes and at least 128×128 pixels. Acquisition failures preserve a stage code and bounded diagnostics, including requested dates/windows, scene and orbit counts, Cloud Score+ matches, selected scene metadata, CRS/grid information when available, and a sanitized technical reason. The public job response omits stack traces and credentials. A successful local authentication check establishes account access only; it does not prove that the selected date windows contain suitable Sentinel-1/Sentinel-2 imagery. The deterministic test suite uses mocks and does not establish live imagery availability.

## Running Tests

The deterministic API, model-input, provider separation, and storage-migration checks are located in `tests/`:

```bash
python -m unittest discover -s tests -p test_reliability_contracts.py -v
```

The older `test_verify_*.py` and overlay scripts are scratch-data diagnostics. They require `scratch_t1.png`, `scratch_t2.png`, and/or `scratch_tensors.npz` in `Backend/`; those fixtures are not committed. `test_gee_full_pipeline.py` queries the live public STAC service and needs network access.

The deterministic suite tests Q range/nodata/heterogeneity cases and provider separation with local arrays and mocked services. It does not validate live GEE credentials or current imagery availability. See [diagnostics/README.md](diagnostics/README.md) for the optional operational diagnostic and its access/configuration requirements.

