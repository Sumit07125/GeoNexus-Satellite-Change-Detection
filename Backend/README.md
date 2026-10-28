# Geo-Nexus — Backend API & Inference Server

Flask-based asynchronous API server providing dual-engine satellite change detection:
1. **Primary — Geo-Nexus AI Model** (`MH-DAPT-CD v3.2`, Siamese ResNet encoder, trained on Sentinel-1 + Sentinel-2 inputs).
2. **Explicit alternative — Spectral Index Analysis (S-CVA)** (optical NDVI, NDWI, NDBI, BSI and change-vector analysis; it does not run the neural model).

The AI endpoint requires real Sentinel-1 VV/VH/VH_VV and spatial quality channels. Google Earth Engine constructs these inputs on the documented 10 m UTM grid. The runtime accepts areas at least 128×128 pixels (about 1.3 km per side) and caps a single download at 400,000 pixels. The public STAC provider currently supplies Sentinel-2 only, so AI inference stops with a structured `missing_sar_channels` error after STAC fallback; it never runs the model with synthetic SAR values. STAC remains available to the spectral endpoint.

## Architecture

- **`app.py`**: Asynchronous Flask REST API server with non-blocking threaded job queues.
- **`inference.py`**: Model loading, int16 multi-band tensor normalization, patch inference (128x128), and semantic RGB change mask generation.
- **`spectral_cd.py`**: Change Vector Analysis (S-CVA) computing surface dynamics without requiring ML weights.
- **`sentinel_stac_service.py`**: Sentinel-2 L2A STAC client (Element84 / AWS Open Data); does not provide real SAR inputs.
- **`gee_service.py`**: Google Earth Engine client for multi-spectral scene extraction and cloud filtering.
- **`storage_manager.py`**: Thread-safe SQLite persistence for predicted outcomes.
- **`Model/`**: Model architecture (`p3_geonexus_model.py`), manifest, normalization stats, and weights.
- **`tests/`**: Integration and validation test suites.

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
- `GEE_PROJECT_ID`: (Optional) Your Google Cloud project with Earth Engine API enabled. If not set, the server automatically defaults to the free Sentinel-2 STAC open data fetcher.
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
| `GET` | `/api/health` | Service health status and environment check |
| `POST` | `/api/detect/start` | Submit an AI change detection job |
| `POST` | `/api/detect/spectral/start` | Submit an enhanced multi-spectral change detection job |
| `GET` | `/api/detect/status/<job_id>` | Poll job progress and retrieve result or structured error |
| `GET` | `/api/predictions` | List all saved predictions from SQLite database |
| `POST` | `/api/predictions` | Save/update a prediction record |
| `DELETE` | `/api/predictions/<id>` | Delete a saved prediction record |

Both start endpoints validate bbox coordinates, ordered ISO dates, cloud percentage, temporal search window, and engine-specific thresholds. Successful responses include an engine ID, display label, algorithm/model version, threshold provenance, and imagery provider/acquisition metadata. Legacy database rows are retained with source `unknown`; cache matching is engine-specific.

## Running Tests

The deterministic API, model-input, provider-fallback, and storage-migration checks are located in `tests/`:

```bash
python -m unittest discover -s tests -p test_reliability_contracts.py -v
```

The older `test_verify_*.py` and overlay scripts are scratch-data diagnostics. They require `scratch_t1.png`, `scratch_t2.png`, and/or `scratch_tensors.npz` in `Backend/`; those fixtures are not committed. `test_gee_full_pipeline.py` queries the live public STAC service and needs network access.

