# Geo-Nexus — Backend API & Inference Server

Flask-based asynchronous API server providing dual-engine satellite change detection:
1. **Few-Shot Domain-Adapted Deep Learning Model** (`MH-DAPT-CD v3.2`, Siamese ResNet encoder with Attention).
2. **Enhanced Multi-Spectral S-CVA Engine** (NDVI, NDWI, NDBI, BSI differencing with Dynamic World validation).

## Architecture

- **`app.py`**: Asynchronous Flask REST API server with non-blocking threaded job queues.
- **`inference.py`**: Model loading, int16 multi-band tensor normalization, patch inference (128x128), and semantic RGB change mask generation.
- **`spectral_cd.py`**: Change Vector Analysis (S-CVA) computing surface dynamics without requiring ML weights.
- **`sentinel_stac_service.py`**: High-speed, auth-free Sentinel-2 L2A STAC client (Element84 / AWS Open Data).
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
| `GET` | `/api/detect/status/<job_id>` | Poll job progress and retrieve result |
| `GET` | `/api/predictions` | List all saved predictions from SQLite database |
| `POST` | `/api/predictions` | Save/update a prediction record |
| `DELETE` | `/api/predictions/<id>` | Delete a saved prediction record |

## Running Tests

Integration and validation scripts are located in `tests/`:

```bash
python tests/test_verify_backend_run.py
python tests/test_verify_spectral.py
```

