# Geo-Nexus — Frontend Application

Modern React + Vite web interface for the Geo-Nexus Satellite Change Detection system.

## Features

- **Interactive Satellite Map (Leaflet)**: Custom bounding box selection, coordinate input, preset regional hotspots, and live scale/area calculations.
- **Dedicated Predictions & Outcome Studio**: 3-panel side-by-side inspection of Historical (T1) and Recent (T2) satellite imagery with dynamic change overlays.
- **Dynamic Layer Filtering Engine**: Real-time client-side canvas blending allowing users to toggle and isolate:
  - 🏗️ **Urban & Structural** (Built-up expansion, infrastructure change)
  - 🌿 **Vegetation Dynamics** (Canopy shift, crop bloom, vegetation loss)
  - 🌾 **Soil & Agriculture** (Fallow land, harvest, exposed ground)
  - 💧 **Water Dynamics** (River expansion, reservoirs, flood inundation)
- **Interactive Before / After Wipe Slider**: Smooth wipe divider to inspect changes against natural optical imagery.
- **Saved Predictions Gallery**: Permanent local and server SQLite persistence with instant 0-second reloads.
- **Explicit analysis provenance**: Geo-Nexus AI is the primary action; Spectral Index Analysis (S-CVA) is a separately selected diagnostic method. Each saved result and history card records its source.

## Getting Started

### 1. Prerequisites

- Node.js (v18 or higher recommended)
- npm or pnpm

### 2. Installation

```bash
cd Frontend
npm install
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Ensure `VITE_API_URL` points to your running Flask backend:

```env
VITE_API_URL=http://localhost:5000
```

AI inference requires a configured and authenticated Google Earth Engine project for real Sentinel-1 inputs. The backend reports configuration, authentication, access, acquisition, and input-validation failures without falling back to Sentinel-2-only STAC. The explicitly selected spectral method can use STAC imagery and is always labeled as spectral analysis. Backend health distinguishes API availability from AI provider readiness; remote Earth Engine access and strict checkpoint loading are confirmed during a run.

The status line includes the backend build ID reported by `GET /api/health`, which helps identify a stale backend process. When Sentinel-1 acquisition fails, the error panel distinguishes a query failure from no eligible scenes or no shared orbit and provides an expandable, sanitized acquisition report. A query failure does not imply that no orbit exists. Retry AI repeats the AI request; Run Spectral Analysis calls its separate endpoint.

Center-and-size coordinate controls recalculate the map selection as the size or center is edited. The selected bbox, displayed dimensions, and submitted request use the same bounds. The backend rejects AI areas smaller than 1.28 km in either dimension; the recovery button expands around the current bbox center to a 1.4 km square. The spectral engine remains available for smaller regions.

Frontend validation commands from `Frontend/`:

```powershell
npm ci
npm run test
npm run lint
npm run build
```

### 4. Development Server

```bash
npm run dev
```

The application will start at `http://localhost:5173`.

### 5. Production Build

```bash
npm run build
npm run preview
```
