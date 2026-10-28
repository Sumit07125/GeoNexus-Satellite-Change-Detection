# Geo-Nexus Provider Failure Report

Audited the local `main` checkout at `c610087` before changing implementation. The local `origin/main` ref points to the same commit. A live GitHub fetch was unavailable in this environment; conclusions are based on the checked-out source and local runtime configuration.

## Verified causes

| Finding | Location | Impact | Correction |
|---|---|---|---|
| The AI worker attempted GEE, then fell back to Sentinel-2 STAC after a GEE error or a partially recognized project placeholder. | `Backend/app.py::_run_job` | STAC tensors contain training-mean SAR and constant Q placeholders, so 17 channels did not mean valid AI inputs. The reported `missing_sar_channels` text proves STAC supplied the final inputs; its exact target dates were absent from metadata. | One shared GEE readiness check is used at startup, AI request preflight, EE initialization, and the local diagnostic. AI now calls only GEE and ends with a typed error on configuration, auth, access, acquisition, validation, or inference failure. STAC remains available to spectral analysis. |
| Startup and request-time project checks used different, case-sensitive placeholder lists. GEE setup printed the project and credential path and included raw provider exception text in raised errors. | `Backend/app.py` startup and AI worker; `Backend/gee_service.py::init_ee` | A placeholder could trigger an unnecessary GEE attempt, and logs/errors could expose configuration details. | `Backend/gee_config.py` normalizes placeholder spellings, checks local or service-account credential presence/shape, and emits sanitized readiness categories. EE initialization no longer logs secrets or raw exception text. |
| STAC metadata retained selected scene timestamps but not the requested target dates as `t1_date`/`t2_date`. | `Backend/sentinel_stac_service.py::fetch_imagery` | Failures displayed T1/T2 as `unknown`, confusing requested dates with actual acquisitions. | Metadata now has requested target dates and distinct selected acquisition datetimes. GEE metadata includes the requested windows, per-date S1/S2 counts, scene IDs/timestamps, grid, and quality provenance. |
| Health represented checkpoint-file existence as model readiness. | `Backend/app.py::health`; `Frontend/src/App.jsx` | The UI could imply AI readiness without authenticated GEE access or a strict runtime checkpoint load. | Health now separates server health from sanitized GEE and checkpoint states and never claims AI is ready based only on file presence. The header reports provider/checkpoint verification state. |

## Supplied request reproduction

The supplied frontend dates are already ISO calendar dates accepted by the backend: `2021-03-15` and `2024-03-15`. I reproduced the request shape with the supplied bbox, 15-day search window, and 30% cloud limit.

The current local `.env` contains a non-placeholder project setting, but no service-account path is configured and no local Earth Engine/ADC credentials were found. The current endpoint therefore returned HTTP 503 with `gee_credentials_missing` before queuing acquisition. In the pre-fix code, that configured project would have entered the GEE attempt and then the STAC fallback if GEE initialization failed. The reported error confirms STAC placeholders caused the final AI validation failure. The original per-request logs are not available, so the exact historical GEE exception category cannot be proven; this workspace's present credential state is missing credentials.

## Input and research contract

The production manifest/inference contract remains the existing 17-channel order: B2–B12, NDVI, NDBI, VV, VH, VH_VV, Q. The GEE path remains responsible for genuine Sentinel-1 and the trained Q feature. The current runtime uses harmonized Sentinel-2, Cloud Score+ masking (`cs_cdf >= 0.60`), raw per-pixel clear-observation counts, `Q = clip(n_clear / 8, 0, 1)`, linear-power S1 compositing before conversion to dB, and the existing 10 m UTM output. Validation continues to require finite/ranged Q, valid raw `n_clear` coverage and the training heterogeneity gate. Model weights, architecture, normalization, and evaluation protocol were not changed.

The exact live scene availability, Earth Engine project access, raw Q statistics, and model forward pass for the supplied coordinates remain unverified because this machine has no Earth Engine credentials. A deterministic mocked run with the same request inputs completed the acquisition-validation-inference flow against the production input validator and mocked provider/model boundaries.
