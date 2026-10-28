# Geo-Nexus Repository Audit and Implementation Plan

## Verified architecture

- The React 19/Vite interface in `Frontend/src` collects a bounding box and date range, submits asynchronous jobs to Flask, polls job status, renders imagery/masks/metrics, and combines server SQLite history with a small local cache.
- `Backend/app.py` exposes separate AI and spectral job endpoints. Both currently acquire imagery through GEE first when configured and otherwise fall back to the public Sentinel-2 STAC service.
- `Backend/inference.py` loads the v3.2 k=30 checkpoint, applies train-only channel normalization, runs 128-pixel tiled inference, thresholds sigmoid logits, and generates probability, mask, overlay, statistics, and heuristic surface categories.
- `Backend/spectral_cd.py` independently computes a multi-index change-vector result. It does not use the neural checkpoint.
- The model manifest requires 17 channels: Sentinel-2 B2–B12, NDVI, NDBI, Sentinel-1 VV, VH, VH_VV, and Q. Training statistics include real SAR distributions; the research notes specify Sentinel-1 preprocessing and spatial quality data.
- The production checkpoint is wrapped with a top-level `model` state dictionary and metadata. The original loader did not unwrap it, then used `strict=False`, allowing the network to run with randomly initialized parameters.
- The STAC service supplies the 11 Sentinel-2 bands, but currently fills VV/VH/VH_VV and Q with spatial constants. GEE is the only current live provider that constructs real Sentinel-1 features.
- GEE currently casts all 17 channels to int16, quantizing NDVI/NDBI and fractional Q values, and passes raw `n_clear`; the training contract preserves float derived channels and defines Q as `clip(n_clear / 8, 0, 1)`.
- SQLite uses a single `predictions` table with JSON fields for bbox, statistics, dynamics, and full result. No migration or engine column currently exists.

## Verified issues to address

1. STAC placeholder SAR and quality features can reach the gated AI model despite not matching its trained input contract; GEE also quantizes derived/quality channels and does not normalize `n_clear` as training specifies.
2. The production checkpoint wrapper was not recognized and permissive loading silently ignored every weight key.
3. API result/history/cache records do not carry a consistent engine identity. Cache matching uses only dates and approximate bbox, allowing one engine's result to satisfy a request for the other.
4. Spectral and AI outputs share UI labels. In particular, probability is shown as confidence and the threshold is presented as a model threshold even for spectral results.
5. API request validation is incomplete, and public error responses can include raw exception text. GEE-to-STAC data fallback is also not distinguished from an analysis-engine fallback in result metadata.
6. Changed-area calculations hard-code 10 m pixels although returned dimensions are aspect-adjusted/resampled and do not encode a verified 10 m output grid.
7. Existing polling timers are cleared on common paths, but terminal job errors are removed immediately and failures lack a stable structured error category for client recovery.

## Implementation sequence

1. Add shared validation for bbox, ISO dates, date order, cloud percentage, temporal window, and threshold; use structured internal error types and safe client messages.
3. Keep the two job routes and queue behavior. Unwrap the production checkpoint's `model` state dictionary and load it strictly. Restore GEE-derived features to the documented training scales without changing model normalization, and reject Sentinel-2-only STAC input rather than infer with synthetic SAR features. Add engine, algorithm/model version, threshold source, provider, acquisition, input-channel, and resolution metadata. Spectral jobs can use either provider.
4. Make the SQLite change backward-compatible: add engine/algorithm/provider columns with defaults that classify legacy rows as `unknown`, preserve all results, and make matching engine-aware. Keep server and browser cache workflows.
5. Update the interface so AI is the primary action, spectral is an explicit advanced diagnostic, failures offer retry/spectral recovery, source badges are always visible, and result/history metrics reflect only values the selected engine actually supplies.
6. Improve provider/error documentation and setup guidance without changing the trained architecture, checkpoint, normalization, or research protocol.
7. Extend and run backend checks and frontend lint/build. Report any checks blocked by missing dependencies, credentials, or external data access.

## Scope safeguards

- Do not modify model weights, architecture, train-only normalization, or the research/evaluation protocol.
- Do not label heuristic probability as calibrated confidence or research evaluation scores as per-run accuracy.
- Do not silently switch from AI to spectral analysis or reuse a cached result from another engine.
- Do not push or publish repository changes.
