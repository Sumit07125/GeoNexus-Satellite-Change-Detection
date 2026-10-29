import os
import io
import math
import datetime
import requests
import numpy as np
from PIL import Image

import ee

# Ensure .env is loaded even when this module is imported standalone
try:
    from dotenv import load_dotenv as _load_dotenv
    import pathlib
    _env_path = pathlib.Path(__file__).parent / ".env"
    _load_dotenv(dotenv_path=_env_path, override=True)
except Exception:
    pass  # dotenv not installed or .env missing — rely on os.environ

_EE_INITIALIZED = False


class GEEProviderError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class GEEAcquisitionError(GEEProviderError):
    """A sanitized, stage-specific acquisition failure with bounded diagnostics."""


def init_ee():
    """Initialize EE from one explicitly configured authentication method."""
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return
    from gee_config import ensure_gee_configuration, classify_gee_error

    config = ensure_gee_configuration()
    project = os.environ["GEE_PROJECT_ID"].strip()
    try:
        if config.get("auth_method") == "service_account":
            key_path = os.environ["GEE_SERVICE_ACCOUNT_KEY"].strip()
            credentials = ee.ServiceAccountCredentials(email=None, key_file=key_path)
            ee.Initialize(credentials=credentials, project=project)
        else:
            ee.Initialize(project=project)
        _EE_INITIALIZED = True
    except Exception as error:
        if config.get("auth_method") == "service_account":
            message = str(error).lower()
            if any(term in message for term in ("private key", "pem", "malformed", "service account", "serviceaccount")):
                raise GEEProviderError(
                    "gee_service_account_invalid",
                    "The configured Earth Engine service-account key could not be loaded. Check that it is a valid, unexpired service-account JSON key.",
                ) from error
        code, message = classify_gee_error(error)
        raise GEEProviderError(code, message) from error


# ─────────────────────────────────────────────────────────────────────────────
# Band helpers
# ─────────────────────────────────────────────────────────────────────────────

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
# Channel order: 11 optical raw + NDVI + NDBI + VV + VH + VH_VV + Q
CHANNEL_ORDER = [*S2_BANDS, "NDVI", "NDBI", "VV", "VH", "VH_VV", "Q"]

RESEARCH_ORBIT_ZONES = {
    "zone_a_pune": {"bbox": (73.70, 18.30, 74.20, 18.80), "orbit": 136},
    "zone_b_satara": {"bbox": (73.50, 17.50, 74.00, 18.00), "orbit": 136},
    "zone_c_vidarbha": {"bbox": (78.90, 20.90, 79.30, 21.30), "orbit": 165},
}


def _window_info(date_str: str, window_days: int) -> dict:
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    start = target - datetime.timedelta(days=window_days)
    end_inclusive = target + datetime.timedelta(days=window_days)
    return {
        "start": start.isoformat(),
        "end_inclusive": end_inclusive.isoformat(),
        "end_exclusive": (end_inclusive + datetime.timedelta(days=1)).isoformat(),
    }


def _matches_research_zone(bbox: dict | tuple[float, ...]) -> tuple[str, int] | None:
    values = tuple(float(bbox[k]) for k in ("lon_min", "lat_min", "lon_max", "lat_max")) if isinstance(bbox, dict) else tuple(map(float, bbox))
    for zone, config in RESEARCH_ORBIT_ZONES.items():
        if all(abs(a - b) <= 1e-5 for a, b in zip(values, config["bbox"])):
            return zone, config["orbit"]
    return None


def choose_shared_orbit(t1_counts: dict[int, int], t2_counts: dict[int, int], frozen_orbit: int | None = None) -> tuple[int | None, list[int], str]:
    """Choose one orbit present in both windows; prefer balanced, abundant coverage."""
    common = sorted(set(t1_counts).intersection(t2_counts))
    if frozen_orbit is not None:
        if frozen_orbit in common:
            return frozen_orbit, common, "frozen research-zone orbit"
        return None, common, "frozen research-zone orbit unavailable in one or both windows"
    if not common:
        return None, common, "no common descending relative orbit"
    selected = max(common, key=lambda orbit: (min(t1_counts[orbit], t2_counts[orbit]), t1_counts[orbit] + t2_counts[orbit], -orbit))
    return selected, common, "maximum shared coverage: highest minimum scene count, then highest total count, then lowest orbit ID"


def _require_s2_scenes(date_label: str, before_cloud_count: int, after_cloud_count: int, diagnostic: dict):
    if before_cloud_count == 0 or after_cloud_count == 0:
        reason = "no scenes in the requested window" if before_cloud_count == 0 else "all scenes were excluded by the scene cloud limit"
        _raise_stage_error(
            f"gee_s2_no_scenes_{date_label.lower()}",
            f"No Sentinel-2 scenes were available for {date_label} after applying the requested window and cloud limit ({reason}).",
            f"s2_{date_label.lower()}_scene_filter", diagnostic,
        )


def _require_s1_scenes(date_label: str, before_filters: int, after_filters: int, orbit_counts: dict[int, int], diagnostic: dict):
    if before_filters == 0 or after_filters == 0 or not orbit_counts:
        _raise_stage_error(
            f"gee_s1_no_scenes_{date_label.lower()}",
            f"No Sentinel-1 descending IW scenes with both VV and VH were available for {date_label} in the requested window.",
            f"s1_{date_label.lower()}_scene_filter", diagnostic,
        )


def _require_cloud_score_matches(date_label: str, linked_scene_count: int, diagnostic: dict):
    if linked_scene_count == 0:
        _raise_stage_error(
            "gee_cloud_score_no_coverage",
            f"Sentinel-2 scenes exist for {date_label}, but none have a matching Cloud Score+ quality scene. Try different dates or a wider window.",
            f"s2_{date_label.lower()}_cloud_score_link", diagnostic,
        )


def _select_shared_orbit(t1_counts: dict[int, int], t2_counts: dict[int, int], frozen_orbit: int | None, diagnostic: dict):
    common_candidates = sorted(set(t1_counts).intersection(t2_counts))
    if frozen_orbit is not None and frozen_orbit not in common_candidates:
        _raise_stage_error(
            "gee_frozen_s1_orbit_unavailable",
            f"The documented research-zone orbit {frozen_orbit} is not available in both requested date windows. The frozen research acquisition protocol was not changed.",
            "s1_frozen_research_orbit_validation",
            {
                **diagnostic,
                "s1_common_orbits": common_candidates,
                "frozen_relative_orbit": frozen_orbit,
                "selected_relative_orbit": None,
            },
        )
    orbit, common, reason = choose_shared_orbit(t1_counts, t2_counts, frozen_orbit)
    if orbit is None:
        _raise_stage_error(
            "gee_no_common_s1_orbit",
            "No single descending Sentinel-1 relative orbit has valid VV/VH scenes in both requested date windows. Increase the temporal search window or choose different dates; the selected window was not changed.",
            "s1_common_orbit_selection", {**diagnostic, "s1_common_orbits": common, "s1_orbit_selection_policy": reason, "selected_relative_orbit": None},
        )
    return orbit, common, reason


def _safe_error_reason(error: Exception) -> str:
    """Keep actionable stage diagnostics while removing paths, URLs and credentials."""
    import re
    reason = str(error)
    reason = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [redacted]", reason)
    reason = re.sub(r"(?i)(access[_ ]?token|refresh[_ ]?token|private[_ ]?key|client[_ ]?secret|token|password|secret|credential|authorization)(\s*[=:]\s*)[^\s,;]+", r"\1\2[redacted]", reason)
    reason = re.sub(r"(?i)\b(access[_ ]?token|refresh[_ ]?token|oauth token)\s+[A-Za-z0-9._~+/=-]{12,}", r"\1 [redacted]", reason)
    reason = re.sub(r"(?i)(projects?/)([A-Za-z0-9][A-Za-z0-9-]{3,})", r"\1[redacted]", reason)
    reason = re.sub(r"(?i)(project(?:\s+id)?\s*[=:]\s*)[A-Za-z0-9][A-Za-z0-9-]{3,}", r"\1[redacted]", reason)
    reason = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[service-account]", reason)
    reason = re.sub(r"(?i)https?://\S+", "[provider-url]", reason)
    reason = re.sub(r"(?i)([a-z]:\\|/)(?:[^\s:]+[\\/])+[^\s:]*", "[path]", reason)
    return reason[:240]


def _parse_integral_number(value, field_name: str) -> int:
    """Parse EE numeric metadata without truncating fractional or non-finite values."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must not be a boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} is not numeric: {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field_name} must be a finite whole number: {value!r}")
    return int(numeric)


def _diagnostic_base(date_t1: str, date_t2: str, window_days: int, max_cloud: int, bbox: tuple[float, ...]) -> dict:
    return {
        "provider": "Google Earth Engine",
        "requested_dates": {"t1": date_t1, "t2": date_t2},
        "search_windows": {"t1": _window_info(date_t1, window_days), "t2": _window_info(date_t2, window_days)},
        "window_days": window_days,
        "max_cloud_pct": max_cloud,
        "bbox": list(bbox),
    }


def _raise_stage_error(code: str, message: str, stage: str, diagnostic: dict, error: Exception | None = None):
    details = {**diagnostic, "stage": stage}
    if error is not None:
        if code in {"gee_api_unavailable", "gee_imagery_download_failed"}:
            # Initialization alone does not prove project access. EE often
            # authenticates lazily on the first collection request, so retain
            # auth and authorization failures at their actual acquisition stage.
            try:
                from gee_config import classify_gee_error
                classified_code, classified_message = classify_gee_error(error)
                if classified_code in {"gee_authentication_failed", "gee_project_access_denied"}:
                    code, message = classified_code, classified_message
            except Exception:
                pass
        details.update({"exception_type": type(error).__name__, "technical_reason": _safe_error_reason(error)})
    raise GEEAcquisitionError(code, message, details) from error


def _scene_metadata(collection):
    """Return bounded, non-sensitive scene IDs and UTC acquisition timestamps."""
    rows = collection.reduceColumns(
        ee.Reducer.toList(2), ["system:index", "system:time_start"]
    ).get("list").getInfo() or []
    scenes = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        timestamp = row[1]
        scenes.append({
            "scene_id": str(row[0]),
            "acquisition_datetime": datetime.datetime.fromtimestamp(
                int(timestamp) / 1000, tz=datetime.timezone.utc
            ).isoformat().replace("+00:00", "Z") if timestamp is not None else None,
        })
    return scenes


def _get_s2_composite(bbox: ee.Geometry, date_str: str, window_days: int, max_cloud: int, date_label: str, diagnostic: dict):
    """
    Return a median Sentinel-2 SR HARMONIZED composite within ±window_days,
    plus the n_clear observation count used by the trained Q feature.
    """
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start = (target - datetime.timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (target + datetime.timedelta(days=window_days + 1)).strftime("%Y-%m-%d")

    base_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox)
        .filterDate(start, end)
    )
    try:
        before_cloud_count = int(base_col.size().getInfo())
        col = base_col.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        after_cloud_count = int(col.size().getInfo())
    except Exception as error:
        _raise_stage_error("gee_api_unavailable", "Earth Engine could not query Sentinel-2 scenes. Check API/project access and retry.", f"s2_{date_label.lower()}_query", {**diagnostic, f"{date_label.lower()}_s2_count_before_cloud_filter": None, f"{date_label.lower()}_s2_count_after_cloud_filter": None}, error)

    date_diag = {
        **diagnostic,
        f"{date_label.lower()}_s2_count_before_cloud_filter": before_cloud_count,
        f"{date_label.lower()}_s2_count_after_cloud_filter": after_cloud_count,
    }
    _require_s2_scenes(date_label, before_cloud_count, after_cloud_count, date_diag)

    # Cloud Score+ masking
    # Match training preprocessing: link Cloud Score+ by source scene index.
    # Unmatched Cloud Score+ bands remain masked, so they cannot count as clear.
    cloud_score = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterBounds(bbox)
        .filterDate(start, end)
    )
    try:
        join = ee.Join.inner().apply(
            primary=col,
            secondary=cloud_score,
            condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )
        linked_scene_count = int(join.size().getInfo())
        linked = col.linkCollection(cloud_score, ["cs_cdf"])
        masked = linked.map(lambda img: img.updateMask(img.select("cs_cdf").gte(0.60)))
    except Exception as error:
        _raise_stage_error("gee_api_unavailable", "Earth Engine could not link Cloud Score+ quality observations to Sentinel-2 scenes.", f"s2_{date_label.lower()}_cloud_score_link", {**date_diag, f"{date_label.lower()}_cloud_score_linked_scene_count": None}, error)

    date_diag[f"{date_label.lower()}_cloud_score_linked_scene_count"] = linked_scene_count
    date_diag[f"{date_label.lower()}_quality_filtering_unit"] = "per-pixel cs_cdf >= 0.60 mask; no scene-level removal"
    _require_cloud_score_matches(date_label, linked_scene_count, date_diag)

    # n_clear: how many valid observations per pixel
    n_clear = masked.select("B4").count().rename("n_clear").unmask(0)

    composite = masked.select(S2_BANDS).median()

    # Add NDVI and NDBI
    eps = 1e-6
    b8, b4, b11 = composite.select("B8"), composite.select("B4"), composite.select("B11")
    ndvi = b8.subtract(b4).divide(b8.add(b4).add(eps)).rename("NDVI")
    ndbi = b11.subtract(b8).divide(b11.add(b8).add(eps)).rename("NDBI")

    try:
        scenes = _scene_metadata(col)
    except Exception as error:
        _raise_stage_error("gee_api_unavailable", f"Earth Engine could not read Sentinel-2 acquisition metadata for {date_label}.", f"s2_{date_label.lower()}_metadata", date_diag, error)
    date_diag[f"{date_label.lower()}_s2_scenes"] = scenes[:50]
    return composite.addBands([ndvi, ndbi, n_clear]), after_cloud_count, scenes, date_diag


def _get_s1_orbit_candidates(bbox: ee.Geometry, date_str: str, window_days: int, date_label: str, diagnostic: dict):
    """Query real descending IW dual-polarization S1 scenes and orbit counts."""
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    start = (target - datetime.timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (target + datetime.timedelta(days=window_days + 1)).strftime("%Y-%m-%d")
    label = date_label.lower()
    request_details = {
        **diagnostic,
        "requested_date": date_str,
        "search_window": {"start": start, "end_exclusive": end},
        "filters": {
            "collection": "COPERNICUS/S1_GRD",
            "bounds": list(bbox) if isinstance(bbox, (list, tuple)) else "requested AOI",
            "instrumentMode": "IW",
            "orbitProperties_pass": "DESCENDING",
            "transmitterReceiverPolarisation": ["VV", "VH"],
            "relativeOrbitNumber_start": "discover from eligible scenes",
        },
    }
    try:
        base = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(bbox).filterDate(start, end)
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", "Earth Engine could not construct the initial Sentinel-1 collection.",
            f"s1_{label}_collection_build", {**request_details, "operation": "ImageCollection/filterBounds/filterDate"}, error,
        )
    try:
        before_filters = base.size().getInfo()
        if isinstance(before_filters, bool) or not isinstance(before_filters, (int, float)) or int(before_filters) != before_filters or before_filters < 0:
            raise ValueError("Earth Engine returned an invalid initial scene count.")
        before_filters = int(before_filters)
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", "Earth Engine failed while counting the initial Sentinel-1 collection.",
            f"s1_{label}_base_count", {**request_details, "operation": "base.size().getInfo()", f"{label}_s1_count_before_filters": None}, error,
        )

    filter_details = {**request_details, f"{label}_s1_count_before_filters": before_filters}
    try:
        eligible = (
            base.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        )
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", "Earth Engine could not apply the Sentinel-1 acquisition filters.",
            f"s1_{label}_filter_build", {**filter_details, "operation": "filter(VV,VH,IW,DESCENDING)"}, error,
        )
    try:
        after_filters = eligible.size().getInfo()
        if isinstance(after_filters, bool) or not isinstance(after_filters, (int, float)) or int(after_filters) != after_filters or after_filters < 0:
            raise ValueError("Earth Engine returned an invalid eligible scene count.")
        after_filters = int(after_filters)
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", "Earth Engine failed while counting eligible Sentinel-1 scenes.",
            f"s1_{label}_filtered_count", {**filter_details, "operation": "eligible.size().getInfo()", f"{label}_s1_count_after_iw_desc_vv_vh_filters": None}, error,
        )

    date_diag = {
        **filter_details,
        f"{label}_s1_count_after_iw_desc_vv_vh_filters": after_filters,
        f"{label}_s1_orbits": None,
    }
    if after_filters == 0:
        _require_s1_scenes(date_label, before_filters, after_filters, {}, date_diag)

    try:
        histogram = eligible.aggregate_histogram("relativeOrbitNumber_start").getInfo()
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", "Earth Engine failed while querying Sentinel-1 relative-orbit metadata.",
            f"s1_{label}_orbit_histogram", {**date_diag, "operation": 'eligible.aggregate_histogram("relativeOrbitNumber_start").getInfo()'}, error,
        )

    try:
        if not isinstance(histogram, dict):
            raise ValueError("Orbit histogram response was not a dictionary.")
        orbit_counts = {}
        for raw_orbit, raw_count in histogram.items():
            orbit = _parse_integral_number(raw_orbit, "relative orbit ID")
            count = _parse_integral_number(raw_count, f"scene count for orbit {orbit}")
            if not 1 <= orbit <= 175:
                raise ValueError("Orbit histogram contained an invalid relative orbit ID.")
            if orbit in orbit_counts:
                raise ValueError("Orbit histogram contained duplicate normalized relative orbit IDs.")
            if count <= 0:
                raise ValueError("Orbit histogram contained a non-positive or non-integral scene count.")
            orbit_counts[orbit] = count
        if not orbit_counts:
            raise ValueError("Orbit histogram was empty despite a non-zero eligible scene count.")
        if sum(orbit_counts.values()) != after_filters:
            raise ValueError("Orbit histogram scene counts did not match the eligible scene count.")
    except Exception as error:
        _raise_stage_error(
            "gee_s1_orbit_metadata_invalid", "Earth Engine returned missing or malformed Sentinel-1 relative-orbit metadata.",
            f"s1_{label}_orbit_metadata_parse", {**date_diag, "operation": "parse relativeOrbitNumber_start histogram", "s1_orbit_histogram": histogram}, error,
        )

    date_diag[f"{label}_s1_orbits"] = orbit_counts
    _require_s1_scenes(date_label, before_filters, after_filters, orbit_counts, date_diag)
    return eligible, orbit_counts, date_diag


def _get_s1_composite(eligible_collection, relative_orbit: int, date_label: str, diagnostic: dict):
    """Composite one selected orbit in linear power, then convert to dB."""

    def to_natural(img):
        return ee.Image(10).pow(img.divide(10))

    def to_db(img):
        return img.log10().multiply(10)

    label = date_label.lower()
    try:
        col = eligible_collection.filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit)).select(["VV", "VH"])
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", f"Earth Engine could not construct the selected Sentinel-1 {date_label} orbit collection.",
            f"s1_{label}_selected_orbit_filter", {**diagnostic, "selected_relative_orbit": relative_orbit, "operation": "filter(relativeOrbitNumber_start).select(VV,VH)"}, error,
        )
    try:
        scene_count = col.size().getInfo()
        if isinstance(scene_count, bool) or not isinstance(scene_count, (int, float)) or int(scene_count) != scene_count or scene_count < 0:
            raise ValueError("Earth Engine returned an invalid selected-orbit scene count.")
        scene_count = int(scene_count)
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", f"Earth Engine failed while counting selected-orbit Sentinel-1 scenes for {date_label}.",
            f"s1_{label}_selected_orbit_count", {**diagnostic, "selected_relative_orbit": relative_orbit, "operation": "selected_orbit.size().getInfo()"}, error,
        )
    if scene_count == 0:
        _raise_stage_error(
            "gee_s1_orbit_scene_count_inconsistent", f"The selected Sentinel-1 orbit {relative_orbit} disappeared before {date_label} composite creation.",
            f"s1_{label}_selected_orbit_count", {**diagnostic, "selected_relative_orbit": relative_orbit, "selected_orbit_scene_count": 0},
        )

    try:
        composite = to_db(col.map(to_natural).median())
        vv = composite.select("VV")
        vh = composite.select("VH")
        vhvv = vh.subtract(vv).rename("VH_VV")
        composite = composite.addBands(vhvv)
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", f"Earth Engine could not construct the selected Sentinel-1 {date_label} composite.",
            f"s1_{label}_composite_build", {**diagnostic, "selected_relative_orbit": relative_orbit, "selected_orbit_scene_count": scene_count, "operation": "linear_power_median_to_db_and_VH_VV"}, error,
        )

    try:
        scenes = _scene_metadata(col)
    except Exception as error:
        _raise_stage_error(
            "gee_api_unavailable", f"Earth Engine could not read selected Sentinel-1 acquisition metadata for {date_label}.",
            f"s1_{label}_selected_scene_metadata", {**diagnostic, "selected_relative_orbit": relative_orbit, "selected_orbit_scene_count": scene_count, "operation": "selected_scene.reduceColumns(system:index,system:time_start).getInfo()"}, error,
        )
    return composite, scene_count, scenes


def _get_s1_pair_composites(t1_candidates, t2_candidates, relative_orbit: int, diagnostic: dict):
    """Build both dates with the same selected relative orbit."""
    return (
        _get_s1_composite(t1_candidates, relative_orbit, "T1", diagnostic),
        _get_s1_composite(t2_candidates, relative_orbit, "T2", diagnostic),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public fetch function
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_optimal_dimensions(lon_min, lat_min, lon_max, lat_max, base_dim=512, min_dim=256, max_dim=768):
    lon_span = abs(float(lon_max) - float(lon_min))
    lat_span = abs(float(lat_max) - float(lat_min))
    lat_mid = (float(lat_min) + float(lat_max)) / 2.0
    lat_cos = max(0.1, math.cos(math.radians(lat_mid)))
    aspect = (lon_span * lat_cos) / (lat_span + 1e-9)

    base = max(base_dim, 512)
    if aspect >= 1.0:
        w = base
        h = int(round(base / aspect))
    else:
        h = base
        w = int(round(base * aspect))

    w = max(min_dim, min(max_dim, (w // 32) * 32))
    h = max(min_dim, min(max_dim, (h // 32) * 32))
    return int(h), int(w)


def fetch_imagery(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float,
    date_t1: str, date_t2: str,
    window_days: int = 15,
    max_cloud: int = 30,
    thumb_size: int = 512,
) -> dict:
    coords = (float(lon_min), float(lat_min), float(lon_max), float(lat_max))
    diagnostic = _diagnostic_base(date_t1, date_t2, int(window_days), int(max_cloud), coords)
    lat_mid = (coords[1] + coords[3]) / 2
    lat_rad = math.radians(lat_mid)
    width_m = abs(coords[2] - coords[0]) * (111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad))
    height_m = abs(coords[3] - coords[1]) * (111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad))
    diagnostic["estimated_bbox_dimensions_m"] = {"width": round(width_m, 1), "height": round(height_m, 1)}
    diagnostic["nominal_resolution_m"] = 10
    diagnostic["estimated_10m_grid_shape"] = [round(height_m / 10), round(width_m / 10)]
    if width_m < 1280 or height_m < 1280:
        _raise_stage_error(
            "area_too_small",
            f"The selected area is approximately {width_m / 1000:.2f} km × {height_m / 1000:.2f} km. Geo-Nexus AI needs at least 1.28 km in both dimensions at 10 m resolution; enlarge the area before acquisition.",
            "area_preflight", diagnostic,
        )

    try:
        init_ee()
    except Exception as error:
        if getattr(error, "code", None):
            _raise_stage_error(error.code, getattr(error, "message", str(error)), "gee_initialization", diagnostic, error)
        from gee_config import classify_gee_error
        code, message = classify_gee_error(error)
        _raise_stage_error(code, message, "gee_initialization", diagnostic, error)

    try:
        bbox = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        center_lon = (float(lon_min) + float(lon_max)) / 2
        output_crs = "EPSG:32643" if center_lon < 78 else "EPSG:32644"
    except Exception as error:
        _raise_stage_error("gee_invalid_input_grid", "The selected bounds could not be represented on the model's projected grid.", "geometry_projection", diagnostic, error)

    out_h, out_w = _calculate_optimal_dimensions(lon_min, lat_min, lon_max, lat_max, base_dim=thumb_size)
    thumb_dim = f"{out_w}x{out_h}"

    s2_t1, t1_count, t1_s2_scenes, t1_s2_diag = _get_s2_composite(bbox, date_t1, window_days, max_cloud, "T1", diagnostic)
    s2_t2, t2_count, t2_s2_scenes, t2_s2_diag = _get_s2_composite(
        bbox, date_t2, window_days, max_cloud, "T2", {**diagnostic, **t1_s2_diag}
    )
    s1_t1_candidates, t1_orbits, t1_s1_diag = _get_s1_orbit_candidates(
        bbox, date_t1, window_days, "T1", {**diagnostic, **t1_s2_diag, **t2_s2_diag}
    )
    s1_t2_candidates, t2_orbits, t2_s1_diag = _get_s1_orbit_candidates(
        bbox, date_t2, window_days, "T2", {**diagnostic, **t1_s1_diag}
    )

    research_zone = _matches_research_zone(coords)
    frozen_orbit = research_zone[1] if research_zone else None
    relative_orbit, common_orbits, selection_reason = _select_shared_orbit(t1_orbits, t2_orbits, frozen_orbit, {**diagnostic, **t1_s1_diag, **t2_s1_diag, "research_zone": research_zone[0] if research_zone else None})
    orbit_diagnostic = {
        **diagnostic,
        **t1_s1_diag,
        **t2_s1_diag,
        "s1_common_orbits": common_orbits,
        "s1_orbit_selection_policy": selection_reason,
        "research_zone": research_zone[0] if research_zone else None,
        "selected_relative_orbit": relative_orbit,
    }
    try:
        t1_s1_result, t2_s1_result = _get_s1_pair_composites(s1_t1_candidates, s1_t2_candidates, relative_orbit, orbit_diagnostic)
        s1_t1, t1_s1_count, t1_s1_scenes = t1_s1_result
        s1_t2, t2_s1_count, t2_s1_scenes = t2_s1_result
    except Exception as error:
        if isinstance(error, GEEAcquisitionError):
            raise
        _raise_stage_error("gee_api_unavailable", "Earth Engine could not construct the selected common Sentinel-1 orbit composites.", "s1_composite", orbit_diagnostic, error)

    acquisition_diagnostics = {
        **diagnostic,
        **t1_s2_diag,
        **t2_s2_diag,
        **t1_s1_diag,
        **t2_s1_diag,
        "s1_common_orbits": common_orbits,
        "s1_orbit_selection_policy": selection_reason,
        "research_zone": research_zone[0] if research_zone else None,
        "selected_relative_orbit": relative_orbit,
        "selected_s1_scenes": {"t1": t1_s1_scenes[:50], "t2": t2_s1_scenes[:50]},
        "selected_s1_scene_counts": {"t1": t1_s1_count, "t2": t2_s1_count},
        "crs": output_crs,
        "nominal_resolution_m": 10,
        "estimated_10m_grid_shape": diagnostic["estimated_10m_grid_shape"],
    }

    # ── RGB thumbnails (B4=Red, B3=Green, B2=Blue) ──────────────────────────
    vis_rgb = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.4}
    try:
        t1_thumb_url = s2_t1.getThumbURL({**vis_rgb, "region": bbox, "dimensions": thumb_dim, "format": "png"})
        t2_thumb_url = s2_t2.getThumbURL({**vis_rgb, "region": bbox, "dimensions": thumb_dim, "format": "png"})
        t1_rgb_b64 = _url_to_b64(t1_thumb_url)
        t2_rgb_b64 = _url_to_b64(t2_thumb_url)
    except Exception as error:
        _raise_stage_error("gee_imagery_download_failed", "Earth Engine imagery thumbnails could not be retrieved for the selected extent.", "thumbnail_download", {**acquisition_diagnostics, "requested_thumbnail_shape": [out_h, out_w]}, error)

    try:
        t1_tensor, t1_quality_stats = _build_tensor(s2_t1, s1_t1, bbox, output_crs)
        t2_tensor, t2_quality_stats = _build_tensor(s2_t2, s1_t2, bbox, output_crs)
    except Exception as error:
        _raise_stage_error("gee_imagery_download_failed", "Earth Engine model-band download or array decoding failed. Check the selected extent, projection, and provider availability.", "model_tensor_download", acquisition_diagnostics, error)
    if t1_tensor.shape != t2_tensor.shape:
        _raise_stage_error("gee_invalid_input_grid", "T1 and T2 downloads produced different raster grids.", "t1_t2_grid_alignment", {**acquisition_diagnostics, "t1_shape": list(t1_tensor.shape), "t2_shape": list(t2_tensor.shape)})
    if t1_tensor.ndim != 3 or t1_tensor.shape[0] != len(CHANNEL_ORDER) or min(t1_tensor.shape[1:]) < 128:
        _raise_stage_error(
            "gee_invalid_input_grid",
            "Earth Engine returned a model grid smaller than 128 × 128 pixels at the requested 10 m resolution. Enlarge the selected area and retry.",
            "model_grid_validation",
            {**acquisition_diagnostics, "output_grid_shape": list(t1_tensor.shape), "minimum_grid_shape": [128, 128]},
        )

    for date_label, stats in (("T1", t1_quality_stats), ("T2", t2_quality_stats)):
        if stats.get("positive_fraction", 0) <= 0:
            _raise_stage_error(
                "gee_cloud_score_no_coverage",
                f"Sentinel-2 scenes exist for {date_label}, but Cloud Score+ masking left no clear observations in the selected area.",
                f"s2_{date_label.lower()}_quality_coverage",
                {**acquisition_diagnostics, f"{date_label.lower()}_quality_raw_n_clear": stats},
            )

    target_t1 = datetime.datetime.strptime(date_t1, "%Y-%m-%d")
    target_t2 = datetime.datetime.strptime(date_t2, "%Y-%m-%d")
    acquisition_diagnostics["output_grid_shape"] = list(t1_tensor.shape[1:])
    meta = {
        "provider": "Google Earth Engine",
        "preprocessing_id": "gee_s2_s1_median_composite_v2",
        "s1_used": True,
        "sar_channels_available": True,
        "s1_note": "Sentinel-1 GRD VV/VH composite and VH_VV difference",
        "crs": output_crs,
        "relative_orbit": relative_orbit,
        "relative_orbit_selection_reason": selection_reason,
        "s1_common_orbits": common_orbits,
        "research_zone": research_zone[0] if research_zone else None,
        "orbit_pass": "DESCENDING",
        "output_shape": list(t1_tensor.shape[1:]),
        "pixel_resolution_m": 10,
        "resolution_note": "GEE model inputs use a 10 m UTM grid; area values are approximate at bbox edges.",
        "quality_channel": "clip(n_clear / 8, 0, 1)",
        "quality_channel_status": "real",
        "quality_channel_source": "per-pixel count of Sentinel-2 B4 observations after linked Cloud Score+ cs_cdf >= 0.60 masking, divided by 8 and clipped to [0,1]",
        "t1_quality_raw_stats": t1_quality_stats,
        "t2_quality_raw_stats": t2_quality_stats,
        "cloud_score_join": "Cloud Score+ linked by source system:index; unmatched quality bands remain masked",
        "s2_collection": "COPERNICUS/S2_SR_HARMONIZED",
        "s1_collection": "COPERNICUS/S1_GRD",
        "t1_date": date_t1,
        "t2_date": date_t2,
        "t1_window": f"{(target_t1 - datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')} → "
                     f"{(target_t1 + datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')}",
        "t2_window": f"{(target_t2 - datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')} → "
                     f"{(target_t2 + datetime.timedelta(days=window_days)).strftime('%Y-%m-%d')}",
        "t1_scene_count": t1_count,
        "t2_scene_count": t2_count,
        "t1_s1_scene_count": t1_s1_count,
        "t2_s1_scene_count": t2_s1_count,
        "t1_s2_scenes": t1_s2_scenes[:50],
        "t2_s2_scenes": t2_s2_scenes[:50],
        "t1_s1_scenes": t1_s1_scenes[:50],
        "t2_s1_scenes": t2_s1_scenes[:50],
        "t1_acquisition_datetimes": [scene["acquisition_datetime"] for scene in t1_s2_scenes[:50]],
        "t2_acquisition_datetimes": [scene["acquisition_datetime"] for scene in t2_s2_scenes[:50]],
        "t1_s1_acquisition_datetimes": [scene["acquisition_datetime"] for scene in t1_s1_scenes[:50]],
        "t2_s1_acquisition_datetimes": [scene["acquisition_datetime"] for scene in t2_s1_scenes[:50]],
        "provider_attempts": [{"provider": "Google Earth Engine", "status": "success"}],
        "acquisition_diagnostics": acquisition_diagnostics,
        "window_days": window_days,
        "max_cloud_pct": max_cloud,
        "clear_score_threshold": 0.60,
        "input_channel_order": CHANNEL_ORDER,
        "bbox": [lon_min, lat_min, lon_max, lat_max],
    }

    return {
        "t1_rgb_b64": t1_rgb_b64,
        "t2_rgb_b64": t2_rgb_b64,
        "t1_tensor": t1_tensor,   # float32 [17, H, W], training scales retained
        "t2_tensor": t2_tensor,
        "meta": meta,
    }


def _build_tensor(s2_img, s1_img, bbox, output_crs: str) -> tuple[np.ndarray, dict]:
    """
    Download the trained [17, H, W] contract and raw n_clear stats on a 10 m UTM grid.
    Channel order: B2..B12(11) + NDVI + NDBI + VV + VH + VH_VV + Q(n_clear)
    Returns (encoded_tensor, raw_n_clear_statistics).
    """
    bands = S2_BANDS + ["NDVI", "NDBI", "VV", "VH", "VH_VV", "n_clear"]

    area_m2 = float(bbox.area(maxError=1).getInfo())
    if area_m2 / 100.0 > 400_000:
        raise RuntimeError("Selected area exceeds the single-request 10 m model grid limit; select a smaller area.")

    # Combine S2 and S1 into one image
    combined = s2_img.addBands(s1_img)

    # Use getDownloadURL for numpy-compatible download
    url = combined.select(bands).getDownloadURL({
        "scale": 10,
        "crs": output_crs,
        "region": bbox,
        "format": "NPY",
    })

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    arr = np.load(io.BytesIO(resp.content))  # structured numpy array

    channels = []
    for b in bands:
        ch = arr[b].astype(np.float32)
        channels.append(ch)

    stacked = np.stack(channels, axis=0)  # [17, H, W]

    quality = stacked[16]
    finite_quality = quality[np.isfinite(quality)].astype(np.float64)
    quality_stats: dict = {
        "source_feature": "n_clear before Q scaling/clipping",
        "valid_pixel_fraction": float(np.isfinite(quality).mean()),
        "positive_fraction": float((quality > 0).mean()),
        "min": float(finite_quality.min()) if finite_quality.size else None,
        "max": float(finite_quality.max()) if finite_quality.size else None,
        "mean": float(finite_quality.mean()) if finite_quality.size else None,
        "std": float(finite_quality.std()) if finite_quality.size else None,
    }
    if finite_quality.size:
        pcts = np.percentile(finite_quality, [5, 25, 50, 75, 95]).tolist()
        quality_stats["percentiles_p5_p25_p50_p75_p95"] = [round(p, 3) for p in pcts]
        max_val = int(finite_quality.max())
        if max_val <= 30:  # Only emit histogram when scene counts are small
            value_counts: dict[str, int] = {}
            for v in range(0, max_val + 1):
                cnt = int(np.sum(quality == v))
                if cnt > 0:
                    value_counts[str(v)] = cnt
            quality_stats["value_counts"] = value_counts
    return _encode_model_tensor(stacked), quality_stats


def _encode_model_tensor(raw: np.ndarray) -> np.ndarray:
    """Convert raw GEE bands to training storage scales without quantizing floats."""
    if raw.ndim != 3 or raw.shape[0] != 17:
        raise ValueError("Expected 17 channels in manifest order.")
    encoded = raw.astype(np.float32, copy=True)
    # Earth Engine Sentinel-2 SR bands are already DN scaled by 10,000.
    encoded[:11] = np.clip(encoded[:11], -32768, 32767)
    encoded[13:16] = np.clip(encoded[13:16] * 100, -32768, 32767)
    encoded[16] = np.clip(encoded[16] / 8.0, 0.0, 1.0)
    return encoded


def _url_to_b64(url: str) -> str:
    """Download image from URL and return as base64-encoded PNG string."""
    import base64
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
