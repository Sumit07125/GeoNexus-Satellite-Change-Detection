"""Safe, shared Google Earth Engine configuration checks.

This module intentionally reports readiness categories only. It never returns
project IDs, key contents, credential values, or local secret paths.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


class GEEConfigurationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


_PLACEHOLDERS = {
    "", "notset", "unset", "none", "null", "changeme", "placeholder",
    "yourgcpprojectid", "yourprojectid", "yourgooglecloudprojectid",
    "yourgeeprojectid", "gcpprojectid",
}


def _project_placeholder(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", value.strip().lower())
    return normalized in _PLACEHOLDERS or normalized.startswith("your") and normalized.endswith("projectid")


def _local_credential_file() -> Path | None:
    """Resolve Earth Engine and ADC credential file locations without reading secrets."""
    try:
        import ee
        path = Path(ee.oauth.get_credentials_path()).expanduser()
        if path.is_file() and path.stat().st_size:
            return path
    except Exception:
        pass
    adc_candidates = [Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")).expanduser()]
    cloud_config = os.environ.get("CLOUDSDK_CONFIG", "")
    if cloud_config:
        adc_candidates.append(Path(cloud_config).expanduser() / "application_default_credentials.json")
    if os.name == "nt":
        app_data = os.environ.get("APPDATA", "")
        if app_data:
            adc_candidates.append(Path(app_data) / "gcloud" / "application_default_credentials.json")
    else:
        adc_candidates.append(Path.home() / ".config" / "gcloud" / "application_default_credentials.json")
    for adc in adc_candidates:
        if str(adc) and adc.is_file() and adc.stat().st_size:
            return adc
    return None


def get_gee_configuration() -> dict:
    """Return sanitized readiness state shared by startup, health, and AI requests."""
    project = os.environ.get("GEE_PROJECT_ID", "").strip()
    if not project or _project_placeholder(project):
        return {
            "status": "not_configured", "code": "gee_not_configured",
            "configured": False,
            "message": "Set GEE_PROJECT_ID to a real Google Cloud project with the Earth Engine API enabled.",
        }

    service_key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "").strip()
    if service_key:
        key_path = Path(service_key).expanduser()
        try:
            with key_path.open("r", encoding="utf-8") as key_file:
                key_data = json.load(key_file)
            if (key_data.get("type") != "service_account" or
                    not key_data.get("client_email") or not key_data.get("private_key")):
                raise ValueError("invalid service account fields")
        except Exception:
            return {
                "status": "service_account_invalid", "code": "gee_service_account_invalid",
                "configured": False,
                "message": "The configured Earth Engine service-account key is missing or invalid. Check its path and service-account JSON format.",
            }
        return {"status": "configured", "code": None, "configured": True, "auth_method": "service_account"}

    if _local_credential_file() is None:
        return {
            "status": "credentials_missing", "code": "gee_credentials_missing",
            "configured": False,
            "message": "No local Earth Engine credentials were found. Run `earthengine authenticate` or configure a valid service-account key.",
        }
    return {"status": "configured", "code": None, "configured": True, "auth_method": "local_credentials"}


def ensure_gee_configuration() -> dict:
    readiness = get_gee_configuration()
    if not readiness["configured"]:
        raise GEEConfigurationError(readiness["code"], readiness["message"])
    return readiness


def classify_gee_error(error: Exception) -> tuple[str, str]:
    """Map EE failures to safe actionable categories without exposing provider text."""
    text = str(error).lower()
    if any(term in text for term in ("credential", "oauth", "unauthenticated", "invalid_grant", "authentication", "token")):
        return "gee_authentication_failed", "Google Earth Engine authentication failed. Re-run `earthengine authenticate` or verify the configured service-account key."
    if any(term in text for term in ("permission", "forbidden", "403", "not authorized", "access denied")) or (
        "project" in text and any(term in text for term in ("invalid", "not found", "does not exist", "not registered"))
    ):
        return "gee_project_access_denied", "Google Earth Engine cannot access this project. Enable the Earth Engine API and grant the account Earth Engine access to the configured project."
    if any(term in text for term in ("no sentinel-1", "no sentinel 1", "no sentinel-2", "no sentinel 2", "no scenes", "no imagery", "empty collection")):
        return "gee_imagery_unavailable", "Google Earth Engine found no suitable satellite imagery for one or both selected date windows. Try a wider search window or different dates."
    if any(term in text for term in ("timeout", "connection", "unavailable", "503", "service not available")):
        return "gee_api_unavailable", "The Earth Engine API is unavailable. Check network access and try again."
    return "gee_provider_failed", "Google Earth Engine could not acquire valid imagery. Check project access and scene availability, then retry."
