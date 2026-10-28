"""Focused tests for engine identity, input contract, API validation, and storage migration."""
import os
import base64
import io
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as api
import storage_manager
import inference
import gee_config
import gee_service
import sentinel_stac_service as stac_service
from gee_service import _encode_model_tensor
from diagnostics.region_pair_diagnostic import generate_candidates, run_diagnostic


class ReliabilityContractTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()
        self.bbox = {"lon_min": 73.0, "lat_min": 18.0, "lon_max": 73.1, "lat_max": 18.1}
        self.payload = {"bbox": self.bbox, "date_t1": "2021-01-01", "date_t2": "2022-01-01"}

    def test_health_reports_distinct_engines(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["engines"], ["geonexus_ai", "spectral_scva"])
        self.assertIn("ai_readiness", response.json)
        self.assertFalse(response.json["ai_readiness"]["ready"])

    def test_gee_configuration_rejects_missing_and_case_insensitive_placeholders(self):
        for project in (None, "", "  YOUR-GCP-PROJECT-ID  ", "Not Set", "your_gee_project_id"):
            with self.subTest(project=project), patch.dict(os.environ, {} if project is None else {"GEE_PROJECT_ID": project}, clear=project is None):
                state = gee_config.get_gee_configuration()
                self.assertEqual(state["code"], "gee_not_configured")
                self.assertFalse(state["configured"])

    def test_gee_configuration_requires_user_credentials_or_valid_service_account(self):
        with patch.dict(os.environ, {"GEE_PROJECT_ID": "real-project", "GEE_SERVICE_ACCOUNT_KEY": ""}), patch.object(gee_config, "_local_credential_file", return_value=None):
            self.assertEqual(gee_config.get_gee_configuration()["code"], "gee_credentials_missing")
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_key = Path(temp_dir) / "service.json"
            invalid_key.write_text('{"type":"user_credentials"}', encoding="utf-8")
            with patch.dict(os.environ, {"GEE_PROJECT_ID": "real-project", "GEE_SERVICE_ACCOUNT_KEY": str(invalid_key)}):
                self.assertEqual(gee_config.get_gee_configuration()["code"], "gee_service_account_invalid")

    def test_gee_authentication_and_project_access_errors_have_safe_categories(self):
        self.assertEqual(gee_config.classify_gee_error(RuntimeError("invalid_grant oauth token"))[0], "gee_authentication_failed")
        self.assertEqual(gee_config.classify_gee_error(RuntimeError("403 project permission denied"))[0], "gee_project_access_denied")
        self.assertEqual(gee_config.classify_gee_error(RuntimeError("No Sentinel-1 scenes"))[0], "gee_imagery_unavailable")

    def test_gee_init_uses_configured_credentials_and_classifies_auth_failure(self):
        config = {"configured": True, "auth_method": "local_credentials"}
        with patch.dict(os.environ, {"GEE_PROJECT_ID": "test-project"}), \
             patch("gee_config.ensure_gee_configuration", return_value=config), \
             patch.object(gee_service.ee, "Initialize", side_effect=RuntimeError("invalid_grant")), \
             patch.object(gee_service, "_EE_INITIALIZED", False):
            with self.assertRaises(gee_service.GEEProviderError) as caught:
                gee_service.init_ee()
        self.assertEqual(caught.exception.code, "gee_authentication_failed")
        self.assertNotIn("invalid_grant", str(caught.exception))

    def test_ai_preflight_fails_before_queue_but_spectral_route_stays_available(self):
        no_gee = {"status": "credentials_missing", "code": "gee_credentials_missing", "configured": False, "message": "Configure Earth Engine credentials."}
        with patch("app.get_gee_configuration", return_value=no_gee), patch("app.threading.Thread") as thread:
            ai = self.client.post("/api/detect/start", json=self.payload)
            spectral = self.client.post("/api/detect/spectral/start", json=self.payload)
        self.assertEqual(ai.status_code, 503)
        self.assertEqual(ai.json["error_code"], "gee_credentials_missing")
        self.assertEqual(ai.json["meta"]["requested_dates"], {"t1": "2021-01-01", "t2": "2022-01-01"})
        thread.assert_called_once()
        self.assertEqual(spectral.status_code, 200)

    def test_health_does_not_call_checkpoint_file_presence_ai_ready(self):
        ready = {"status": "configured", "code": None, "configured": True}
        with patch("app.get_gee_configuration", return_value=ready):
            response = self.client.get("/api/health")
        self.assertFalse(response.json["ai_readiness"]["ready"])
        self.assertIn(response.json["ai_readiness"]["status"], ("checkpoint_available_unverified", "checkpoint_missing"))

    def test_invalid_bbox_date_and_engine_threshold_are_rejected(self):
        cases = [
            ("/api/detect/start", {**self.payload, "bbox": {**self.bbox, "lon_min": 73.2}}, 400),
            ("/api/detect/start", {**self.payload, "date_t2": "2020-01-01"}, 400),
            ("/api/detect/start", {**self.payload, "threshold": "not-a-number"}, 400),
            ("/api/detect/start", {**self.payload, "threshold": 1.7}, 400),
            ("/api/detect/spectral/start", {**self.payload, "threshold": float("nan")}, 400),
            ("/api/detect/spectral/start", {**self.payload, "threshold": 1.7}, 200),
        ]
        with patch("app.threading.Thread"):
            for path, payload, expected in cases:
                with self.subTest(path=path, payload=payload):
                    response = self.client.post(path, json=payload)
                    self.assertEqual(response.status_code, expected)

    def test_both_valid_endpoints_start_as_distinct_async_jobs(self):
        ready = {"status": "configured", "code": None, "configured": True, "auth_method": "local_credentials"}
        with patch("app.get_gee_configuration", return_value=ready), patch("app.threading.Thread"):
            ai = self.client.post("/api/detect/start", json=self.payload)
            spectral = self.client.post("/api/detect/spectral/start", json=self.payload)
        self.assertEqual(ai.status_code, 200)
        self.assertEqual(spectral.status_code, 200)
        self.assertEqual(api._jobs[ai.json["job_id"]]["engine"], "geonexus_ai")
        self.assertEqual(api._jobs[spectral.json["job_id"]]["engine"], "spectral_scva")

    def test_trained_channel_encoding_preserves_float_features(self):
        raw = np.zeros((17, 2, 2), dtype=np.float32)
        raw[:11] = 2000.0
        raw[11] = 0.42
        raw[12] = -0.13
        raw[13] = -10.0
        raw[14] = -17.0
        raw[15] = -7.0
        raw[16] = 4.0
        encoded = _encode_model_tensor(raw)
        self.assertEqual(encoded.dtype, np.float32)
        np.testing.assert_allclose(encoded[:11], 2000.0)
        np.testing.assert_allclose(encoded[11], 0.42)
        np.testing.assert_allclose(encoded[12], -0.13)
        np.testing.assert_allclose(encoded[13:16], raw[13:16] * 100)
        np.testing.assert_allclose(encoded[16], 0.5)

    def test_production_checkpoint_loads_strictly_and_runs_one_patch(self):
        y, x = np.mgrid[:128, :128]
        t1 = np.zeros((17, 128, 128), dtype=np.float32)
        t1[:11] = 1200 + x[None] * 0.1
        t1[11] = 0.15
        t1[12] = -0.08
        t1[13] = -950
        t1[14] = -1650
        t1[15] = -700
        t1[16] = y / 127
        t2 = t1.copy()
        t2[2] += 30
        rgb = Image.new("RGB", (128, 128), color=(90, 110, 80))
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        result = inference.run_inference(t1, t2, base64.b64encode(buffer.getvalue()).decode("ascii"))
        self.assertEqual(result["model_version"], "3.2.0")
        self.assertEqual(result["stats"]["output_size"], [128, 128])
        self.assertIn("binary_mask_b64", result)

    def test_stac_placeholders_are_rejected_for_ai_but_allowed_for_spectral(self):
        tensor = np.ones((17, 128, 128), dtype=np.float32) * 1000
        tensor[13:16] = -1000
        tensor[16] = 0  # the real STAC implementation marks this slot as a constant placeholder
        imagery = {"t1_tensor": tensor, "t2_tensor": tensor.copy(), "meta": {"provider": "Sentinel-2 STAC", "s1_used": False, "sar_channels_available": False, "quality_channel_status": "placeholder"}}
        with self.assertRaises(api.PipelineError) as caught:
            api._validate_imagery(imagery, require_sar=True)
        self.assertEqual(caught.exception.code, "missing_sar_channels")
        api._validate_imagery(imagery, require_sar=False)

    def test_stac_metadata_keeps_requested_dates_separate_from_scene_datetimes(self):
        from types import SimpleNamespace
        first = SimpleNamespace(id="t1-scene", properties={"eo:cloud_cover": 4}, datetime="2021-03-14T06:20:00Z")
        second = SimpleNamespace(id="t2-scene", properties={"eo:cloud_cover": 6}, datetime="2024-03-16T06:30:00Z")
        with patch.object(stac_service, "_search_best_scene", side_effect=[(first, []), (second, [])]), \
             patch.object(stac_service, "_load_s1_means", return_value=([0.0, 0.0, 0.0], 5.0)), \
             patch.object(stac_service, "_calculate_optimal_dimensions", return_value=(128, 128)), \
             patch.object(stac_service, "_get_band", return_value=np.full((128, 128), 1200, dtype=np.float32)), \
             patch.object(stac_service, "_rgb_from_ch_arrays", return_value="rgb"):
            imagery = stac_service.fetch_imagery(
                72.8434, 19.1163, 72.8625, 19.1343,
                "2021-03-15", "2024-03-15", window_days=15, max_cloud=30,
            )
        meta = imagery["meta"]
        self.assertEqual((meta["t1_date"], meta["t2_date"]), ("2021-03-15", "2024-03-15"))
        self.assertEqual(meta["t1_acquisition_datetime"], first.datetime)
        self.assertEqual(meta["t2_acquisition_datetime"], second.datetime)

    def _imagery(self, s1_used=False):
        tensor = np.full((17, 128, 128), 1200, dtype=np.float32)
        tensor[11] = 0.15
        tensor[12] = -0.08
        tensor[13:16] = np.array([-900, -1700, -800], dtype=np.float32)[:, None, None]
        tensor[16] = np.indices((128, 128))[0] / 127
        return {
            "t1_tensor": tensor,
            "t2_tensor": tensor.copy(),
            "t1_rgb_b64": "t1-rgb",
            "t2_rgb_b64": "t2-rgb",
            "meta": {
                "provider": "Google Earth Engine",
                "preprocessing_id": "gee_s2_s1_median_composite_v2",
                "s1_used": s1_used,
                "sar_channels_available": s1_used,
                "quality_channel_status": "real",
                "quality_channel": "clip(n_clear / 8, 0, 1)",
                "t1_quality_raw_stats": {"std": 2.0, "valid_pixel_fraction": 1.0, "min": 0.0, "max": 8.0},
                "t2_quality_raw_stats": {"std": 2.0, "valid_pixel_fraction": 1.0, "min": 0.0, "max": 8.0},
                "t1_date": "2021-01-01", "t2_date": "2022-01-01",
                "t1_scene_count": 8, "t2_scene_count": 9,
                "t1_s1_scene_count": 4, "t2_s1_scene_count": 3,
                "crs": "EPSG:32643", "pixel_resolution_m": 10,
            },
        }

    def test_valid_gee_quality_variation_passes_complete_ai_contract(self):
        imagery = self._imagery(s1_used=True)
        api._validate_imagery(imagery, require_sar=True)

    def test_missing_sentinel1_scene_count_is_rejected_before_inference(self):
        imagery = self._imagery(s1_used=True)
        imagery["meta"]["t2_s1_scene_count"] = 0
        with self.assertRaises(api.PipelineError) as caught:
            api._validate_imagery(imagery, require_sar=True)
        self.assertEqual(caught.exception.code, "imagery_unavailable")
        self.assertEqual(caught.exception.details["date"], "2022-01-01")

    def test_constant_q_without_raw_nclear_heterogeneity_is_rejected(self):
        imagery = self._imagery(s1_used=True)
        imagery["t1_tensor"][16] = 0.5
        imagery["meta"]["t1_quality_raw_stats"]["std"] = 0.0
        with self.assertRaises(api.PipelineError) as caught:
            api._validate_imagery(imagery, require_sar=True)
        self.assertEqual(caught.exception.code, "invalid_quality_channel")
        self.assertEqual(caught.exception.details["channel"], "Q")
        self.assertEqual(caught.exception.details["date"], "2021-01-01")

    def test_clipped_constant_q_is_allowed_when_raw_nclear_meets_training_gate(self):
        imagery = self._imagery(s1_used=True)
        imagery["t1_tensor"][16] = 1.0
        imagery["meta"]["t1_scene_count"] = 30
        imagery["meta"]["t1_quality_raw_stats"].update({"std": 3.0, "min": 10.0, "max": 20.0})
        api._validate_imagery(imagery, require_sar=True)

    def test_raw_nclear_diagnostics_reject_nonfinite_and_impossible_counts(self):
        cases = [
            ({"valid_pixel_fraction": 0.99}, "non-finite or missing"),
            ({"max": 100.0}, "inconsistent with the acquired Sentinel-2 scene count"),
        ]
        for update, expected in cases:
            with self.subTest(update=update):
                imagery = self._imagery(s1_used=True)
                imagery["t1_tensor"][16] = 1.0
                imagery["meta"]["t1_quality_raw_stats"].update(update)
                with self.assertRaises(api.PipelineError) as caught:
                    api._validate_imagery(imagery, require_sar=True)
                self.assertEqual(caught.exception.code, "invalid_quality_channel")
                self.assertIn(expected, str(caught.exception))

    def test_all_zero_q_reports_invalid_clear_observation_input(self):
        imagery = self._imagery(s1_used=True)
        imagery["t2_tensor"][16] = 0
        with self.assertRaises(api.PipelineError) as caught:
            api._validate_imagery(imagery, require_sar=True)
        self.assertEqual(caught.exception.code, "invalid_quality_channel")
        self.assertIn("all zero", str(caught.exception))

    def test_q_with_only_nodata_level_positive_coverage_is_rejected(self):
        imagery = self._imagery(s1_used=True)
        imagery["t1_tensor"][16] = 0
        imagery["t1_tensor"][16, :16, :16] = 1
        with self.assertRaises(api.PipelineError) as caught:
            api._validate_imagery(imagery, require_sar=True)
        self.assertEqual(caught.exception.code, "invalid_quality_channel")
        self.assertIn("under 5%", str(caught.exception))

    def test_q_nan_infinity_and_out_of_range_are_rejected(self):
        for invalid, expected in ((np.nan, "NaN or infinite"), (np.inf, "NaN or infinite"), (1.01, "outside"), (4.0, "outside"), (-0.01, "outside")):
            with self.subTest(value=invalid):
                imagery = self._imagery(s1_used=True)
                imagery["t1_tensor"][16, 0, 0] = invalid
                with self.assertRaises(api.PipelineError) as caught:
                    api._validate_imagery(imagery, require_sar=True)
                self.assertEqual(caught.exception.code, "invalid_quality_channel")
                self.assertIn(expected, str(caught.exception))

    def test_missing_q_channel_is_rejected(self):
        imagery = self._imagery(s1_used=True)
        imagery["t1_tensor"] = imagery["t1_tensor"][:16]
        with self.assertRaises(api.PipelineError) as caught:
            api._validate_imagery(imagery, require_sar=True)
        self.assertEqual(caught.exception.code, "invalid_input_channels")

    def test_per_channel_diagnostic_has_bounded_statistics_and_provenance(self):
        rows = api._input_channel_diagnostics(self._imagery(s1_used=True))
        self.assertEqual(len(rows), 34)
        q = next(row for row in rows if row["date"] == "2021-01-01" and row["channel"] == "Q")
        self.assertEqual(q["channel_index"], 16)
        self.assertEqual(q["feature_status"], "real")
        self.assertEqual(q["preprocessing_id"], "gee_s2_s1_median_composite_v2")
        self.assertEqual(q["shape"], [128, 128])
        self.assertIn("finite_fraction", q)
        self.assertIn("valid_pixel_fraction", q)

    def test_operational_candidates_are_seeded_and_stay_inside_configured_aoi(self):
        config = {
            "aoi_bbox": {"lon_min": 73.5, "lat_min": 17.5, "lon_max": 74.5, "lat_max": 19.0},
            "region_size_km": 2,
            "t1_dates": ["2021-01-01", "2022-01-01"],
            "t2_dates": ["2024-01-01", "2025-01-01"],
            "window_days": 15,
            "max_cloud": 30,
        }
        first = generate_candidates(config, 30, 1234)
        second = generate_candidates(config, 30, 1234)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        for item in first:
            self.assertGreater(item["date_t2"], item["date_t1"])
            self.assertGreaterEqual(item["bbox"]["lon_min"], config["aoi_bbox"]["lon_min"])
            self.assertLessEqual(item["bbox"]["lon_max"], config["aoi_bbox"]["lon_max"])

    def test_operational_diagnostic_records_provider_unavailability_without_replacement(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            token = "unit-test-admin-token"
            config_path = root / "diagnostic.json"
            config_path.write_text(json.dumps({
                "admin_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "aoi_bbox": {"lon_min": 73.5, "lat_min": 17.5, "lon_max": 74.5, "lat_max": 19.0},
                "region_size_km": 2,
                "t1_dates": ["2021-01-01"], "t2_dates": ["2024-01-01"],
            }), encoding="utf-8")
            with patch.dict(os.environ, {"GEONEXUS_DIAGNOSTIC_ADMIN_TOKEN": token, "GEE_PROJECT_ID": ""}):
                report = run_diagnostic(config_path, root / "output", count=2, seed=77)
            self.assertEqual(report["summary"]["ai_success"], 0)
            self.assertEqual(report["summary"]["ai_failed"], 2)
            self.assertTrue(all(row["status"] == "acquisition_failed" for row in report["runs"]))
            self.assertTrue(all(row["error_code"] == "gee_not_configured" for row in report["runs"]))
            saved = json.loads((root / "output" / "diagnostic_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["random_seed"], 77)
            self.assertEqual(len(saved["runs"]), 2)

    def test_ai_provider_failure_never_calls_stac_or_the_model(self):
        stac = types.ModuleType("sentinel_stac_service")
        stac.fetch_imagery = lambda **kwargs: self.fail("AI must never use STAC")
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: (_ for _ in ()).throw(PermissionError("403 project access denied"))
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: self.fail("model must not run after provider failure")
        with patch.dict(sys.modules, {"gee_service": gee, "sentinel_stac_service": stac, "inference": model}):
            api._jobs["stac-ai"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_job("stac-ai", {**self.payload, "bbox": self.bbox})
        self.assertEqual(api._jobs["stac-ai"]["status"], "error")
        self.assertEqual(api._jobs["stac-ai"]["error_code"], "gee_project_access_denied")
        self.assertEqual(api._jobs["stac-ai"]["meta"]["provider_attempts"][0]["provider"], "Google Earth Engine")
        self.assertEqual(api._jobs["stac-ai"]["meta"]["diagnostic"]["dates"]["t1"], "2021-01-01")

    def test_ai_provider_failure_is_not_reported_as_provider_fallback(self):
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("403 permission denied for project"))
        stac = types.ModuleType("sentinel_stac_service")
        stac.fetch_imagery = lambda **kwargs: self.fail("AI must not fallback to STAC")
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: self.fail("inference must not run")
        with patch.dict(os.environ, {"GEE_PROJECT_ID": "test-project"}), patch.dict(sys.modules, {"gee_service": gee, "sentinel_stac_service": stac, "inference": model}):
            api._jobs["provider-fallback-ai"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_job("provider-fallback-ai", {**self.payload, "bbox": self.bbox})
        job = api._jobs["provider-fallback-ai"]
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["error_code"], "gee_project_access_denied")
        self.assertEqual(job["meta"]["provider"], "Google Earth Engine")
        self.assertNotIn("provider_fallback", job["meta"])
        attempt = job["meta"]["provider_attempts"][0]
        self.assertEqual(attempt["status"], "gee_project_access_denied")
        self.assertEqual(attempt["error_type"], "RuntimeError")
        self.assertIn("cannot access this project", attempt["detail"])
        self.assertNotIn("403", attempt["detail"])
        status = self.client.get("/api/detect/status/provider-fallback-ai").json
        self.assertEqual(status["error_code"], "gee_project_access_denied")
        self.assertEqual(status["meta"]["provider"], "Google Earth Engine")
        self.assertEqual(status["meta"]["provider_attempts"][0]["error_type"], "RuntimeError")

    def test_mocked_exact_maharashtra_request_acquires_gee_then_runs_model(self):
        exact_payload = {
            "bbox": {"lon_min": 72.8434, "lat_min": 19.1163, "lon_max": 72.8625, "lat_max": 19.1343},
            "date_t1": "2021-03-15", "date_t2": "2024-03-15",
            "window_days": 15, "max_cloud": 30,
        }
        imagery = self._imagery(s1_used=True)
        imagery["meta"].update({"t1_date": "2021-03-15", "t2_date": "2024-03-15"})
        calls = []
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: (calls.append(kwargs) or imagery)
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: {
            "model_version": "3.2.0", "threshold_source": "manifest",
            "change_mask_b64": "mask", "binary_mask_b64": "binary",
            "change_prob_b64": "prob", "overlay_b64": "overlay", "dynamics": {},
            "stats": {"change_pct": 1.0, "changed_pixels": 1, "total_pixels": 16384, "output_size": [128, 128], "threshold": 0.28},
        }
        with patch.dict(sys.modules, {"gee_service": gee, "inference": model}), patch.object(storage_manager, "save_prediction", return_value={"id": "exact"}):
            api._jobs["maharashtra-exact"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_job("maharashtra-exact", exact_payload)
        self.assertEqual(api._jobs["maharashtra-exact"]["status"], "done")
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["date_t1"], calls[0]["date_t2"]), ("2021-03-15", "2024-03-15"))
        self.assertEqual(calls[0]["window_days"], 15)
        self.assertEqual(calls[0]["max_cloud"], 30)
        self.assertEqual(calls[0]["lon_min"], 72.8434)
        self.assertEqual(api._jobs["maharashtra-exact"]["result"]["engine"], "geonexus_ai")

    def test_gee_quality_failure_returns_provider_date_and_channel(self):
        imagery = self._imagery(s1_used=True)
        imagery["t1_tensor"][16] = 0.5
        imagery["meta"]["t1_quality_raw_stats"].update({"std": 0.0, "min": 4.0, "max": 4.0})
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: imagery
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: self.fail("invalid Q must stop before inference")
        with patch.dict(os.environ, {"GEE_PROJECT_ID": "test-project"}), patch.dict(sys.modules, {"gee_service": gee, "inference": model}):
            api._jobs["gee-invalid-q"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_job("gee-invalid-q", {**self.payload, "bbox": self.bbox})
        job = api._jobs["gee-invalid-q"]
        self.assertEqual(job["error_code"], "invalid_quality_channel")
        self.assertEqual(job["meta"]["diagnostic"]["provider"], "Google Earth Engine")
        self.assertEqual(job["meta"]["diagnostic"]["date"], "2021-01-01")
        self.assertEqual(job["meta"]["diagnostic"]["channel"], "Q")

    def test_ai_inference_failure_is_terminal_and_structured(self):
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: self._imagery(s1_used=True)
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("checkpoint error with private path"))
        with patch.dict(os.environ, {"GEE_PROJECT_ID": "test-project"}), patch.dict(sys.modules, {"gee_service": gee, "inference": model}):
            api._jobs["inference-error"] = {"status": "pending", "progress": "", "engine": "geonexus_ai", "created_at": 0}
            api._run_job("inference-error", {**self.payload, "bbox": self.bbox})
        job = api._jobs["inference-error"]
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["error_code"], "inference_failed")
        self.assertNotIn("private path", job["error"])

    def test_spectral_failure_does_not_return_an_ai_result(self):
        stac = types.ModuleType("sentinel_stac_service")
        stac.fetch_imagery = lambda **kwargs: self._imagery(s1_used=False)
        spectral = types.ModuleType("spectral_cd")
        spectral.run_spectral_inference = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("spectral failure"))
        with patch.dict(os.environ, {"GEE_PROJECT_ID": ""}), patch.dict(sys.modules, {"sentinel_stac_service": stac, "spectral_cd": spectral}):
            api._jobs["spectral-error"] = {"status": "pending", "progress": "", "engine": "spectral_scva", "created_at": 0}
            api._run_spectral_job("spectral-error", {**self.payload, "bbox": self.bbox})
        job = api._jobs["spectral-error"]
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["error_code"], "spectral_failed")
        self.assertIsNone(job.get("result"))

    def test_provider_fallback_and_spectral_engine_identity_are_reported(self):
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: (_ for _ in ()).throw(PermissionError("permission denied"))
        stac = types.ModuleType("sentinel_stac_service")
        stac.fetch_imagery = lambda **kwargs: self._imagery(s1_used=False)
        spectral = types.ModuleType("spectral_cd")
        spectral.run_spectral_inference = lambda **kwargs: {
            "change_mask_b64": "mask", "binary_mask_b64": "binary", "change_prob_b64": "heat",
            "overlay_b64": "overlay", "dynamics": {},
            "stats": {"change_pct": 4.0, "changed_pixels": 10, "total_pixels": 100, "output_size": [128, 128], "threshold": 0.2},
        }
        ready = {"status": "configured", "code": None, "configured": True, "auth_method": "local_credentials"}
        with patch.dict(os.environ, {"GEE_PROJECT_ID": "test-project"}), patch("app.get_gee_configuration", return_value=ready), patch.dict(sys.modules, {
            "gee_service": gee, "sentinel_stac_service": stac, "spectral_cd": spectral,
        }), patch.object(storage_manager, "save_prediction", return_value={"id": "spec-test"}):
            api._jobs["spectral-fallback"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_spectral_job("spectral-fallback", {**self.payload, "bbox": self.bbox})
        job = api._jobs["spectral-fallback"]
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["engine"], "spectral_scva")
        self.assertEqual(job["result"]["engine_label"], "Spectral Index Analysis (S-CVA)")
        self.assertEqual(job["result"]["meta"]["provider"], "Sentinel-2 STAC")
        self.assertTrue(job["result"]["meta"]["provider_fallback"])
        self.assertEqual(job["result"]["meta"]["provider_attempts"][0]["status"], "gee_project_access_denied")
        self.assertEqual(job["result"]["meta"]["t1_date"], self.payload["date_t1"])

    def test_legacy_database_migrates_as_unknown_and_matches_by_engine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "legacy.db")
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("""CREATE TABLE predictions (
                    id TEXT PRIMARY KEY, location_name TEXT, timestamp TEXT, created_at REAL,
                    bbox_json TEXT, date_t1 TEXT, date_t2 TEXT, stats_json TEXT,
                    dynamics_json TEXT, result_json TEXT)""")
                conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    "old", "Old", "2022", 1.0,
                    '{"lon_min":73,"lat_min":18,"lon_max":73.1,"lat_max":18.1}',
                    "2021-01-01", "2022-01-01", "{}", "{}", "{}"))
                conn.commit()
            finally:
                conn.close()
            with patch.object(storage_manager, "STORAGE_DIR", temp_dir), patch.object(storage_manager, "DB_PATH", db_path):
                storage_manager._init_db()
                rows = storage_manager.get_all_predictions()
                self.assertEqual(rows[0]["engine"], "unknown")
                self.assertEqual(rows[0]["status"], "unknown")
                self.assertIsNone(storage_manager.find_matching_prediction(self.bbox, "2021-01-01", "2022-01-01", engine="geonexus_ai"))
                storage_manager.save_prediction({
                    "id": "spectral", "bbox": self.bbox, "dateT1": "2021-01-01", "dateT2": "2022-01-01",
                    "result": {"success": True, "engine": "spectral_scva", "algorithm": "S-CVA", "meta": {"provider": "Sentinel-2 STAC"}},
                })
                match = storage_manager.find_matching_prediction(self.bbox, "2021-01-01", "2022-01-01", engine="spectral_scva")
                self.assertEqual(match["id"], "spectral")
                self.assertEqual(match["status"], "success")


if __name__ == "__main__":
    unittest.main()
