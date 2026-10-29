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
import gee_service
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
        self.assertRegex(response.json["build_id"], r"^[A-Za-z0-9._-]+$")

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

    def test_shared_orbit_is_discovered_deterministically_for_both_dates(self):
        t1 = {136: 8, 37: 5, 51: 5}
        t2 = {136: 2, 37: 7, 51: 7}
        selected, common, policy = gee_service.choose_shared_orbit(t1, t2)
        self.assertEqual(common, [37, 51, 136])
        self.assertEqual(selected, 37)  # ties in pair-minimum and total counts resolve to the lower orbit ID
        self.assertIn("lowest orbit ID", policy)
        self.assertEqual(gee_service.choose_shared_orbit(t1, t2)[0], gee_service.choose_shared_orbit(t1, t2)[0])

    def _mock_s1_candidate_query(self, *, base_count=4, eligible_count=2, histogram=None, fail_at=None, failure=None):
        class Info:
            def __init__(self, value=None, error=None):
                self.value, self.error = value, error
            def getInfo(self):
                if self.error:
                    raise self.error
                return self.value

        class Collection:
            def __init__(self, filters=0):
                self.filter_count = filters
            def filterBounds(self, _bbox): return Collection(self.filter_count)
            def filterDate(self, _start, _end): return Collection(self.filter_count)
            def filter(self, _filter): return Collection(self.filter_count + 1)
            def size(self):
                operation = "base_count" if self.filter_count == 0 else "filtered_count"
                value = base_count if operation == "base_count" else eligible_count
                return Info(value, failure if fail_at == operation else None)
            def aggregate_histogram(self, _property):
                return Info(histogram, failure if fail_at == "histogram" else None)

        collection = Collection()
        with patch.object(gee_service.ee, "ImageCollection", return_value=collection), \
             patch.object(gee_service.ee.Filter, "listContains", side_effect=lambda *args, **kwargs: ("contains", args, kwargs)), \
             patch.object(gee_service.ee.Filter, "eq", side_effect=lambda *args, **kwargs: ("equals", args, kwargs)):
            return gee_service._get_s1_orbit_candidates(
                object(), "2021-03-15", 15, "T1",
                gee_service._diagnostic_base("2021-03-15", "2024-03-15", 15, 30, (72.84, 19.11, 72.86, 19.13)),
            )

    def test_s1_base_count_failure_has_exact_stage_and_operation(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
            self._mock_s1_candidate_query(fail_at="base_count", failure=RuntimeError("temporarily unavailable"))
        self.assertEqual(caught.exception.code, "gee_api_unavailable")
        self.assertEqual(caught.exception.details["stage"], "s1_t1_base_count")
        self.assertEqual(caught.exception.details["operation"], "base.size().getInfo()")

    def test_s1_filtered_count_failure_retains_base_count(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
            self._mock_s1_candidate_query(fail_at="filtered_count", failure=RuntimeError("evaluation failed"))
        self.assertEqual(caught.exception.details["stage"], "s1_t1_filtered_count")
        self.assertEqual(caught.exception.details["t1_s1_count_before_filters"], 4)
        self.assertEqual(caught.exception.details["operation"], "eligible.size().getInfo()")

    def test_s1_histogram_failure_retains_both_collection_counts(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
            self._mock_s1_candidate_query(fail_at="histogram", failure=RuntimeError("histogram unavailable"))
        self.assertEqual(caught.exception.details["stage"], "s1_t1_orbit_histogram")
        self.assertEqual(caught.exception.details["t1_s1_count_before_filters"], 4)
        self.assertEqual(caught.exception.details["t1_s1_count_after_iw_desc_vv_vh_filters"], 2)
        self.assertIn("aggregate_histogram", caught.exception.details["operation"])

    def test_s1_histogram_metadata_parse_failure_is_distinct_from_query_failure(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
            self._mock_s1_candidate_query(histogram={"bad-orbit": 2})
        self.assertEqual(caught.exception.code, "gee_s1_orbit_metadata_invalid")
        self.assertEqual(caught.exception.details["stage"], "s1_t1_orbit_metadata_parse")
        self.assertEqual(caught.exception.details["s1_orbit_histogram"], {"bad-orbit": 2})

    def test_s1_histogram_accepts_integer_valued_numeric_orbit_ids(self):
        for raw_orbit in ("34.0", 34.0, "34", 34):
            with self.subTest(raw_orbit=raw_orbit):
                _, orbit_counts, _ = self._mock_s1_candidate_query(
                    eligible_count=3, histogram={raw_orbit: 3}
                )
                self.assertEqual(orbit_counts, {34: 3})
        _, orbit_counts, _ = self._mock_s1_candidate_query(
            eligible_count=3, histogram={34.0: 3.0}
        )
        self.assertEqual(orbit_counts, {34: 3})

    def test_s1_histogram_rejects_invalid_orbit_ids(self):
        invalid_orbits = (
            "34.5", "NaN", "inf", "-inf", True, None, "not-a-number", 0, -1, 176,
        )
        for raw_orbit in invalid_orbits:
            with self.subTest(raw_orbit=raw_orbit), self.assertRaises(gee_service.GEEAcquisitionError) as caught:
                self._mock_s1_candidate_query(eligible_count=1, histogram={raw_orbit: 1})
            self.assertEqual(caught.exception.code, "gee_s1_orbit_metadata_invalid")
            self.assertEqual(caught.exception.details["stage"], "s1_t1_orbit_metadata_parse")

    def test_s1_histogram_rejects_invalid_scene_counts(self):
        invalid_counts = (0, -1, 1.5, "NaN", "inf", float("nan"), float("inf"), True, None)
        for raw_count in invalid_counts:
            with self.subTest(raw_count=raw_count), self.assertRaises(gee_service.GEEAcquisitionError) as caught:
                self._mock_s1_candidate_query(eligible_count=1, histogram={"34.0": raw_count})
            self.assertEqual(caught.exception.code, "gee_s1_orbit_metadata_invalid")
            self.assertEqual(caught.exception.details["stage"], "s1_t1_orbit_metadata_parse")

    def test_s1_histogram_rejects_duplicate_normalized_orbits_and_count_mismatch(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as duplicate:
            self._mock_s1_candidate_query(eligible_count=2, histogram={"34": 1, "34.0": 1})
        self.assertEqual(duplicate.exception.code, "gee_s1_orbit_metadata_invalid")
        self.assertIn("duplicate normalized", duplicate.exception.details["technical_reason"])

        with self.assertRaises(gee_service.GEEAcquisitionError) as mismatch:
            self._mock_s1_candidate_query(eligible_count=3, histogram={"34.0": 2})
        self.assertEqual(mismatch.exception.code, "gee_s1_orbit_metadata_invalid")
        self.assertIn("did not match", mismatch.exception.details["technical_reason"])

    def test_valid_s1_histogram_proceeds_to_shared_orbit_selection(self):
        _, t1_orbits, _ = self._mock_s1_candidate_query(eligible_count=3, histogram={"34.0": 3})
        _, t2_orbits, _ = self._mock_s1_candidate_query(
            eligible_count=4, histogram={34.0: 4}
        )
        selected, common, _ = gee_service.choose_shared_orbit(t1_orbits, t2_orbits)
        self.assertEqual(t1_orbits, {34: 3})
        self.assertEqual(t2_orbits, {34: 4})
        self.assertEqual((selected, common), (34, [34]))

    def test_parser_failure_is_not_mislabeled_as_no_common_orbit(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
            self._mock_s1_candidate_query(eligible_count=3, histogram={"34.5": 3})
        self.assertEqual(caught.exception.code, "gee_s1_orbit_metadata_invalid")
        self.assertNotEqual(caught.exception.code, "gee_no_common_s1_orbit")

    def test_empty_s1_collection_and_empty_histogram_have_distinct_meanings(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as empty_collection:
            self._mock_s1_candidate_query(base_count=0, eligible_count=0, histogram=None)
        self.assertEqual(empty_collection.exception.code, "gee_s1_no_scenes_t1")
        self.assertEqual(empty_collection.exception.details["t1_s1_count_before_filters"], 0)
        with self.assertRaises(gee_service.GEEAcquisitionError) as no_eligible:
            self._mock_s1_candidate_query(base_count=4, eligible_count=0, histogram=None)
        self.assertEqual(no_eligible.exception.code, "gee_s1_no_scenes_t1")
        self.assertEqual(no_eligible.exception.details["t1_s1_count_after_iw_desc_vv_vh_filters"], 0)
        with self.assertRaises(gee_service.GEEAcquisitionError) as empty_histogram:
            self._mock_s1_candidate_query(base_count=4, eligible_count=2, histogram={})
        self.assertEqual(empty_histogram.exception.code, "gee_s1_orbit_metadata_invalid")
        self.assertEqual(empty_histogram.exception.details["stage"], "s1_t1_orbit_metadata_parse")

    def test_s1_t1_and_t2_query_failure_stages_are_not_confused(self):
        for label, target in (("T1", "2021-03-15"), ("T2", "2024-03-15")):
            with self.subTest(label=label):
                with patch.object(gee_service.ee, "ImageCollection", side_effect=RuntimeError("permission denied")):
                    with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
                        gee_service._get_s1_orbit_candidates(object(), target, 15, label, {})
                self.assertEqual(caught.exception.code, "gee_project_access_denied")
                self.assertEqual(caught.exception.details["stage"], f"s1_{label.lower()}_collection_build")

    def test_s1_authentication_failure_is_not_misreported_as_missing_scenes(self):
        with patch.object(gee_service.ee, "ImageCollection", side_effect=RuntimeError("invalid_grant authorization=supersecret")):
            with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
                gee_service._get_s1_orbit_candidates(object(), "2021-03-15", 15, "T1", {})
        self.assertEqual(caught.exception.code, "gee_authentication_failed")
        self.assertEqual(caught.exception.details["stage"], "s1_t1_collection_build")
        self.assertNotIn("supersecret", caught.exception.details["technical_reason"])

    def test_s1_project_access_failure_does_not_expose_project_id(self):
        with patch.object(gee_service.ee, "ImageCollection", side_effect=RuntimeError("403 permission denied for projects/private-cloud-project")):
            with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
                gee_service._get_s1_orbit_candidates(object(), "2021-03-15", 15, "T1", {})
        self.assertEqual(caught.exception.code, "gee_project_access_denied")
        self.assertNotIn("private-cloud-project", caught.exception.details["technical_reason"])

    def test_successful_s1_orbit_queries_share_one_orbit_for_both_composites(self):
        t1_counts = {136: 3, 37: 4}
        t2_counts = {37: 2, 165: 6}
        orbit, common, _ = gee_service.choose_shared_orbit(t1_counts, t2_counts)
        self.assertEqual((orbit, common), (37, [37]))
        with patch.object(gee_service, "_get_s1_composite", side_effect=["T1", "T2"]) as build:
            result = gee_service._get_s1_pair_composites("t1 candidates", "t2 candidates", orbit, {"stage": "s1_common_orbit_selection"})
        self.assertEqual(result, ("T1", "T2"))
        self.assertEqual([call.args[1] for call in build.call_args_list], [37, 37])

    def test_orbit_seen_in_only_one_window_is_rejected_with_candidate_counts(self):
        t1 = {136: 6}
        t2 = {165: 4}
        with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
            gee_service._select_shared_orbit(t1, t2, None, {"t1_s1_orbits": t1, "t2_s1_orbits": t2})
        self.assertEqual(caught.exception.code, "gee_no_common_s1_orbit")
        self.assertEqual(caught.exception.details["s1_common_orbits"], [])
        self.assertIn("t1_s1_orbits", caught.exception.details)

    def test_frozen_research_orbit_unavailable_is_not_mislabeled_as_no_common_orbit(self):
        with self.assertRaises(gee_service.GEEAcquisitionError) as caught:
            gee_service._select_shared_orbit({136: 2, 37: 4}, {37: 5}, 136, {"research_zone": "zone_a_pune"})
        self.assertEqual(caught.exception.code, "gee_frozen_s1_orbit_unavailable")
        self.assertEqual(caught.exception.details["s1_common_orbits"], [37])
        self.assertEqual(caught.exception.details["frozen_relative_orbit"], 136)

    def test_arbitrary_mumbai_box_uses_discovered_orbits_and_exact_research_zone_keeps_lock(self):
        mumbai = (72.8432, 19.1113, 72.8527, 19.1203)
        self.assertIsNone(gee_service._matches_research_zone(mumbai))
        selected, _, _ = gee_service.choose_shared_orbit({37: 3}, {37: 2})
        self.assertEqual(selected, 37)
        self.assertEqual(gee_service._matches_research_zone((73.70, 18.30, 74.20, 18.80)), ("zone_a_pune", 136))
        locked, _, reason = gee_service.choose_shared_orbit({136: 9, 37: 20}, {136: 4, 37: 20}, frozen_orbit=136)
        self.assertEqual(locked, 136)
        self.assertEqual(reason, "frozen research-zone orbit")

    def test_stage_specific_s1_s2_and_cloud_score_failures_preserve_windows(self):
        diagnostic = {"search_windows": {"t1": {"start": "2021-02-28", "end_inclusive": "2021-03-30"}}}
        cases = [
            (lambda: gee_service._require_s1_scenes("T1", 0, 0, {}, diagnostic), "gee_s1_no_scenes_t1"),
            (lambda: gee_service._require_s2_scenes("T2", 0, 0, diagnostic), "gee_s2_no_scenes_t2"),
            (lambda: gee_service._require_cloud_score_matches("T1", 0, diagnostic), "gee_cloud_score_no_coverage"),
        ]
        for operation, expected_code in cases:
            with self.subTest(expected_code=expected_code), self.assertRaises(gee_service.GEEAcquisitionError) as caught:
                operation()
            self.assertEqual(caught.exception.code, expected_code)
            self.assertIn("search_windows", caught.exception.details)

    def test_small_ai_bbox_is_rejected_before_thread_or_satellite_acquisition(self):
        small = {"bbox": {"lon_min": 72.8432, "lat_min": 19.1113, "lon_max": 72.8527, "lat_max": 19.1203}, "date_t1": "2021-03-15", "date_t2": "2024-03-15"}
        with patch("app.threading.Thread") as thread:
            response = self.client.post("/api/detect/start", json=small)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json["error_code"], "area_too_small")
        self.assertLess(response.json["meta"]["estimated_bbox_dimensions_m"]["width"], 1280)
        thread.assert_not_called()

    def test_two_km_mumbai_bbox_is_validated_and_the_same_bbox_is_queued(self):
        from math import cos, radians
        center_lat, center_lon, size_km = 19.1158, 72.84795, 2.0
        half_lat = (size_km / 2) / 111.13
        half_lon = (size_km / 2) / (111.32 * cos(radians(center_lat)))
        bbox = {"lon_min": center_lon-half_lon, "lat_min": center_lat-half_lat, "lon_max": center_lon+half_lon, "lat_max": center_lat+half_lat}
        body = {"bbox": bbox, "date_t1": "2021-03-15", "date_t2": "2024-03-15"}
        readiness = {"status": "configured", "configured": True, "auth_method": "local_credentials"}
        with patch("app.get_gee_configuration", return_value=readiness), patch("app.threading.Thread") as thread:
            response = self.client.post("/api/detect/start", json=body)
        self.assertEqual(response.status_code, 200)
        queued_payload = thread.call_args.kwargs["args"][1]
        self.assertEqual(queued_payload["bbox"], bbox)
        dimensions = api._bbox_dimensions_m(queued_payload["bbox"])
        self.assertAlmostEqual(dimensions["width"] / 1000, 2, delta=0.02)
        self.assertAlmostEqual(dimensions["height"] / 1000, 2, delta=0.02)

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
        self.assertEqual(api._jobs["stac-ai"]["meta"]["diagnostic"]["requested_dates"]["t1"], "2021-01-01")

    def test_s1_histogram_parse_failure_never_invokes_model_inference(self):
        error = gee_service.GEEAcquisitionError(
            "gee_s1_orbit_metadata_invalid",
            "Earth Engine returned malformed Sentinel-1 orbit metadata.",
            {
                "provider": "Google Earth Engine",
                "stage": "s1_t1_orbit_metadata_parse",
                "operation": "parse relativeOrbitNumber_start histogram",
                "t1_s1_orbits": None,
                "s1_orbit_histogram": {"34.0": 3},
                "exception_type": "ValueError",
                "technical_reason": "invalid orbit metadata",
            },
        )
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: (_ for _ in ()).throw(error)
        stac = types.ModuleType("sentinel_stac_service")
        stac.fetch_imagery = lambda **kwargs: self.fail("AI must not use STAC after parser failure")
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: self.fail("model must not run after parser failure")
        with patch.dict(sys.modules, {
            "gee_service": gee,
            "sentinel_stac_service": stac,
            "inference": model,
        }):
            api._jobs["s1-parser-failure"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_job("s1-parser-failure", {**self.payload, "bbox": self.bbox})
        status = self.client.get("/api/detect/status/s1-parser-failure").json
        self.assertEqual(status["error_code"], "gee_s1_orbit_metadata_invalid")
        self.assertEqual(status["meta"]["diagnostic"]["stage"], "s1_t1_orbit_metadata_parse")

    def test_s1_orbit_query_diagnostics_survive_job_polling(self):
        error = gee_service.GEEAcquisitionError(
            "gee_api_unavailable", "Earth Engine could not query Sentinel-1 orbits for T1.",
            {
                "provider": "Google Earth Engine", "stage": "s1_t1_orbit_histogram",
                "operation": 'eligible.aggregate_histogram("relativeOrbitNumber_start").getInfo()',
                "requested_dates": {"t1": "2021-03-15", "t2": "2024-03-15"},
                "search_windows": {"t1": {"start": "2021-02-28", "end_inclusive": "2021-03-30"}, "t2": {"start": "2024-02-29", "end_inclusive": "2024-03-30"}},
                "t1_s1_count_before_filters": 9, "t1_s1_count_after_iw_desc_vv_vh_filters": 4,
                "exception_type": "EEException", "technical_reason": "query failed safely",
            },
        )
        gee = types.ModuleType("gee_service")
        gee.fetch_imagery = lambda **kwargs: (_ for _ in ()).throw(error)
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: self.fail("model must not run after query failure")
        readiness = {"status": "configured", "configured": True, "auth_method": "local_credentials"}
        with patch("app.get_gee_configuration", return_value=readiness), patch.dict(sys.modules, {"gee_service": gee, "inference": model}):
            api._jobs["s1-stage-debug"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_job("s1-stage-debug", {**self.payload, "bbox": self.bbox})
        status = self.client.get("/api/detect/status/s1-stage-debug").json
        self.assertEqual(status["request_id"], "s1-stage-debug")
        self.assertEqual(status["error_code"], "gee_api_unavailable")
        diagnostic = status["meta"]["diagnostic"]
        self.assertEqual(diagnostic["stage"], "s1_t1_orbit_histogram")
        self.assertIn("aggregate_histogram", diagnostic["operation"])
        self.assertEqual(diagnostic["t1_s1_count_before_filters"], 9)
        self.assertEqual(diagnostic["t1_s1_count_after_iw_desc_vv_vh_filters"], 4)
        self.assertEqual(diagnostic["request_id"], "s1-stage-debug")

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
