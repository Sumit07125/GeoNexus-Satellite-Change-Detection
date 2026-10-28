"""Focused tests for engine identity, input contract, API validation, and storage migration."""
import os
import base64
import io
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as api
import storage_manager
import inference
from gee_service import _encode_model_tensor


class ReliabilityContractTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()
        self.bbox = {"lon_min": 73.0, "lat_min": 18.0, "lon_max": 73.1, "lat_max": 18.1}
        self.payload = {"bbox": self.bbox, "date_t1": "2021-01-01", "date_t2": "2022-01-01"}

    def test_health_reports_distinct_engines(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["engines"], ["geonexus_ai", "spectral_scva"])

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
        with patch("app.threading.Thread"):
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
        tensor[16] = np.indices((128, 128))[0] / 127
        imagery = {"t1_tensor": tensor, "t2_tensor": tensor.copy(), "meta": {"s1_used": False}}
        with self.assertRaises(api.PipelineError) as caught:
            api._validate_imagery(imagery, require_sar=True)
        self.assertEqual(caught.exception.code, "missing_sar_channels")
        api._validate_imagery(imagery, require_sar=False)

    def _imagery(self, s1_used=False):
        tensor = np.full((17, 128, 128), 1200, dtype=np.float32)
        tensor[13:16] = np.array([-900, -1700, -800], dtype=np.float32)[:, None, None]
        tensor[16] = np.indices((128, 128))[0] / 127
        return {
            "t1_tensor": tensor,
            "t2_tensor": tensor.copy(),
            "t1_rgb_b64": "t1-rgb",
            "t2_rgb_b64": "t2-rgb",
            "meta": {"s1_used": s1_used, "t1_date": "2021-01-01", "t2_date": "2022-01-01"},
        }

    def test_ai_job_rejects_stac_without_calling_the_model(self):
        stac = types.ModuleType("sentinel_stac_service")
        stac.fetch_imagery = lambda **kwargs: self._imagery(s1_used=False)
        model = types.ModuleType("inference")
        model.run_inference = lambda **kwargs: self.fail("AI must not run with placeholder SAR")
        with patch.dict(os.environ, {"GEE_PROJECT_ID": ""}), patch.dict(sys.modules, {"sentinel_stac_service": stac, "inference": model}):
            api._jobs["stac-ai"] = {"status": "pending", "progress": "", "created_at": 0}
            api._run_job("stac-ai", {**self.payload, "bbox": self.bbox})
        self.assertEqual(api._jobs["stac-ai"]["status"], "error")
        self.assertEqual(api._jobs["stac-ai"]["error_code"], "missing_sar_channels")

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
        with patch.dict(os.environ, {"GEE_PROJECT_ID": "test-project"}), patch.dict(sys.modules, {
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
        self.assertEqual(job["result"]["meta"]["provider_attempts"][0]["status"], "authentication_failed")

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
