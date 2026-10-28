"""
storage_manager.py — Permanent SQLite database storage for Geo-Nexus predictions
Ensures all analyzed locations and change detection results persist across
server restarts, reboots, and browser sessions.
"""

import os
import time
import json
import uuid
import sqlite3
import datetime
import threading

STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
DB_PATH = os.path.join(STORAGE_DIR, "geonexus_predictions.db")
_db_lock = threading.Lock()


def _get_connection():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                location_name TEXT,
                timestamp TEXT,
                created_at REAL,
                bbox_json TEXT,
                date_t1 TEXT,
                date_t2 TEXT,
                stats_json TEXT,
                dynamics_json TEXT,
                result_json TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dates ON predictions(date_t1, date_t2)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at DESC)")
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(predictions)").fetchall()}
        for name in ("engine", "algorithm", "provider", "status"):
            if name not in columns:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {name} TEXT NOT NULL DEFAULT 'unknown'")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_engine_dates ON predictions(engine, date_t1, date_t2)")
        conn.commit()
        conn.close()


# Auto-initialize database on module import
_init_db()


def save_prediction(record: dict) -> dict:
    """
    Saves or updates a prediction record permanently in the SQLite database.
    """
    if not record:
        return {}

    pred_id = record.get("id") or f"pred_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    loc_name = record.get("locationName") or record.get("location_name") or "Selected Area"
    timestamp = record.get("timestamp") or datetime.datetime.now().isoformat()
    created_at = float(record.get("created_at", time.time()))

    bbox = record.get("bbox") or {}
    date_t1 = record.get("dateT1") or record.get("date_t1") or ""
    date_t2 = record.get("dateT2") or record.get("date_t2") or ""

    result = record.get("result") or {}
    stats = result.get("stats") or record.get("stats") or {}
    dynamics = result.get("dynamics") or record.get("dynamics") or {}
    engine = result.get("engine") or record.get("engine") or "unknown"
    algorithm = result.get("algorithm") or record.get("algorithm") or "unknown"
    provider = (result.get("meta") or {}).get("provider") or record.get("provider") or "unknown"
    status = record.get("status") or result.get("status") or ("success" if result.get("success") else "unknown")

    bbox_json = json.dumps(bbox)
    stats_json = json.dumps(stats)
    dynamics_json = json.dumps(dynamics)
    result_json = json.dumps(result)

    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO predictions (
                id, location_name, timestamp, created_at,
                bbox_json, date_t1, date_t2,
                stats_json, dynamics_json, result_json, engine, algorithm, provider, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred_id, loc_name, timestamp, created_at,
            bbox_json, date_t1, date_t2,
            stats_json, dynamics_json, result_json, engine, algorithm, provider, status
        ))
        conn.commit()
        conn.close()

    return {
        "id": pred_id,
        "locationName": loc_name,
        "timestamp": timestamp,
        "createdAt": created_at,
        "bbox": bbox,
        "dateT1": date_t1,
        "dateT2": date_t2,
        "stats": stats,
        "dynamics": dynamics,
        "result": result,
        "engine": engine,
        "algorithm": algorithm,
        "provider": provider,
        "status": status,
    }


def get_all_predictions(include_full_result: bool = True) -> list[dict]:
    """
    Returns all saved predictions from disk, newest first.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, location_name, timestamp, created_at,
                   bbox_json, date_t1, date_t2,
                   stats_json, dynamics_json, result_json, engine, algorithm, provider, status
            FROM predictions
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()

    results = []
    for r in rows:
        try:
            bbox = json.loads(r["bbox_json"]) if r["bbox_json"] else {}
        except Exception:
            bbox = {}

        try:
            stats = json.loads(r["stats_json"]) if r["stats_json"] else {}
        except Exception:
            stats = {}

        try:
            dynamics = json.loads(r["dynamics_json"]) if r["dynamics_json"] else {}
        except Exception:
            dynamics = {}

        item = {
            "id": r["id"],
            "locationName": r["location_name"] or "Selected Area",
            "timestamp": r["timestamp"],
            "createdAt": r["created_at"],
            "bbox": bbox,
            "dateT1": r["date_t1"],
            "dateT2": r["date_t2"],
            "stats": stats,
            "dynamics": dynamics,
            "engine": r["engine"] or "unknown",
            "algorithm": r["algorithm"] or "unknown",
            "provider": r["provider"] or "unknown",
            "status": r["status"] or "unknown",
        }

        if include_full_result:
            try:
                full_res = json.loads(r["result_json"]) if r["result_json"] else {}
                item["result"] = full_res
            except Exception:
                item["result"] = {}

        results.append(item)

    return results


def get_prediction_by_id(pred_id: str) -> dict | None:
    """
    Returns full prediction details for a specific ID.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, location_name, timestamp, created_at,
                   bbox_json, date_t1, date_t2,
                   stats_json, dynamics_json, result_json, engine, algorithm, provider, status
            FROM predictions
            WHERE id = ?
        """, (pred_id,))
        row = cursor.fetchone()
        conn.close()

    if not row:
        return None

    try:
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else {}
    except Exception:
        bbox = {}
    try:
        stats = json.loads(row["stats_json"]) if row["stats_json"] else {}
    except Exception:
        stats = {}
    try:
        dynamics = json.loads(row["dynamics_json"]) if row["dynamics_json"] else {}
    except Exception:
        dynamics = {}
    try:
        result = json.loads(row["result_json"]) if row["result_json"] else {}
    except Exception:
        result = {}

    return {
        "id": row["id"],
        "locationName": row["location_name"] or "Selected Area",
        "timestamp": row["timestamp"],
        "createdAt": row["created_at"],
        "bbox": bbox,
        "dateT1": row["date_t1"],
        "dateT2": row["date_t2"],
        "stats": stats,
        "dynamics": dynamics,
        "result": result,
        "engine": row["engine"] or "unknown",
        "algorithm": row["algorithm"] or "unknown",
        "provider": row["provider"] or "unknown",
        "status": row["status"] or "unknown",
    }


def delete_prediction(pred_id: str) -> bool:
    """
    Deletes a prediction record from permanent SQLite storage.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
    return deleted


def is_bbox_match(b1: dict, b2: dict, tol: float = 0.008) -> bool:
    """
    Checks if two bounding boxes are within the geographic tolerance (~800m).
    """
    if not b1 or not b2:
        return False
    try:
        return (
            abs(float(b1["lon_min"]) - float(b2["lon_min"])) <= tol and
            abs(float(b1["lat_min"]) - float(b2["lat_min"])) <= tol and
            abs(float(b1["lon_max"]) - float(b2["lon_max"])) <= tol and
            abs(float(b1["lat_max"]) - float(b2["lat_max"])) <= tol
        )
    except (KeyError, ValueError, TypeError):
        return False


def find_matching_prediction(bbox: dict, date_t1: str, date_t2: str, tol: float = 0.008, engine: str = "geonexus_ai") -> dict | None:
    """
    Searches the permanent database for a matching prior prediction with identical
    dates and matching bbox bounds within tolerance.
    """
    if not bbox or not date_t1 or not date_t2:
        return None

    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, location_name, timestamp, created_at,
                   bbox_json, date_t1, date_t2,
                   stats_json, dynamics_json, result_json, engine, algorithm, provider, status
            FROM predictions
            WHERE date_t1 = ? AND date_t2 = ?
            ORDER BY created_at DESC
        """, (date_t1, date_t2))
        rows = cursor.fetchall()
        conn.close()

    for r in rows:
        try:
            b = json.loads(r["bbox_json"]) if r["bbox_json"] else {}
            if (r["engine"] or "unknown") == engine and is_bbox_match(b, bbox, tol=tol):
                try:
                    res = json.loads(r["result_json"]) if r["result_json"] else {}
                except Exception:
                    res = {}
                return {
                    "id": r["id"],
                    "locationName": r["location_name"] or "Selected Area",
                    "timestamp": r["timestamp"],
                    "createdAt": r["created_at"],
                    "bbox": b,
                    "dateT1": r["date_t1"],
                    "dateT2": r["date_t2"],
                    "stats": json.loads(r["stats_json"]) if r["stats_json"] else {},
                    "dynamics": json.loads(r["dynamics_json"]) if r["dynamics_json"] else {},
                    "result": res,
                    "engine": r["engine"] or "unknown",
                    "algorithm": r["algorithm"] or "unknown",
                    "provider": r["provider"] or "unknown",
                    "status": r["status"] or "unknown",
                }
        except Exception:
            continue

    return None
