/**
 * storage.js — Permanent SQLite Server Sync + Fast Local Fallback for Geo-Nexus predictions
 * Ensures predictions persist permanently across server restarts, reboots, and browser sessions.
 */
import axios from 'axios';

const STORAGE_KEY = 'geonexus_saved_predictions_v3';
const API = import.meta.env.VITE_API_URL || 'http://localhost:5000';
const MAX_LOCAL_CACHE = 10;

/**
 * Reads local cached predictions synchronously for instantaneous render on mount
 */
export function getSavedPredictions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.warn('[Storage] Failed to read local cache:', e);
    return [];
  }
}

/**
 * Saves to local storage cache with safe quota handling
 */
function setLocalCache(items) {
  try {
    const trimmed = items.slice(0, MAX_LOCAL_CACHE);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch (e) {
    try {
      const reduced = items.slice(0, 3);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(reduced));
    } catch {
      // Ignore quota overflow on localStorage
    }
  }
}

/**
 * Fetches all saved predictions directly from the permanent server SQLite database.
 * Survives server restarts, reboots, and browser session clears!
 */
export async function fetchServerPredictions() {
  try {
    const { data } = await axios.get(`${API}/api/predictions`, { timeout: 8000 });
    if (data && data.success && Array.isArray(data.predictions)) {
      setLocalCache(data.predictions);
      return data.predictions;
    }
  } catch (err) {
    console.warn('[Storage] Server fetch failed, falling back to local cache:', err.message);
  }
  return getSavedPredictions();
}

/**
 * Saves a prediction both permanently to the server SQLite database and local cache.
 */
export async function savePrediction(prediction) {
  if (!prediction || !prediction.result) return null;

  const id = prediction.id || `pred_${Date.now()}`;
  const timestamp = prediction.timestamp || new Date().toISOString();

  const record = {
    ...prediction,
    id,
    timestamp,
  };

  // 1. Update local cache immediately
  try {
    const local = getSavedPredictions();
    const filtered = local.filter((item) => {
      if (item.id === id) return false;
      if (isBBoxAndDateMatch(item.bbox, item.dateT1, item.dateT2, record.bbox, record.dateT1, record.dateT2)) {
        return false;
      }
      return true;
    });
    filtered.unshift(record);
    setLocalCache(filtered);
  } catch (err) {
    console.warn('[Storage] Local cache update note:', err);
  }

  // 2. Persist permanently to server SQLite database
  try {
    await axios.post(`${API}/api/predictions`, record, { timeout: 10000 });
  } catch (serverErr) {
    console.warn('[Storage] Server save note (will rely on auto-save or retry):', serverErr.message);
  }

  return record;
}

/**
 * Deletes a prediction both from server SQLite database and local cache.
 */
export async function deletePrediction(id) {
  // 1. Update local cache immediately
  let updated = [];
  try {
    const local = getSavedPredictions();
    updated = local.filter((p) => p.id !== id);
    setLocalCache(updated);
  } catch (e) {
    console.error('[Storage] Error updating local cache:', e);
  }

  // 2. Delete permanently from server SQLite disk
  try {
    await axios.delete(`${API}/api/predictions/${id}`, { timeout: 8000 });
  } catch (serverErr) {
    console.warn('[Storage] Server delete note:', serverErr.message);
  }

  return updated;
}

/**
 * Geographic bounding box and date range matching tolerance (~800m)
 */
export function isBBoxAndDateMatch(b1, t1_a, t2_a, b2, t1_b, t2_b, tol = 0.008) {
  if (!b1 || !b2) return false;
  if (t1_a !== t1_b || t2_a !== t2_b) return false;

  return (
    Math.abs(b1.lon_min - b2.lon_min) <= tol &&
    Math.abs(b1.lat_min - b2.lat_min) <= tol &&
    Math.abs(b1.lon_max - b2.lon_max) <= tol &&
    Math.abs(b1.lat_max - b2.lat_max) <= tol
  );
}

/**
 * Finds if an active bounding box and date pair matches an existing saved prediction.
 */
export function findMatchingPrediction(bbox, dateT1, dateT2, predictionsList = null) {
  if (!bbox || !dateT1 || !dateT2) return null;
  const list = predictionsList || getSavedPredictions();
  return (
    list.find((item) =>
      isBBoxAndDateMatch(item.bbox, item.dateT1, item.dateT2, bbox, dateT1, dateT2)
    ) || null
  );
}
