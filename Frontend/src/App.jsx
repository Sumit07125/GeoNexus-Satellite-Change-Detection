import React, { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import './index.css';
import LocationMap from './components/LocationMap';
import ResultsPanel from './components/ResultsPanel';
import {
  getSavedPredictions,
  fetchServerPredictions,
  savePrediction,
  deletePrediction,
  findMatchingPrediction,
} from './utils/storage';
import { createFilteredOverlay } from './utils/overlayFilter';

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function LoadingOverlay({ progress }) {
  return (
    <div className="loading-overlay">
      <div className="loading-ring" />
      <div>
        <div className="loading-title">Analyzing Satellite Data</div>
        <div className="loading-step" style={{ marginTop: '0.5rem' }}>
          {progress || 'Initializing…'}
        </div>
      </div>
      <div className="loading-progress">
        <div className="loading-progress-bar" />
      </div>
      <div style={{ fontSize: '0.65rem', color: 'var(--c-text3)', fontFamily: 'var(--font-mono)' }}>
        Geo-Nexus v3.2 · MH-DAPT-CD · k=30
      </div>
    </div>
  );
}

export default function App() {
  // ── Navigation State ──────────────────────────────────────────
  // 'map' | 'predictions' | 'saved'
  const [activeView, setActiveView] = useState('map');

  // ── Search & Location State ───────────────────────────────────
  const [bbox, setBbox] = useState(null);
  const [locationName, setLocationName] = useState('');
  const [dateT1, setDateT1] = useState('2021-03-15');
  const [dateT2, setDateT2] = useState('2024-03-15');
  const [windowDays, setWindowDays] = useState(15);
  const [maxCloud, setMaxCloud] = useState(30);

  // ── Manual Coordinate Inputs State ────────────────────────────
  const [coordLat, setCoordLat] = useState('18.9900');
  const [coordLon, setCoordLon] = useState('73.0700');
  const [coordSizeKm, setCoordSizeKm] = useState('2.0');
  const [coordMode, setCoordMode] = useState('center'); // 'center' | 'bounds'
  const [boundLatMin, setBoundLatMin] = useState('');
  const [boundLatMax, setBoundLatMax] = useState('');
  const [boundLonMin, setBoundLonMin] = useState('');
  const [boundLonMax, setBoundLonMax] = useState('');

  // ── Execution & Result State ──────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [liveProgress, setLiveProgress] = useState('');
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [showMapToast, setShowMapToast] = useState(false);

  // ── Saved / Cached Predictions State ──────────────────────────
  const [savedPredictions, setSavedPredictions] = useState(() => getSavedPredictions());
  const [selectedPredictionId, setSelectedPredictionId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // ── Sidebar Resizable Width State ─────────────────────────────
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    try {
      const saved = localStorage.getItem('geonexus_sidebar_w');
      const num = parseInt(saved, 10);
      return num && num >= 280 && num <= 700 ? num : 380;
    } catch {
      return 380;
    }
  });
  const [isResizing, setIsResizing] = useState(false);

  const startResizing = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);
  }, []);

  const handleResetSidebarWidth = useCallback(() => {
    setSidebarWidth(380);
    try {
      localStorage.setItem('geonexus_sidebar_w', '380');
    } catch {}
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
  }, []);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e) => {
      const minW = 280;
      const maxW = Math.min(750, Math.round(window.innerWidth * 0.6));
      const newW = Math.max(minW, Math.min(maxW, e.clientX));
      setSidebarWidth(newW);
      window.dispatchEvent(new Event('resize'));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      setSidebarWidth((curr) => {
        try {
          localStorage.setItem('geonexus_sidebar_w', curr.toString());
        } catch {}
        return curr;
      });
      setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Sync coordinate input fields when bbox is set/drawn on the map
  useEffect(() => {
    if (bbox) {
      const cLat = ((bbox.lat_min + bbox.lat_max) / 2).toFixed(4);
      const cLon = ((bbox.lon_min + bbox.lon_max) / 2).toFixed(4);
      setCoordLat(cLat);
      setCoordLon(cLon);
      setBoundLatMin(bbox.lat_min.toFixed(4));
      setBoundLatMax(bbox.lat_max.toFixed(4));
      setBoundLonMin(bbox.lon_min.toFixed(4));
      setBoundLonMax(bbox.lon_max.toFixed(4));
    }
  }, [bbox]);

  // Check if a saved prediction already exists for the active bbox + dates
  const cachedMatch = findMatchingPrediction(bbox, dateT1, dateT2, savedPredictions);

  // ── Permanent SQLite server sync & health check on mount ──────
  useEffect(() => {
    axios.get(`${API}/api/health`, { timeout: 5000 })
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'));

    // Pull permanent database predictions from server disk
    fetchServerPredictions().then((serverList) => {
      if (serverList && serverList.length > 0) {
        setSavedPredictions(serverList);
        setSelectedPredictionId((prev) => prev || serverList[0].id);
      }
    });
  }, []);

  // ── Apply coordinates manually from inputs ────────────────────
  const handleApplyCenterCoordinates = () => {
    const lat = parseFloat(coordLat);
    const lon = parseFloat(coordLon);
    const size = parseFloat(coordSizeKm) || 2.0;

    if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      setError('Please enter valid numeric latitude (-90 to 90) and longitude (-180 to 180).');
      return;
    }
    setError(null);

    const latDelta = (size / 2) / 111.32;
    const lonDelta = (size / 2) / (111.32 * Math.max(0.1, Math.cos((lat * Math.PI) / 180)));

    const newBbox = {
      lon_min: lon - lonDelta,
      lat_min: lat - latDelta,
      lon_max: lon + lonDelta,
      lat_max: lat + latDelta,
    };

    setBbox(newBbox);
    if (!locationName) {
      setLocationName(`Location (${lat.toFixed(3)}°, ${lon.toFixed(3)}°)`);
    }
  };

  const handleApplyBoundsCoordinates = () => {
    const latMin = parseFloat(boundLatMin);
    const latMax = parseFloat(boundLatMax);
    const lonMin = parseFloat(boundLonMin);
    const lonMax = parseFloat(boundLonMax);

    if (isNaN(latMin) || isNaN(latMax) || isNaN(lonMin) || isNaN(lonMax) || latMin >= latMax || lonMin >= lonMax) {
      setError('Please enter valid bounding box bounds with Min < Max.');
      return;
    }
    setError(null);

    setBbox({
      lon_min: lonMin,
      lat_min: latMin,
      lon_max: lonMax,
      lat_max: latMax,
    });
  };

  // ── Load cached prediction immediately ────────────────────────
  const handleLoadCachedPrediction = (cachedItem) => {
    const match = cachedItem || cachedMatch;
    if (!match) return;

    setBbox(match.bbox);
    setDateT1(match.dateT1);
    setDateT2(match.dateT2);
    if (match.locationName) setLocationName(match.locationName);
    setResult(match.result);
    setSelectedPredictionId(match.id);
    setActiveView('predictions');
    setShowMapToast(false);
  };

  // ── Analyze handler — with automatic cache retrieval ─────────
  const handleAnalyze = useCallback(async () => {
    if (!bbox) { setError('Please select or enter coordinates for an area on the map.'); return; }
    if (!dateT1 || !dateT2) { setError('Please select both T1 and T2 dates.'); return; }
    if (dateT1 >= dateT2) { setError('T1 (Before) date must be earlier than T2 (After) date.'); return; }

    // If an identical prediction was already generated and saved, load instantly!
    if (cachedMatch) {
      handleLoadCachedPrediction(cachedMatch);
      return;
    }

    setError(null);
    setResult(null);
    setSelectedPredictionId('active');
    setLiveProgress('Submitting satellite change detection job…');
    setLoading(true);

    let pollInterval = null;

    try {
      const payload = {
        location_name: locationName || 'Selected Area',
        bbox,
        date_t1: dateT1,
        date_t2: dateT2,
        window_days: windowDays,
        max_cloud: maxCloud,
      };

      // Step 1: Submit job — returns immediately with job_id
      const { data: startData } = await axios.post(
        `${API}/api/detect/start`, payload, { timeout: 15_000 }
      );
      if (!startData.success) throw new Error(startData.error || 'Failed to start job');

      const jobId = startData.job_id;

      // Step 2: Poll /detect/status/:id every 3 seconds
      await new Promise((resolve, reject) => {
        pollInterval = setInterval(async () => {
          try {
            const { data: statusData } = await axios.get(
              `${API}/api/detect/status/${jobId}`, { timeout: 10_000 }
            );

            if (statusData.progress) setLiveProgress(statusData.progress);

            if (statusData.status === 'done') {
              clearInterval(pollInterval);
              resolve(statusData.result);
            } else if (statusData.status === 'error') {
              clearInterval(pollInterval);
              reject(new Error(statusData.error || 'Job failed on server'));
            }
          } catch (pollErr) {
            clearInterval(pollInterval);
            if (pollErr.response?.status === 404 || pollErr.response?.data?.error?.includes('not found')) {
              reject(new Error('Server was restarted — please click Analyze again.'));
            } else {
              reject(pollErr);
            }
          }
        }, 3000);

        setTimeout(() => {
          clearInterval(pollInterval);
          reject(new Error('Analysis timed out after 8 minutes. Try a smaller area.'));
        }, 480_000);
      }).then(async (res) => {
        // Automatically save to permanent server storage & local cache
        const savedRecord = await savePrediction({
          id: res.saved_id,
          locationName: locationName || 'Selected Area',
          bbox,
          dateT1,
          dateT2,
          result: res,
        });

        // Sync with permanent database
        const serverList = await fetchServerPredictions();
        setSavedPredictions(serverList);
        setSelectedPredictionId(savedRecord?.id || res.saved_id || 'active');
        setResult(res);
        setShowMapToast(true);
      });

    } catch (err) {
      if (pollInterval) clearInterval(pollInterval);
      const msg = err.response?.data?.error || err.message || 'Unexpected error';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [bbox, locationName, dateT1, dateT2, windowDays, maxCloud, cachedMatch]);

  // ── Enhanced Spectral Analysis handler (no ML model — pure spectral indices) ──
  const handleSpectralAnalyze = useCallback(async () => {
    if (!bbox) { setError('Please select or enter coordinates for an area on the map.'); return; }
    if (!dateT1 || !dateT2) { setError('Please select both T1 and T2 dates.'); return; }
    if (dateT1 >= dateT2) { setError('T1 (Before) date must be earlier than T2 (After) date.'); return; }

    setError(null);
    setResult(null);
    setSelectedPredictionId('active');
    setLiveProgress('Submitting enhanced spectral analysis job…');
    setLoading(true);

    let pollInterval = null;

    try {
      const payload = {
        location_name: locationName || 'Selected Area',
        bbox,
        date_t1: dateT1,
        date_t2: dateT2,
        window_days: windowDays,
        max_cloud: maxCloud,
      };

      const { data: startData } = await axios.post(
        `${API}/api/detect/spectral/start`, payload, { timeout: 15_000 }
      );
      if (!startData.success) throw new Error(startData.error || 'Failed to start spectral job');

      const jobId = startData.job_id;

      await new Promise((resolve, reject) => {
        pollInterval = setInterval(async () => {
          try {
            const { data: statusData } = await axios.get(
              `${API}/api/detect/status/${jobId}`, { timeout: 10_000 }
            );
            if (statusData.progress) setLiveProgress(statusData.progress);
            if (statusData.status === 'done') {
              clearInterval(pollInterval);
              resolve(statusData.result);
            } else if (statusData.status === 'error') {
              clearInterval(pollInterval);
              reject(new Error(statusData.error || 'Spectral job failed on server'));
            }
          } catch (pollErr) {
            clearInterval(pollInterval);
            reject(pollErr);
          }
        }, 3000);
        setTimeout(() => {
          clearInterval(pollInterval);
          reject(new Error('Analysis timed out after 8 minutes. Try a smaller area.'));
        }, 480_000);
      }).then(async (res) => {
        const savedRecord = await savePrediction({
          id: res.saved_id,
          locationName: locationName || 'Selected Area',
          bbox, dateT1, dateT2,
          result: res,
        });
        const serverList = await fetchServerPredictions();
        setSavedPredictions(serverList);
        setSelectedPredictionId(savedRecord?.id || res.saved_id || 'active');
        setResult(res);
        setShowMapToast(true);
        setActiveView('predictions');
      });
    } catch (err) {
      if (pollInterval) clearInterval(pollInterval);
      const msg = err.response?.data?.error || err.message || 'Unexpected error';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [bbox, locationName, dateT1, dateT2, windowDays, maxCloud]);

  // ── Determine which prediction to display in Predictions Studio ──
  const effectivePredictionId = selectedPredictionId || (result ? 'active' : (savedPredictions[0]?.id || null));

  const activeDisplayedRecord = (() => {
    if (effectivePredictionId && effectivePredictionId !== 'active') {
      return savedPredictions.find((p) => p.id === effectivePredictionId) || null;
    }
    return null;
  })();

  const activeDisplayedResult = (() => {
    if (effectivePredictionId && effectivePredictionId !== 'active') {
      const found = savedPredictions.find((p) => p.id === effectivePredictionId);
      if (found?.result) return found.result;
    }
    return result || (savedPredictions[0]?.result ?? null);
  })();

  // ── Dynamic category color filters (Urban, Vegetation, Soil, Water) ──
  const [overlayFilters, setOverlayFilters] = useState({
    urban: true,
    vegetation: true,
    bare_soil: true,
    water: true,
  });
  const [activeOverlayUrl, setActiveOverlayUrl] = useState(null);

  const handleToggleOverlayFilter = useCallback((key) => {
    setOverlayFilters((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleSetAllOverlayFilters = useCallback((enable) => {
    setOverlayFilters({
      urban: enable,
      vegetation: enable,
      bare_soil: enable,
      water: enable,
    });
  }, []);

  // Synchronize client-side blended overlay for active displayed prediction
  useEffect(() => {
    if (!activeDisplayedResult) {
      setActiveOverlayUrl(null);
      return;
    }
    createFilteredOverlay(
      activeDisplayedResult.t2_rgb,
      activeDisplayedResult.change_mask,
      activeDisplayedResult.overlay,
      overlayFilters
    ).then((url) => {
      setActiveOverlayUrl(url);
    });
  }, [
    activeDisplayedResult?.t2_rgb,
    activeDisplayedResult?.change_mask,
    activeDisplayedResult?.overlay,
    overlayFilters,
  ]);

  const canAnalyze = bbox && dateT1 && dateT2 && !loading;

  return (
    <div className="app-wrapper">
      {/* ── Navbar ─────────────────────────────────────────── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="navbar-logo">GN</div>
          <div>
            <div className="navbar-title">Geo-Nexus</div>
            <div className="navbar-subtitle">MH-DAPT-CD v3.2 · Satellite Change Detection</div>
          </div>
        </div>

        {/* ── Top Navbar Navigation Tabs ───────────────────── */}
        <div className="navbar-nav">
          <button
            className={`nav-tab-btn ${activeView === 'map' ? 'active' : ''}`}
            onClick={() => setActiveView('map')}
            title="Explore satellite map and select area"
          >
            <span>🗺️</span>
            <span>Satellite Map</span>
          </button>

          <button
            className={`nav-tab-btn ${activeView === 'predictions' ? 'active' : ''}`}
            onClick={() => {
              setActiveView('predictions');
              setShowMapToast(false);
            }}
            title="View change detection prediction outcome and dynamics"
          >
            <span>📊</span>
            <span>Predictions & Outcome</span>
            {loading ? (
              <span className="nav-badge-pill running">⏳ Analyzing…</span>
            ) : activeDisplayedResult ? (
              <span className="nav-badge-pill ready">✅ Outcome Ready</span>
            ) : (
              <span className="nav-badge-pill neutral">Ready</span>
            )}
          </button>

          <button
            className={`nav-tab-btn ${activeView === 'saved' ? 'active' : ''}`}
            onClick={() => setActiveView('saved')}
            title="Browse all saved location predictions"
          >
            <span>📁</span>
            <span>Saved Locations</span>
            <span className="nav-badge-pill neutral">{savedPredictions.length}</span>
          </button>
        </div>

        {/* ── API Status ──────────────────────────────────── */}
        <div className="navbar-status">
          <div className="status-dot" style={{
            background: apiStatus === 'ok' ? 'var(--c-green)' : apiStatus === 'error' ? 'var(--c-red)' : 'var(--c-amber)',
          }} />
          {apiStatus === 'ok' ? 'API Online' : apiStatus === 'error' ? 'API Offline' : 'Connecting…'}
        </div>
      </nav>

      {/* ── Main Layout: VIEW 1 - Satellite Map ──────────────── */}
      {activeView === 'map' && (
        <div
          className={`main-content ${sidebarOpen ? '' : 'sidebar-collapsed'} ${isResizing ? 'is-resizing' : ''}`}
          style={{
            gridTemplateColumns: sidebarOpen ? `${sidebarWidth}px 1fr` : '0px 1fr',
          }}
        >
          {/* ── Left Sidebar (Controls & Coordinate Inputs) ──── */}
          <aside
            className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}
            style={{ width: sidebarOpen ? `${sidebarWidth}px` : '0px' }}
          >
            {/* Collapse/Expand Toggle Button */}
            <button
              className="sidebar-toggle-btn"
              onClick={() => {
                setSidebarOpen(!sidebarOpen);
                setTimeout(() => window.dispatchEvent(new Event('resize')), 320);
              }}
              title={sidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
            >
              {sidebarOpen ? '◀' : '▶'}
            </button>

            {/* Interactive Resizer Handle on the right border */}
            {sidebarOpen && (
              <div
                className={`sidebar-resizer ${isResizing ? 'active' : ''}`}
                onMouseDown={startResizing}
                onDoubleClick={handleResetSidebarWidth}
                title="Drag to adjust width · Double-click to reset (380px)"
              >
                <div className="resizer-line" />
                <div className="resizer-grip">
                  <span />
                  <span />
                  <span />
                </div>
                {isResizing && (
                  <div className="resizer-tooltip">
                    {sidebarWidth}px
                  </div>
                )}
              </div>
            )}

            <div className="sidebar-inner">

            {/* ── Location Panel with Coordinate Inputs ──────── */}
            <div className="panel fade-in-up">
              <div className="panel-header">
                <div className="panel-icon blue">📍</div>
                <div className="panel-title">Location & Coordinates</div>
              </div>
              <div className="panel-body">
                <div className="form-group">
                  <label className="form-label">Location Name</label>
                  <input
                    className="form-input"
                    placeholder="e.g. Navi Mumbai Airport, Maharashtra"
                    value={locationName}
                    onChange={(e) => setLocationName(e.target.value)}
                  />
                </div>

                {/* Direct Coordinate Inputs */}
                <div className="coord-section">
                  <div className="coord-header">
                    <span className="coord-title">🎯 Coordinate Entry</span>
                    <button
                      className="coord-mode-link"
                      onClick={() => setCoordMode(coordMode === 'center' ? 'bounds' : 'center')}
                    >
                      {coordMode === 'center' ? 'Switch to 4-Bounds' : 'Switch to Center+Span'}
                    </button>
                  </div>

                  {coordMode === 'center' ? (
                    <>
                      <div className="coord-row">
                        <div className="coord-field">
                          <label className="coord-label">Latitude (°N)</label>
                          <input
                            type="number"
                            step="any"
                            className="coord-input"
                            placeholder="18.9900"
                            value={coordLat}
                            onChange={(e) => setCoordLat(e.target.value)}
                          />
                        </div>
                        <div className="coord-field">
                          <label className="coord-label">Longitude (°E)</label>
                          <input
                            type="number"
                            step="any"
                            className="coord-input"
                            placeholder="73.0700"
                            value={coordLon}
                            onChange={(e) => setCoordLon(e.target.value)}
                          />
                        </div>
                      </div>

                      <div className="coord-field">
                        <label className="coord-label">Box Coverage Size</label>
                        <select
                          className="coord-select"
                          value={coordSizeKm}
                          onChange={(e) => setCoordSizeKm(e.target.value)}
                        >
                          <option value="0.5">500 m × 500 m (High-Detail Local)</option>
                          <option value="1.0">1 km × 1 km (Compact Site)</option>
                          <option value="2.0">2 km × 2 km (High Clarity)</option>
                          <option value="5.0">5 km × 5 km (Optimal Regional)</option>
                          <option value="10.0">10 km × 10 km (District Macro)</option>
                        </select>
                      </div>

                      <button
                        className="coord-apply-btn"
                        onClick={handleApplyCenterCoordinates}
                      >
                        📍 Apply Coordinates to Map
                      </button>
                    </>
                  ) : (
                    <>
                      <div className="coord-row">
                        <div className="coord-field">
                          <label className="coord-label">Lat Min</label>
                          <input
                            type="number"
                            step="any"
                            className="coord-input"
                            placeholder="18.9600"
                            value={boundLatMin}
                            onChange={(e) => setBoundLatMin(e.target.value)}
                          />
                        </div>
                        <div className="coord-field">
                          <label className="coord-label">Lat Max</label>
                          <input
                            type="number"
                            step="any"
                            className="coord-input"
                            placeholder="19.0200"
                            value={boundLatMax}
                            onChange={(e) => setBoundLatMax(e.target.value)}
                          />
                        </div>
                      </div>
                      <div className="coord-row">
                        <div className="coord-field">
                          <label className="coord-label">Lon Min</label>
                          <input
                            type="number"
                            step="any"
                            className="coord-input"
                            placeholder="73.0400"
                            value={boundLonMin}
                            onChange={(e) => setBoundLonMin(e.target.value)}
                          />
                        </div>
                        <div className="coord-field">
                          <label className="coord-label">Lon Max</label>
                          <input
                            type="number"
                            step="any"
                            className="coord-input"
                            placeholder="73.1000"
                            value={boundLonMax}
                            onChange={(e) => setBoundLonMax(e.target.value)}
                          />
                        </div>
                      </div>
                      <button
                        className="coord-apply-btn"
                        onClick={handleApplyBoundsCoordinates}
                      >
                        📐 Set Exact Bounding Box
                      </button>
                    </>
                  )}
                </div>

                {/* Bounding Box Info Badge */}
                {bbox ? (
                  <div className="bbox-display" style={{ marginTop: '10px' }}>
                    <div className="bbox-grid">
                      <div className="bbox-item">
                        <span className="bbox-item-label">Lon Min</span>
                        <span className="bbox-item-val">{bbox.lon_min.toFixed(4)}°</span>
                      </div>
                      <div className="bbox-item">
                        <span className="bbox-item-label">Lat Min</span>
                        <span className="bbox-item-val">{bbox.lat_min.toFixed(4)}°</span>
                      </div>
                      <div className="bbox-item">
                        <span className="bbox-item-label">Lon Max</span>
                        <span className="bbox-item-val">{bbox.lon_max.toFixed(4)}°</span>
                      </div>
                      <div className="bbox-item">
                        <span className="bbox-item-label">Lat Max</span>
                        <span className="bbox-item-val">{bbox.lat_max.toFixed(4)}°</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bbox-hint">
                    <span>🗺</span>
                    Enter coordinates above or draw rectangle on map
                  </div>
                )}

                {/* Cached Prediction Detection Alert */}
                {cachedMatch && (
                  <div className="cached-alert-card">
                    <div className="cached-alert-title">
                      <span>⚡</span>
                      <span>Saved Prediction Available!</span>
                    </div>
                    <div className="cached-alert-desc">
                      This exact location has already been predicted ({cachedMatch.result?.stats?.change_pct}% change). You can load it instantly with 0s wait.
                    </div>
                    <div className="cached-alert-actions">
                      <button
                        className="btn btn-primary"
                        style={{ fontSize: '0.72rem', padding: '0.35rem 0.75rem' }}
                        onClick={() => handleLoadCachedPrediction(cachedMatch)}
                      >
                        ⚡ Load Saved Prediction
                      </button>
                    </div>
                  </div>
                )}

                {bbox && (
                  <button
                    className="btn btn-ghost"
                    onClick={() => setBbox(null)}
                    style={{ fontSize: '0.78rem', padding: '0.45rem 0.85rem' }}
                  >
                    ✕ Clear Selection
                  </button>
                )}
              </div>
            </div>

            {/* ── Date Panel ─────────────────────────────── */}
            <div className="panel fade-in-up fade-delay-1">
              <div className="panel-header">
                <div className="panel-icon teal">📅</div>
                <div className="panel-title">Temporal Range</div>
              </div>
              <div className="panel-body">
                <div className="date-row">
                  <div className="form-group">
                    <label className="form-label">T1 — Before</label>
                    <input
                      type="date"
                      className="form-input"
                      value={dateT1}
                      max={dateT2}
                      onChange={(e) => setDateT1(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">T2 — After</label>
                    <input
                      type="date"
                      className="form-input"
                      value={dateT2}
                      min={dateT1}
                      onChange={(e) => setDateT2(e.target.value)}
                    />
                  </div>
                </div>

                <div style={{ fontSize: '0.7rem', color: 'var(--c-text3)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span>ℹ</span>
                  Searching ±{windowDays} days for cleanest Sentinel-2 scene
                </div>
              </div>
            </div>

            {/* ── Settings Panel ─────────────────────────── */}
            <div className="panel fade-in-up fade-delay-2">
              <div className="panel-header">
                <div className="panel-icon amber">⚙️</div>
                <div className="panel-title">Settings & Model</div>
              </div>
              <div className="panel-body">
                <div className="date-row">
                  <div className="form-group">
                    <label className="form-label">Search Window</label>
                    <input
                      type="number"
                      className="form-input"
                      value={windowDays}
                      min={1} max={45}
                      onChange={(e) => setWindowDays(Number(e.target.value))}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Max Cloud %</label>
                    <input
                      type="number"
                      className="form-input"
                      value={maxCloud}
                      min={0} max={80}
                      onChange={(e) => setMaxCloud(Number(e.target.value))}
                    />
                  </div>
                </div>

                <div style={{ padding: '0.6rem', background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.2)', borderRadius: 'var(--radius)', fontSize: '0.7rem', color: 'var(--c-text3)', lineHeight: 1.6 }}>
                  <strong style={{ color: 'var(--c-purple)', fontFamily: 'var(--font-mono)' }}>Model:</strong> GeoNexus v3.2 · Few-Shot k=30<br />
                  <strong style={{ color: 'var(--c-purple)', fontFamily: 'var(--font-mono)' }}>Threshold:</strong> 0.28 (MH-VAL Calibrated)
                </div>
              </div>
            </div>

            {/* ── Error Message ──────────────────────────── */}
            {error && (
              <div className="error-box fade-in-up" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.6rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="error-icon">⚠</span>
                  <span>{error}</span>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
                  <button
                    className="btn btn-primary"
                    style={{ flex: 1, fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
                    onClick={handleAnalyze}
                    disabled={!bbox || !dateT1 || !dateT2}
                  >
                    🔄 Try Again
                  </button>
                  <button
                    className="btn btn-ghost"
                    style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
                    onClick={() => setError(null)}
                  >
                    ✕ Dismiss
                  </button>
                </div>
              </div>
            )}

            {/* ── Main Analyze Button ────────────────────── */}
            <button
              className="btn btn-primary btn-full btn-lg fade-in-up fade-delay-3"
              disabled={!canAnalyze}
              onClick={handleAnalyze}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analyzing…
                </>
              ) : cachedMatch ? (
                <>
                  ⚡ Load Saved Prediction (Instant)
                </>
              ) : (
                <>
                  🔍 Analyze Change Detection
                </>
              )}
            </button>

            {/* ── Enhanced (Spectral) Analysis Button ── */}
            <button
              className="btn btn-spectral btn-full fade-in-up fade-delay-3"
              disabled={!canAnalyze}
              onClick={handleSpectralAnalyze}
              title="Compute change using multi-spectral index differencing (NDVI · NDWI · NDBI · BSI ensemble)"
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Computing…
                </>
              ) : (
                <>
                  <span style={{ fontSize: '0.9em' }}>⚲</span>
                  Enhanced Analysis
                </>
              )}
            </button>

            {/* ── Quick Presets ─────────────────────────── */}
            <div className="panel fade-in-up fade-delay-4" style={{ opacity: 0.85 }}>
              <div className="panel-header">
                <div className="panel-icon green">⚡</div>
                <div className="panel-title">Maharashtra Hotspots</div>
              </div>
              <div className="panel-body" style={{ gap: '0.5rem' }}>
                {[
                  { name: 'Navi Mumbai Airport', bbox: { lon_min: 73.04, lat_min: 18.96, lon_max: 73.10, lat_max: 19.02 }, loc: 'Navi Mumbai International Airport' },
                  { name: 'Pune (Zone A)', bbox: { lon_min: 73.70, lat_min: 18.30, lon_max: 74.20, lat_max: 18.80 }, loc: 'Pune, Maharashtra' },
                  { name: 'Satara (Zone B)', bbox: { lon_min: 73.50, lat_min: 17.50, lon_max: 74.00, lat_max: 18.00 }, loc: 'Satara, Maharashtra' },
                  { name: 'Vidarbha (Zone C)', bbox: { lon_min: 78.90, lat_min: 20.90, lon_max: 79.30, lat_max: 21.30 }, loc: 'Vidarbha (Nagpur), Maharashtra' },
                ].map((preset) => (
                  <button
                    key={preset.name}
                    className="btn btn-ghost"
                    style={{ fontSize: '0.78rem', padding: '0.45rem 0.85rem', justifyContent: 'flex-start' }}
                    onClick={() => {
                      setBbox(preset.bbox);
                      setLocationName(preset.loc);
                    }}
                  >
                    📍 {preset.name}
                  </button>
                ))}
              </div>
            </div>

          </div>
        </aside>

          {/* ── Right Display Area: Satellite Map ── */}
          <div className="right-panel" style={{ height: 'calc(100vh - 64px)' }}>
            <div className="map-wrapper" style={{ height: '100%', minHeight: '100%' }}>
              <LocationMap bbox={bbox} setBbox={setBbox} />
              {loading && <LoadingOverlay progress={liveProgress} />}

              {/* Floating notification when prediction completes or is running */}
              {showMapToast && activeDisplayedResult && !loading && (
                <div className="map-toast-pill">
                  <span>🎉</span>
                  <span style={{ fontSize: '0.8rem', color: '#f1f5f9' }}>
                    Prediction complete for <strong>{locationName || 'Selected Area'}</strong>!
                  </span>
                  <button
                    className="map-toast-btn"
                    onClick={() => {
                      setActiveView('predictions');
                      setShowMapToast(false);
                    }}
                  >
                    View Predicted Outcome →
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: Dedicated Prediction & Outcome Studio (100% Full-Screen Width) */}
      {activeView === 'predictions' && (
        <div className="main-content full-page-view">
          <div className="prediction-studio">
            {/* Top Toolbar with Location & Prediction Switcher */}
            <div className="prediction-top-bar">
              <div className="prediction-selector-group">
                <span className="prediction-select-label">📍 Selected Prediction:</span>
                <select
                  className="prediction-dropdown"
                  value={effectivePredictionId || ''}
                  onChange={(e) => {
                    const val = e.target.value;
                    setSelectedPredictionId(val);
                    if (val !== 'active') {
                      const match = savedPredictions.find((p) => p.id === val);
                      if (match) {
                        setBbox(match.bbox);
                        setDateT1(match.dateT1);
                        setDateT2(match.dateT2);
                        if (match.locationName) setLocationName(match.locationName);
                      }
                    }
                  }}
                >
                  {result && (
                    <option value="active">
                      ⭐ Current Active Scan · {locationName || 'Selected Area'} ({dateT1} → {dateT2})
                    </option>
                  )}
                  {savedPredictions.map((saved) => (
                    <option key={saved.id} value={saved.id}>
                      📍 {saved.locationName} ({saved.dateT1} → {saved.dateT2}) — {saved.result?.stats?.change_pct}% Change
                    </option>
                  ))}
                  {!result && savedPredictions.length === 0 && (
                    <option value="">No predictions available yet</option>
                  )}
                </select>
              </div>

              <div className="prediction-top-actions">
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: '0.78rem', padding: '0.45rem 0.85rem' }}
                  onClick={() => setActiveView('map')}
                >
                  🗺️ Back to Map
                </button>
                {activeDisplayedRecord && (
                  <button
                    className="btn btn-ghost"
                    style={{ fontSize: '0.78rem', padding: '0.45rem 0.85rem', color: '#f87171' }}
                    onClick={async () => {
                      const updated = await deletePrediction(activeDisplayedRecord.id);
                      setSavedPredictions(updated);
                      setSelectedPredictionId(updated.length > 0 ? updated[0].id : (result ? 'active' : null));
                    }}
                  >
                    🗑 Remove
                  </button>
                )}
                <button
                  className="btn btn-primary"
                  style={{ fontSize: '0.78rem', padding: '0.45rem 0.85rem' }}
                  onClick={handleAnalyze}
                  disabled={loading || !bbox}
                >
                  🔄 Re-run Scan
                </button>
              </div>
            </div>

            {/* Prediction Studio Content */}
            <div className="prediction-content-wrapper">
              {loading ? (
                <div className="live-running-card">
                  <div className="radar-scan-box">
                    <span className="radar-satellite-icon">🛰️</span>
                  </div>
                  <div>
                    <h3 style={{ color: '#60a5fa', marginBottom: '8px' }}>
                      Detection Job In Progress
                    </h3>
                    <div style={{ color: 'var(--c-text2)', fontSize: '0.9rem', fontFamily: 'var(--font-mono)' }}>
                      {liveProgress || 'Processing multi-band satellite data…'}
                    </div>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--c-text3)' }}>
                    Location: <strong>{locationName || 'Selected Area'}</strong> | Period: {dateT1} to {dateT2}
                  </div>
                </div>
              ) : activeDisplayedResult ? (
                <ResultsPanel
                  data={activeDisplayedResult}
                  activeFilters={overlayFilters}
                  onToggleFilter={handleToggleOverlayFilter}
                  onSetAllFilters={handleSetAllOverlayFilters}
                  onOverlayChange={setActiveOverlayUrl}
                />
              ) : (
                <div className="prediction-empty-card">
                  <span style={{ fontSize: '48px' }}>🛰️</span>
                  <h3>No Predictions Selected</h3>
                  <p style={{ color: 'var(--c-text3)', fontSize: '0.85rem' }}>
                    Select an area on the satellite map or enter coordinates in the sidebar to run change detection.
                  </p>
                  <button className="btn btn-primary" onClick={() => setActiveView('map')}>
                    🗺️ Open Satellite Map
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* VIEW 3: Saved Locations Gallery (100% Full-Screen Width) */}
      {activeView === 'saved' && (
        <div className="main-content full-page-view">
          <div className="saved-gallery-wrapper">
            <div className="saved-gallery-header">
              <div>
                <h2>Saved Change Detections</h2>
                <div style={{ fontSize: '0.8rem', color: 'var(--c-text3)' }}>
                  All previously analyzed locations are saved locally. Click any location to view its predicted outcome with 0s reload time.
                </div>
              </div>
              <button
                className="btn btn-ghost"
                onClick={() => setActiveView('map')}
              >
                🗺️ Go to Map
              </button>
            </div>

            {savedPredictions.length === 0 ? (
              <div className="prediction-empty-card">
                <span style={{ fontSize: '48px' }}>📁</span>
                <h3>No Saved Locations Yet</h3>
                <p style={{ color: 'var(--c-text3)', fontSize: '0.85rem' }}>
                  When you analyze any location, it will automatically be saved here so you never need to repeat the same prediction twice.
                </p>
                <button className="btn btn-primary" onClick={() => setActiveView('map')}>
                  🗺️ Analyze Your First Area
                </button>
              </div>
            ) : (
              <div className="saved-gallery-grid">
                {savedPredictions.map((saved) => {
                  const stats = saved.result?.stats;
                  const changePct = stats?.change_pct ?? 0;
                  return (
                    <div key={saved.id} className="saved-item-card">
                      <div className="saved-item-top">
                        <div>
                          <div className="saved-item-name">{saved.locationName}</div>
                          <div className="saved-item-date">
                            {saved.dateT1} → {saved.dateT2} · Saved {new Date(saved.timestamp).toLocaleDateString()}
                          </div>
                        </div>
                        <span
                          style={{
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            padding: '3px 8px',
                            borderRadius: '99px',
                            background: changePct > 20 ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.2)',
                            color: changePct > 20 ? '#f87171' : '#4ade80',
                            border: `1px solid ${changePct > 20 ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.4)'}`,
                          }}
                        >
                          {changePct}% Change
                        </span>
                      </div>

                      <div className="saved-item-metrics">
                        <div className="saved-metric-block">
                          <span className="saved-metric-val" style={{ color: '#60a5fa' }}>
                            {stats?.changed_area_ha ?? '—'} ha
                          </span>
                          <span className="saved-metric-lbl">Changed Area</span>
                        </div>
                        <div className="saved-metric-block">
                          <span className="saved-metric-val" style={{ color: '#cbd5e1' }}>
                            {saved.bbox ? `${saved.bbox.lat_min.toFixed(2)}°, ${saved.bbox.lon_min.toFixed(2)}°` : '—'}
                          </span>
                          <span className="saved-metric-lbl">Coordinates</span>
                        </div>
                      </div>

                      <div className="saved-item-actions">
                        <button
                          className="btn btn-primary"
                          style={{ flex: 1, fontSize: '0.75rem', padding: '0.45rem' }}
                          onClick={() => {
                            setBbox(saved.bbox);
                            setDateT1(saved.dateT1);
                            setDateT2(saved.dateT2);
                            setLocationName(saved.locationName);
                            setResult(saved.result);
                            setSelectedPredictionId(saved.id);
                            setActiveView('predictions');
                          }}
                        >
                          📊 View Outcome
                        </button>
                        <button
                          className="btn btn-ghost"
                          style={{ fontSize: '0.75rem', padding: '0.45rem 0.8rem' }}
                          onClick={() => {
                            setBbox(saved.bbox);
                            setLocationName(saved.locationName);
                            setActiveView('map');
                          }}
                          title="Show area on map"
                        >
                          🗺️
                        </button>
                        <button
                          className="btn btn-ghost"
                          style={{ fontSize: '0.75rem', padding: '0.45rem 0.8rem', color: '#f87171' }}
                          onClick={async () => {
                            const updated = await deletePrediction(saved.id);
                            setSavedPredictions(updated);
                            if (selectedPredictionId === saved.id) {
                              setSelectedPredictionId(null);
                            }
                          }}
                          title="Delete permanently from saved database"
                        >
                          🗑
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
