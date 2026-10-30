import React, { useState } from 'react';
import { MapContainer, TileLayer, Rectangle, useMapEvents, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { bboxDimensionsKm, bboxDimensionsMeters, createSquareBbox } from '../utils/geospatial';

// Minimum draw size in metres — prevents accidental 1-click / thin-stripe draws
const MIN_DRAW_SIDE_M = 200;

// Fix Leaflet default icon paths broken by Vite
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const MAP_CENTER = [18.55, 73.95]; // Maharashtra, India
const MAP_ZOOM = 8;

/* Calculate ground dimensions in kilometers and meters */
function getBBoxDimensions(b) {
  if (!b) return null;
  const dimensions = bboxDimensionsKm(b);
  const { widthKm, heightKm, areaKm2 } = dimensions;

  const formatDist = (km) => {
    if (km < 1.0) {
      return `${Math.round(km * 1000)} m`;
    }
    return `${Math.round(km * 10) / 10} km`;
  };

  const formatArea = (km2) => {
    if (km2 < 0.1) {
      return `${Math.round(km2 * 1000000).toLocaleString()} m²`;
    }
    return `${Math.round(km2 * 100) / 100} km²`;
  };

  return {
    widthKm: Math.round(widthKm * 100) / 100,
    heightKm: Math.round(heightKm * 100) / 100,
    areaKm2: Math.round(areaKm2 * 100) / 100,
    formattedWidth: formatDist(widthKm),
    formattedHeight: formatDist(heightKm),
    formattedArea: formatArea(areaKm2),
    isSub2Km: widthKm < 2.0 || heightKm < 2.0,
    isSubModelMinimum: dimensions.isSubModelMinimum,
  };
}

// ── Snap a raw lat/lon bbox to the larger of its two sides (square),
//    and ensure it is at least MIN_DRAW_SIDE_M metres on each side.
function snapToSquare(rawB) {
  const centerLat = (rawB.lat_min + rawB.lat_max) / 2;
  const centerLon = (rawB.lon_min + rawB.lon_max) / 2;
  const dims = bboxDimensionsMeters(rawB);
  if (!dims) return rawB;
  // Pick the larger side so we never shrink the user's intended extent
  const sideM = Math.max(dims.width, dims.height, MIN_DRAW_SIDE_M);
  return createSquareBbox(centerLat, centerLon, sideM / 1000);
}

// ── Draw-by-click layer with live kilometer measurements ───────
function ClickDrawLayer({ bbox, setBbox, drawing, setDrawing, onHoverDims }) {
  const [anchor, setAnchor] = useState(null);
  const [hover, setHover] = useState(null);

  useMapEvents({
    click(e) {
      if (!drawing) return;
      if (!anchor) {
        setAnchor(e.latlng);
      } else {
        const rawB = {
          lon_min: Math.min(anchor.lng, e.latlng.lng),
          lat_min: Math.min(anchor.lat, e.latlng.lat),
          lon_max: Math.max(anchor.lng, e.latlng.lng),
          lat_max: Math.max(anchor.lat, e.latlng.lat),
        };
        // Snap to square so accidental thin stripes become a proper square
        const b = snapToSquare(rawB);
        setBbox(b);
        setAnchor(null);
        setHover(null);
        setDrawing(false);
        if (onHoverDims) onHoverDims(null);
      }
    },
    mousemove(e) {
      if (drawing && anchor) {
        setHover(e.latlng);
        const rawB = {
          lon_min: Math.min(anchor.lng, e.latlng.lng),
          lat_min: Math.min(anchor.lat, e.latlng.lat),
          lon_max: Math.max(anchor.lng, e.latlng.lng),
          lat_max: Math.max(anchor.lat, e.latlng.lat),
        };
        // Show snapped-square preview dims so user sees what they'll get
        const snapped = snapToSquare(rawB);
        if (onHoverDims) onHoverDims(getBBoxDimensions(snapped));
      }
    },
  });

  // Live rubber-band rectangle while drawing — show snapped square preview
  const liveSnapped = (drawing && anchor && hover)
    ? snapToSquare({
        lon_min: Math.min(anchor.lng, hover.lng),
        lat_min: Math.min(anchor.lat, hover.lat),
        lon_max: Math.max(anchor.lng, hover.lng),
        lat_max: Math.max(anchor.lat, hover.lat),
      })
    : null;
  const liveRect = liveSnapped
    ? [[liveSnapped.lat_min, liveSnapped.lon_min], [liveSnapped.lat_max, liveSnapped.lon_max]]
    : null;

  return (
    <>
      {liveRect && (
        <Rectangle
          bounds={liveRect}
          pathOptions={{ color: '#60a5fa', weight: 2, dashArray: '6 4', fillOpacity: 0.1 }}
        />
      )}
      {bbox && (
        <Rectangle
          bounds={[
            [bbox.lat_min, bbox.lon_min],
            [bbox.lat_max, bbox.lon_max],
          ]}
          pathOptions={{ color: '#3b82f6', weight: 2.5, fillColor: '#3b82f6', fillOpacity: 0.15 }}
        />
      )}
    </>
  );
}

// ── Map controls inside Leaflet context ────────────────────────
function MapControls({ setBbox, drawing, setDrawing, bbox, hoverDims }) {
  const map = useMap();

  React.useEffect(() => {
    map.getContainer().style.cursor = drawing ? 'crosshair' : '';
  }, [drawing, map]);

  const setPresetCenterBox = (sizeKm) => {
    const center = map.getCenter();
    setBbox(createSquareBbox(center.lat, center.lng, sizeKm));
    setDrawing(false);
  };

  const currentDims = hoverDims || getBBoxDimensions(bbox);

  return (
    <div className="map-top-controls">
      {/* ── Top-right Toolset ── */}
      <div className="map-toolbar">

        <button
          className={`map-tool-btn ${drawing ? 'drawing' : ''}`}
          onClick={() => {
            if (drawing) {
              setDrawing(false);
            } else {
              setBbox(null);
              setDrawing(true);
            }
          }}
        >
          {drawing ? '✕ Cancel' : '▬ Draw Custom Area'}
        </button>

        <div className="map-preset-group">
          <span className="map-preset-label">Quick Area:</span>
          <button className="map-preset-btn" onClick={() => setPresetCenterBox(0.5)} title="Focused 500m × 500m high-detail local site">
            500 m
          </button>
          <button className="map-preset-btn" onClick={() => setPresetCenterBox(1.0)} title="Compact 1 km × 1 km local site">
            1 km
          </button>
          <button className="map-preset-btn" onClick={() => setPresetCenterBox(1.4)} title="AI-ready 1.4 km × 1.4 km minimum area">
            1.4 km · AI Ready
          </button>
          <button className="map-preset-btn" onClick={() => setPresetCenterBox(2.0)} title="High-clarity 2 km × 2 km area">
            2 km
          </button>
          <button className="map-preset-btn" onClick={() => setPresetCenterBox(5.0)} title="Optimal 5 km × 5 km regional area">
            5 km
          </button>
          <button className="map-preset-btn" onClick={() => setPresetCenterBox(10.0)} title="10 km × 10 km district area">
            10 km
          </button>
        </div>

        {bbox && (
          <button className="map-clear-btn" onClick={() => setBbox(null)}>
            🗑 Clear
          </button>
        )}
      </div>

      {/* ── Status Bar / Live Dimension Badge ── */}
      <div className="map-badge">
        {drawing ? (
          <>
            <span style={{ color: '#60a5fa' }}>🖱</span>
            {hoverDims ? (
              <span>
                Drawing: <strong>{hoverDims.formattedWidth} × {hoverDims.formattedHeight}</strong> ({hoverDims.formattedArea}) — click to confirm
              </span>
            ) : (
              <span>Click 1st corner to anchor, move mouse, then click 2nd corner</span>
            )}
          </>
        ) : bbox && currentDims ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: '#3b82f6' }}>✓</span>
            <span>
              Area: <strong>{currentDims.formattedWidth} × {currentDims.formattedHeight}</strong> ({currentDims.formattedArea})
            </span>
            {currentDims.isSub2Km ? (
              <span style={{ color: '#38bdf8', fontSize: '0.68rem', fontWeight: 600 }}>
                (🔍 High-Detail Zoom — 10m Ground Resolution)
              </span>
            ) : (
              <span style={{ color: '#34d399', fontSize: '0.68rem', fontWeight: 600 }}>
                (✅ Optimal Sentinel-2 Area)
              </span>
            )}
          </div>
        ) : (
          <>
            <span>🛰</span>
            <span>Click &ldquo;Draw Custom Area&rdquo; or pick a Quick Area preset (top-right)</span>
          </>
        )}
      </div>
    </div>
  );
}

function MapFlyTo({ bbox }) {
  const map = useMap();
  const prevKeyRef = React.useRef(null);

  React.useEffect(() => {
    if (!bbox) return;
    const key = `${bbox.lat_min.toFixed(4)}_${bbox.lat_max.toFixed(4)}_${bbox.lon_min.toFixed(4)}_${bbox.lon_max.toFixed(4)}`;
    if (prevKeyRef.current !== key) {
      prevKeyRef.current = key;
      const bounds = [
        [bbox.lat_min, bbox.lon_min],
        [bbox.lat_max, bbox.lon_max],
      ];
      try {
        const currentBounds = map.getBounds();
        if (!currentBounds.contains(bounds)) {
          map.fitBounds(bounds, { padding: [40, 40], maxZoom: 13, duration: 0.6 });
        }
      } catch {
        // silent catch if unmounted
      }
    }
  }, [bbox, map]);

  return null;
}

function MapResizeHandler() {
  const map = useMap();
  React.useEffect(() => {
    const handleResize = () => {
      try {
        map.invalidateSize();
      } catch {}
    };
    handleResize();
    const t1 = setTimeout(handleResize, 100);
    const t2 = setTimeout(handleResize, 350);
    window.addEventListener('resize', handleResize);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      window.removeEventListener('resize', handleResize);
    };
  }, [map]);
  return null;
}

export default function LocationMap({ bbox, setBbox }) {
  const [drawing, setDrawing] = useState(false);
  const [hoverDims, setHoverDims] = useState(null);

  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }}>
      <MapContainer
        center={MAP_CENTER}
        zoom={MAP_ZOOM}
        style={{ height: '100%', width: '100%' }}
        zoomControl
      >
        {/* Invalidate size on tab switch or resize */}
        <MapResizeHandler />

        {/* Auto fit to bbox when entered */}
        <MapFlyTo bbox={bbox} />

        {/* ESRI World Imagery (satellite) */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="Tiles &copy; Esri &mdash; Earthstar Geographics"
          maxZoom={18}
        />
        {/* Place labels */}
        <TileLayer
          url="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          attribution=""
          maxZoom={18}
          opacity={0.75}
        />

        <ClickDrawLayer
          bbox={bbox}
          setBbox={setBbox}
          drawing={drawing}
          setDrawing={setDrawing}
          onHoverDims={setHoverDims}
        />

        <MapControls
          bbox={bbox}
          setBbox={setBbox}
          drawing={drawing}
          setDrawing={setDrawing}
          hoverDims={hoverDims}
        />
      </MapContainer>
    </div>
  );
}
