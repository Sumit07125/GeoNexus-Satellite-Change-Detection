import React, { useState } from 'react';
import { MapContainer, TileLayer, Rectangle, useMapEvents, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

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
  const latSpanDeg = Math.abs(b.lat_max - b.lat_min);
  const lonSpanDeg = Math.abs(b.lon_max - b.lon_min);
  const midLatRad = (((b.lat_min + b.lat_max) / 2) * Math.PI) / 180;

  const heightKm = latSpanDeg * 111.32;
  const widthKm = lonSpanDeg * 111.32 * Math.cos(midLatRad);
  const areaKm2 = heightKm * widthKm;

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
  };
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
        const b = {
          lon_min: Math.min(anchor.lng, e.latlng.lng),
          lat_min: Math.min(anchor.lat, e.latlng.lat),
          lon_max: Math.max(anchor.lng, e.latlng.lng),
          lat_max: Math.max(anchor.lat, e.latlng.lat),
        };
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
        const b = {
          lon_min: Math.min(anchor.lng, e.latlng.lng),
          lat_min: Math.min(anchor.lat, e.latlng.lat),
          lon_max: Math.max(anchor.lng, e.latlng.lng),
          lat_max: Math.max(anchor.lat, e.latlng.lat),
        };
        if (onHoverDims) onHoverDims(getBBoxDimensions(b));
      }
    },
  });

  // Live rubber-band rectangle while drawing
  const liveRect = drawing && anchor && hover
    ? [[Math.min(anchor.lat, hover.lat), Math.min(anchor.lng, hover.lng)],
       [Math.max(anchor.lat, hover.lat), Math.max(anchor.lng, hover.lng)]]
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
    const latDelta = (sizeKm / 2) / 111.32;
    const lonDelta = (sizeKm / 2) / (111.32 * Math.max(0.1, Math.cos((center.lat * Math.PI) / 180)));
    setBbox({
      lon_min: center.lng - lonDelta,
      lat_min: center.lat - latDelta,
      lon_max: center.lng + lonDelta,
      lat_max: center.lat + latDelta,
    });
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
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15, duration: 0.8 });
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
