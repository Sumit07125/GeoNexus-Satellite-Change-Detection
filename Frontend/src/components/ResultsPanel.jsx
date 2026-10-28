import React, { useState, useEffect, useMemo } from 'react';
import { createFilteredOverlay } from '../utils/overlayFilter';

/* ─────────────────────────────────────────────────────────────
   Surface dynamics color palette & metadata
───────────────────────────────────────────────────────────── */
const DYNAMICS_META = {
  urban: {
    key: 'urban',
    label: 'Urban & Structural',
    short: 'Urban',
    sub: 'Built-up expansion, infrastructure change',
    color: '#eab308',
    icon: '🏗️',
  },
  vegetation: {
    key: 'vegetation',
    label: 'Vegetation Dynamics',
    short: 'Vegetation',
    sub: 'Canopy shift, crop bloom, vegetation loss',
    color: '#22c55e',
    icon: '🌿',
  },
  bare_soil: {
    key: 'bare_soil',
    label: 'Soil & Agriculture',
    short: 'Soil/Agri',
    sub: 'Fallow land, harvest, exposed ground',
    color: '#f97316',
    icon: '🌾',
  },
  water: {
    key: 'water',
    label: 'Water Dynamics',
    short: 'Water',
    sub: 'River expansion, reservoirs, flood inundation',
    color: '#2563eb',
    icon: '💧',
  },
};

/* ── Single image panel card ───────────────────────────────── */
function PanelCard({ title, subtitle, icon, accent, src, date, badge, onClick, extraControl }) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div
      className="cd-panel"
      style={{ '--panel-accent': accent }}
      onClick={() => onClick(src, title, date)}
      title="Click to enlarge"
    >
      <div className="cd-panel-bar" />

      <div className="cd-panel-label">
        <span className="cd-panel-icon">{icon}</span>
        <div className="cd-panel-text">
          <span className="cd-panel-title">{title}</span>
          <span className="cd-panel-sub">{subtitle}</span>
        </div>
        {date && <span className="cd-panel-date">{date}</span>}
        {badge && <span className="cd-panel-badge">{badge}</span>}
      </div>

      {extraControl && (
        <div className="cd-panel-controls" onClick={(e) => e.stopPropagation()}>
          {extraControl}
        </div>
      )}

      <div className="cd-panel-img-wrap">
        {!loaded && <div className="cd-panel-skeleton" />}
        <img
          src={src}
          alt={title}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          style={{ opacity: loaded ? 1 : 0, transition: 'opacity 0.25s ease' }}
        />
        <div className="cd-panel-expand-hint">⤢ Enlarge</div>
      </div>
    </div>
  );
}

/* ── Interactive Before / After Split Slider ───────────────── */
function ComparisonSlider({ t1Src, t2Src, overlaySrc, t1Date, t2Date }) {
  const [sliderPos, setSliderPos] = useState(50);
  const [compareMode, setCompareMode] = useState('t1_t2'); // 't1_t2' | 't2_overlay'

  const leftSrc = compareMode === 't1_t2' ? t1Src : t2Src;
  const rightSrc = compareMode === 't1_t2' ? t2Src : overlaySrc;
  const leftLabel = compareMode === 't1_t2' ? `Before T1 (${t1Date || 'Pre'})` : `Optical T2 (${t2Date || 'Post'})`;
  const rightLabel = compareMode === 't1_t2' ? `Current T2 (${t2Date || 'Post'})` : 'T2 + Filtered Overlay';

  return (
    <div className="comparison-container">
      <div className="comparison-toolbar">
        <div className="comparison-mode-btns">
          <button
            className={`btn-toggle ${compareMode === 't1_t2' ? 'active' : ''}`}
            onClick={() => setCompareMode('t1_t2')}
          >
            T1 (Before) ↔ T2 (Current)
          </button>
          <button
            className={`btn-toggle ${compareMode === 't2_overlay' ? 'active' : ''}`}
            onClick={() => setCompareMode('t2_overlay')}
          >
            T2 (Current) ↔ Overlay Fusion
          </button>
        </div>
        <div className="comparison-hint">
          ↔ Drag slider to wipe & inspect changes
        </div>
      </div>

      <div className="comparison-viewport">
        {/* Underneath: Right Image */}
        <img src={rightSrc} alt={rightLabel} className="comparison-img" />
        <div className="comparison-tag right">{rightLabel}</div>

        {/* Clipped Over: Left Image */}
        <div
          className="comparison-clip"
          style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
        >
          <img src={leftSrc} alt={leftLabel} className="comparison-img" />
          <div className="comparison-tag left">{leftLabel}</div>
        </div>

        {/* Slider Divider Handle */}
        <div
          className="comparison-handle"
          style={{ left: `${sliderPos}%` }}
        >
          <div className="handle-line" />
          <div className="handle-circle">
            <span>‹</span>
            <span>›</span>
          </div>
        </div>

        {/* Invisible Range Input on top */}
        <input
          type="range"
          min="0"
          max="100"
          value={sliderPos}
          onChange={(e) => setSliderPos(Number(e.target.value))}
          className="comparison-range"
          aria-label="Comparison slider"
        />
      </div>
    </div>
  );
}

/* ── Surface Dynamics Classification Section with interactive toggles ── */
function DynamicsSection({ dynamics, totalHa, changePct, activeFilters, onToggleFilter }) {
  if (!dynamics || Object.keys(dynamics).length === 0) return null;

  // Render order: Urban & Structural first, then Vegetation, Soil, Water
  const orderedKeys = ['urban', 'vegetation', 'bare_soil', 'water'];

  return (
    <div className="dynamics-section fade-in-up">
      <div className="dynamics-header">
        <div className="dynamics-title">
          <span>🌐</span> Surface Dynamics Classification
        </div>
        <div className="dynamics-header-right">
          <span className="dynamics-hint">
            💡 Click cards or check/uncheck to filter colors on satellite overlay
          </span>
        </div>
      </div>

      <div className="dynamics-grid">
        {orderedKeys.map((key) => {
          const item = dynamics[key];
          if (!item) return null;

          const meta = DYNAMICS_META[key] || {
            label: item.label || key,
            sub: '',
            color: item.color || '#3b82f6',
            icon: '📌',
          };

          const isChecked = activeFilters ? (activeFilters[key] ?? true) : true;

          return (
            <div
              key={key}
              className={`dynamics-card ${isChecked ? 'is-active' : 'is-muted'}`}
              style={{
                '--cat-color': meta.color,
                cursor: onToggleFilter ? 'pointer' : 'default',
              }}
              onClick={() => onToggleFilter && onToggleFilter(key)}
              title={isChecked ? `Click to hide ${meta.label} from overlay` : `Click to show ${meta.label} on overlay`}
            >
              <div className="dynamics-card-top">
                <div
                  className="dynamics-icon-box"
                  style={{
                    background: isChecked ? `${meta.color}25` : 'rgba(255,255,255,0.05)',
                    color: isChecked ? meta.color : 'var(--c-text3)',
                    border: `1px solid ${isChecked ? `${meta.color}50` : 'transparent'}`,
                  }}
                >
                  {meta.icon}
                </div>
                <div className="dynamics-card-meta">
                  <div className="dynamics-card-label-row">
                    <span className="dynamics-card-label" style={{ color: isChecked ? 'var(--c-text)' : 'var(--c-text3)' }}>
                      {meta.label}
                    </span>
                    {/* Interactive checkbox tick badge */}
                    <div
                      className={`dynamics-card-tick-box ${isChecked ? 'checked' : ''}`}
                      style={{
                        background: isChecked ? meta.color : 'transparent',
                        borderColor: isChecked ? meta.color : 'var(--c-border)',
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleFilter && onToggleFilter(key);
                      }}
                      title={isChecked ? "Layer visible (click to hide)" : "Layer hidden (click to show)"}
                    >
                      {isChecked ? '✓' : ''}
                    </div>
                  </div>
                  <div className="dynamics-card-sub">{meta.sub}</div>
                </div>
              </div>

              <div className="dynamics-card-stats">
                <div
                  className="dynamics-stat-pct"
                  style={{ color: isChecked ? meta.color : 'var(--c-text3)' }}
                >
                  {item.pct}%
                </div>
                <div className="dynamics-stat-ha">
                  {item.ha?.toLocaleString() ?? 0} ha
                  <span className={`dynamics-visibility-tag ${isChecked ? 'visible' : 'hidden'}`}>
                    {isChecked ? '● Visible' : '○ Hidden'}
                  </span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="dynamics-progress-track">
                <div
                  className="dynamics-progress-bar"
                  style={{
                    width: `${Math.min(100, Math.max(2, item.pct))}%`,
                    background: isChecked ? meta.color : 'var(--c-text3)',
                    opacity: isChecked ? 1 : 0.3,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Modal for full-resolution inspection & download ───────── */
function Modal({ src, title, date, onClose }) {
  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = `${title.replace(/\s+/g, '_')}_${date || 'export'}.png`;
    a.click();
  };

  return (
    <div className="cd-modal-backdrop" onClick={onClose}>
      <div className="cd-modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="cd-modal-header">
          <div className="cd-modal-title">
            {title} {date ? `· ${date}` : ''}
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button className="btn btn-ghost" onClick={handleDownload} style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}>
              ⬇ Download PNG
            </button>
            <button className="cd-modal-close" onClick={onClose}>✕</button>
          </div>
        </div>
        <img src={src} alt={title} className="cd-modal-img" />
      </div>
    </div>
  );
}

/* ── Main ResultsPanel Component ───────────────────────────── */
export default function ResultsPanel({
  data,
  onOverlayChange,
  activeFilters: parentFilters,
  onToggleFilter: parentToggleFilter,
  onSetAllFilters: parentSetAllFilters,
}) {
  const [activeTab, setActiveTab] = useState('grid'); // 'grid' | 'comparison'
  const [modal, setModal] = useState(null);

  // Dynamic layer visibility filters (fallback if not controlled by parent)
  const [localFilters, setLocalFilters] = useState({
    urban: true,
    vegetation: true,
    bare_soil: true,
    water: true,
  });

  const activeFilters = parentFilters || localFilters;

  const toggleFilter = (key) => {
    if (parentToggleFilter) {
      parentToggleFilter(key);
    } else {
      setLocalFilters((prev) => ({ ...prev, [key]: !prev[key] }));
    }
  };

  const setAllFilters = (enable) => {
    if (parentSetAllFilters) {
      parentSetAllFilters(enable);
    } else {
      setLocalFilters({
        urban: enable,
        vegetation: enable,
        bare_soil: enable,
        water: enable,
      });
    }
  };

  // Filtered overlay state generated in client-side canvas
  const [filteredOverlay, setFilteredOverlay] = useState(data?.overlay || null);

  // Re-generate client-side blended overlay when active filters or imagery changes
  useEffect(() => {
    if (!data) return;

    createFilteredOverlay(
      data.t2_rgb,
      data.change_mask,
      data.overlay,
      activeFilters
    ).then((resUrl) => {
      setFilteredOverlay(resUrl);
      if (onOverlayChange) onOverlayChange(resUrl);
    });
  }, [data?.t2_rgb, data?.change_mask, data?.overlay, activeFilters]);

  if (!data) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🛰️</div>
        <div className="empty-title">Ready for Satellite Analysis</div>
        <div className="empty-desc">
          Select a region on the map, specify historical (T1) and monitoring (T2) dates, and click
          &ldquo;Analyze Change Detection&rdquo;.
        </div>
      </div>
    );
  }

  const {
    t1_rgb,
    t2_rgb,
    change_mask,
    binary_mask,
    change_prob,
    overlay,
    dynamics,
    stats,
    meta,
    location_name,
  } = data;

  const changePct = stats?.change_pct ?? 0;
  const badgeClass = changePct > 20 ? 'change-high' : changePct > 8 ? 'change-med' : 'change-low';
  const changeLabel = changePct > 20 ? '⚠ High Change' : changePct > 8 ? '⚡ Moderate Change' : '✅ Low Change';

  // Compute active filters count and label
  const activeKeys = Object.keys(activeFilters).filter((k) => activeFilters[k]);
  const activeCount = activeKeys.length;
  let layerBadgeText = `${activeCount}/4 Layers`;
  if (activeCount === 4) layerBadgeText = '🔥 All Layers (4/4)';
  else if (activeCount === 0) layerBadgeText = '⚪ Satellite Only';
  else if (activeCount === 1) {
    const single = DYNAMICS_META[activeKeys[0]]?.short || activeKeys[0];
    layerBadgeText = `🎯 ${single} Only`;
  }

  const openModal = (src, title, date) => setModal({ src, title, date });

  // Currently displayed overlay source for Panel 3 & Comparison
  const currentOverlaySrc = filteredOverlay || overlay;

  return (
    <div className="results-section fade-in-up">

      {/* ── Top Bar with View Mode Tabs ── */}
      <div className="results-header">
        <div className="results-title-group">
          <div className="results-title">
            📊 {location_name || 'Analysis Results'}
          </div>
          <span className={`results-badge ${badgeClass}`}>
            {changePct}% · {changeLabel}
          </span>
        </div>

        {/* View Switcher Tabs */}
        <div className="view-mode-tabs">
          <button
            className={`view-tab ${activeTab === 'grid' ? 'active' : ''}`}
            onClick={() => setActiveTab('grid')}
          >
            ▦ 3-Panel Detail Grid
          </button>
          <button
            className={`view-tab ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveTab('comparison')}
          >
            🖼 Side-by-Side Comparison
          </button>
        </div>
      </div>

      {/* ── View 1: 3-Panel Detail Grid ── */}
      {activeTab === 'grid' && (
        <div className="cd-grid">
          {/* Panel 1: T1 Optical (Historical Baseline) */}
          <PanelCard
            title="T1 Optical (Historical Baseline)"
            subtitle="Pre-event Satellite Image"
            icon="■"
            accent="#3b82f6"
            src={t1_rgb}
            date={meta?.t1_date}
            onClick={openModal}
            extraControl={
              <div className="panel-sub-info">
                <span>🗓️ Historical Reference Scene</span>
              </div>
            }
          />

          {/* Panel 2: T2 Optical (Recent Monitoring) */}
          <PanelCard
            title="T2 Optical (Recent Monitoring)"
            subtitle="Post-event Satellite Image"
            icon="■"
            accent="#06b6d4"
            src={t2_rgb}
            date={meta?.t2_date}
            onClick={openModal}
            extraControl={
              <div className="panel-sub-info">
                <span>🛰️ Recent Sentinel-2 Acquisition</span>
              </div>
            }
          />

          {/* Panel 3: T2 Overlap Mask (Satellite Fusion) with Dynamic Layer Filter */}
          <PanelCard
            title="T2 Overlap Mask (Satellite Fusion)"
            subtitle="Satellite + Dynamics Overlay"
            icon="🔥"
            accent="#ef4444"
            badge={layerBadgeText}
            src={currentOverlaySrc}
            onClick={openModal}
            extraControl={
              <div className="panel-layer-filter-bar">
                <span className="filter-bar-lead">Show Colors:</span>
                <div className="filter-chips-list">
                  {/* 🏗️ Urban & Structural */}
                  <label
                    className={`filter-tick-chip ${activeFilters.urban ? 'active' : ''}`}
                    style={{ '--cat-accent': '#eab308' }}
                    title="Toggle Urban & Structural (Yellow)"
                  >
                    <input
                      type="checkbox"
                      checked={activeFilters.urban}
                      onChange={() => toggleFilter('urban')}
                    />
                    <span className="chip-tick-mark">✓</span>
                    <span className="chip-label">🏗️ Urban</span>
                  </label>

                  {/* 🌿 Vegetation Dynamics */}
                  <label
                    className={`filter-tick-chip ${activeFilters.vegetation ? 'active' : ''}`}
                    style={{ '--cat-accent': '#22c55e' }}
                    title="Toggle Vegetation Dynamics (Green)"
                  >
                    <input
                      type="checkbox"
                      checked={activeFilters.vegetation}
                      onChange={() => toggleFilter('vegetation')}
                    />
                    <span className="chip-tick-mark">✓</span>
                    <span className="chip-label">🌿 Veg</span>
                  </label>

                  {/* 🌾 Soil & Agriculture */}
                  <label
                    className={`filter-tick-chip ${activeFilters.bare_soil ? 'active' : ''}`}
                    style={{ '--cat-accent': '#f97316' }}
                    title="Toggle Soil & Agriculture (Orange)"
                  >
                    <input
                      type="checkbox"
                      checked={activeFilters.bare_soil}
                      onChange={() => toggleFilter('bare_soil')}
                    />
                    <span className="chip-tick-mark">✓</span>
                    <span className="chip-label">🌾 Soil</span>
                  </label>

                  {/* 💧 Water Dynamics */}
                  <label
                    className={`filter-tick-chip ${activeFilters.water ? 'active' : ''}`}
                    style={{ '--cat-accent': '#2563eb' }}
                    title="Toggle Water Dynamics (Blue)"
                  >
                    <input
                      type="checkbox"
                      checked={activeFilters.water}
                      onChange={() => toggleFilter('water')}
                    />
                    <span className="chip-tick-mark">✓</span>
                    <span className="chip-label">💧 Water</span>
                  </label>

                  <div className="filter-quick-actions">
                    <button
                      type="button"
                      className="btn-filter-quick"
                      onClick={() => setAllFilters(true)}
                      title="Show all dynamic layers"
                    >
                      All
                    </button>
                    <button
                      type="button"
                      className="btn-filter-quick"
                      onClick={() => setAllFilters(false)}
                      title="Hide all colors (show natural satellite)"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </div>
            }
          />
        </div>
      )}

      {/* ── View 2: Interactive Side-by-Side Comparison ── */}
      {activeTab === 'comparison' && (
        <ComparisonSlider
          t1Src={t1_rgb}
          t2Src={t2_rgb}
          overlaySrc={currentOverlaySrc}
          t1Date={meta?.t1_date}
          t2Date={meta?.t2_date}
        />
      )}

      {/* ── Surface Dynamics Classification with interactive card toggles ── */}
      <DynamicsSection
        dynamics={dynamics}
        totalHa={stats?.changed_area_ha}
        changePct={changePct}
        activeFilters={activeFilters}
        onToggleFilter={toggleFilter}
      />

      {/* ── Dynamics Color Legend (Interactive: click to toggle) ── */}
      <div className="cd-legend">
        <span className="cd-legend-label">Dynamics Legend (Click to filter) →</span>
        <div className="cd-legend-bar">
          <div
            className={`cd-legend-stop ${activeFilters.water ? 'active' : 'inactive'}`}
            title="Click to toggle Water Dynamics"
            onClick={() => toggleFilter('water')}
            style={{ cursor: 'pointer' }}
          >
            <div className="cd-legend-swatch" style={{ background: '#2563eb', opacity: activeFilters.water ? 1 : 0.3 }} />
            <span className="cd-legend-text" style={{ opacity: activeFilters.water ? 1 : 0.5 }}>
              💧 Water {activeFilters.water ? '✓' : ''}
            </span>
          </div>

          <div
            className={`cd-legend-stop ${activeFilters.vegetation ? 'active' : 'inactive'}`}
            title="Click to toggle Vegetation Dynamics"
            onClick={() => toggleFilter('vegetation')}
            style={{ cursor: 'pointer' }}
          >
            <div className="cd-legend-swatch" style={{ background: '#22c55e', opacity: activeFilters.vegetation ? 1 : 0.3 }} />
            <span className="cd-legend-text" style={{ opacity: activeFilters.vegetation ? 1 : 0.5 }}>
              🌿 Vegetation {activeFilters.vegetation ? '✓' : ''}
            </span>
          </div>

          <div
            className={`cd-legend-stop ${activeFilters.urban ? 'active' : 'inactive'}`}
            title="Click to toggle Urban & Structural"
            onClick={() => toggleFilter('urban')}
            style={{ cursor: 'pointer' }}
          >
            <div className="cd-legend-swatch" style={{ background: '#eab308', opacity: activeFilters.urban ? 1 : 0.3 }} />
            <span className="cd-legend-text" style={{ opacity: activeFilters.urban ? 1 : 0.5 }}>
              🏗️ Urban/Struct {activeFilters.urban ? '✓' : ''}
            </span>
          </div>

          <div
            className={`cd-legend-stop ${activeFilters.bare_soil ? 'active' : 'inactive'}`}
            title="Click to toggle Soil & Agriculture"
            onClick={() => toggleFilter('bare_soil')}
            style={{ cursor: 'pointer' }}
          >
            <div className="cd-legend-swatch" style={{ background: '#f97316', opacity: activeFilters.bare_soil ? 1 : 0.3 }} />
            <span className="cd-legend-text" style={{ opacity: activeFilters.bare_soil ? 1 : 0.5 }}>
              🌾 Soil/Agri {activeFilters.bare_soil ? '✓' : ''}
            </span>
          </div>

          <div className="cd-legend-stop" title="No detected surface change">
            <div className="cd-legend-swatch" style={{ background: '#0a0f1d', border: '1px solid rgba(255,255,255,0.1)' }} />
            <span className="cd-legend-text">⬛ Stable Ground</span>
          </div>
        </div>
      </div>

      {/* ── Overall Metrics Row ── */}
      <div className="stats-row" style={{ marginTop: '1rem' }}>
        <div className="stat-card">
          <div
            className="stat-value"
            style={{ color: changePct > 20 ? '#f87171' : changePct > 8 ? '#fbbf24' : '#34d399' }}
          >
            {changePct}%
          </div>
          <div className="stat-label">Changed Area</div>
        </div>

        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--c-teal)' }}>
            {stats?.changed_area_ha?.toLocaleString() ?? '—'} ha
          </div>
          <div className="stat-label">Affected Hectares</div>
        </div>

        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--c-accent2)' }}>
            {stats?.max_prob ? `${(stats.max_prob * 100).toFixed(0)}%` : '—'}
          </div>
          <div className="stat-label">Peak Confidence</div>
        </div>

        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--c-purple)' }}>
            {stats?.threshold ?? 0.28}
          </div>
          <div className="stat-label">Model Threshold</div>
        </div>

        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--c-text2)' }}>
            {stats?.output_size?.join('×') || '512×512'}
          </div>
          <div className="stat-label">Resolution (px)</div>
        </div>
      </div>

      {/* ── Meta Info ── */}
      {meta && (
        <div className="cd-meta-row">
          <span>📅 T1 Window: {meta.t1_window || meta.t1_date}</span>
          <span>📅 T2 Date: {meta.t2_date}</span>
          <span>🎯 BBox: [{meta.bbox?.map((n) => Number(n).toFixed(3)).join(', ')}]</span>
        </div>
      )}

      {/* ── Modal ── */}
      {modal && (
        <Modal
          src={modal.src}
          title={modal.title}
          date={modal.date}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  );
}
