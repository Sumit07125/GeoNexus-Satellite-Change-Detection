import React, { useState } from 'react';
import { getEngineLabel } from '../utils/analysisMethods';

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
function ComparisonSlider({ t1Src, t2Src, t1Date, t2Date }) {
  const [sliderPos, setSliderPos] = useState(50);

  const leftSrc = t1Src;
  const rightSrc = t2Src;
  const leftLabel = `Before T1 (${t1Date || 'Pre'})`;
  const rightLabel = `Current T2 (${t2Date || 'Post'})`;

  return (
    <div className="comparison-container">
      <div className="comparison-toolbar">
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
function DynamicsSection({ dynamics }) {
  if (!dynamics || Object.keys(dynamics).length === 0) return null;

  // Render order: Urban & Structural first, then Vegetation, Soil, Water
  const orderedKeys = ['urban', 'vegetation', 'bare_soil', 'water'];

  return (
    <div className="dynamics-section fade-in-up">
      <div className="dynamics-header">
        <div className="dynamics-title" title="Deterministic spectral rule-based post-processing breakdown of predicted changed pixels. Not a learned multi-class segmentation head.">
          <span>🌐</span> Surface Dynamics Interpretation (Rule-Based Decomposition)
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

          const isChecked = true;

          return (
            <div
              key={key}
              className={`dynamics-card ${isChecked ? 'is-active' : 'is-muted'}`}
              style={{
                '--cat-color': meta.color,
                cursor: 'default',
              }}
              
              title={meta.label}
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
export default function ResultsPanel({ data }) {
  const [activeTab, setActiveTab] = useState('grid'); // 'grid' | 'comparison'
  const [modal, setModal] = useState(null);
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
    binary_mask,
    change_prob,
    dynamics,
    stats,
    meta,
    location_name,
    engine,
    algorithm,
    model_version,
    threshold,
    threshold_source,
  } = data;
  const sourceLabel = getEngineLabel(data);
  const isAi = engine === 'geonexus_ai';
  const exportSummary = () => {
    const summary = {
      prediction_id: data.saved_id || null,
      status: data.success ? 'success' : 'unknown',
      engine: engine || 'unknown',
      source: sourceLabel,
      algorithm: algorithm || 'unknown',
      model_version: model_version || null,
      location: location_name || 'Selected Area',
      dates: { t1: meta?.t1_date || null, t2: meta?.t2_date || null },
      provider: meta?.provider || 'unknown',
      acquisition: meta || {},
      threshold: threshold ?? stats?.threshold ?? null,
      threshold_source: threshold_source || 'unknown',
      statistics: stats || {},
      dynamics: dynamics || {},
      note: 'Changed-area percentage is a model or spectral output estimate, not evaluation accuracy. Probability/score is not a calibrated confidence metric.',
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `geo-nexus_${engine || 'unknown'}_${(meta?.t1_date || 't1')}_${(meta?.t2_date || 't2')}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const changePct = stats?.change_pct ?? 0;
  const badgeClass = changePct > 20 ? 'change-high' : changePct > 8 ? 'change-med' : 'change-low';
  const changeLabel = changePct > 20 ? '⚠ High Change' : changePct > 8 ? '⚡ Moderate Change' : '✅ Low Change';
  const openModal = (src, title, date) => setModal({ src, title, date });
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
          <span className={`engine-source-badge ${isAi ? 'ai' : engine === 'spectral_scva' ? 'spectral' : 'unknown'}`}>
            {sourceLabel}
          </span>
        </div>

        <button className="btn btn-ghost" type="button" onClick={exportSummary}>↓ Export result summary</button>

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

          <PanelCard
            title="Binary Change Mask"
            subtitle="White = flagged change · Black = unchanged"
            icon="◩"
            accent="#a78bfa"
            src={binary_mask}
            onClick={openModal}
            extraControl={
              <div className="mask-legend"><span><i className="mask-swatch changed" /> Flagged change</span><span><i className="mask-swatch stable" /> Unchanged</span></div>
            }
          />
          <PanelCard
            title={isAi ? 'Change Probability Map' : 'Spectral Change Magnitude'}
            subtitle={isAi ? 'Sigmoid model output · not calibrated confidence' : 'Normalized heuristic score · not AI confidence'}
            icon="◉"
            accent="#38bdf8"
            src={change_prob}
            onClick={openModal}
            extraControl={<div className="heatmap-legend"><span>Lower score</span><i /><span>Higher score</span></div>}
          />
        </div>
      )}

      {/* ── View 2: Interactive Side-by-Side Comparison ── */}
      {activeTab === 'comparison' && (
        <ComparisonSlider
          t1Src={t1_rgb}
          t2Src={t2_rgb}
          t1Date={meta?.t1_date}
          t2Date={meta?.t2_date}
        />
      )}

      {/* ── Surface Dynamics Classification with interactive card toggles ── */}
      <DynamicsSection
        dynamics={dynamics}
      />


      {/* static Dynamics Color Legend */}
      <div className="cd-legend">
        <span className="cd-legend-label">Surface Dynamics Legend</span>
        <div className="cd-legend-bar">
          <div className="cd-legend-stop"><div className="cd-legend-swatch" style={{ background: "#2563eb" }} /><span className="cd-legend-text">Water</span></div>
          <div className="cd-legend-stop"><div className="cd-legend-swatch" style={{ background: "#22c55e" }} /><span className="cd-legend-text">Vegetation</span></div>
          <div className="cd-legend-stop"><div className="cd-legend-swatch" style={{ background: "#eab308" }} /><span className="cd-legend-text">Urban</span></div>
          <div className="cd-legend-stop"><div className="cd-legend-swatch" style={{ background: "#f97316" }} /><span className="cd-legend-text">Soil/Agri</span></div>
          <div className="cd-legend-stop"><div className="cd-legend-swatch" style={{ background: "#0a0f1d", border: "1px solid rgba(255,255,255,0.1)" }} /><span className="cd-legend-text">Stable</span></div>
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
            {stats?.changed_area_ha != null ? `${stats.changed_area_ha.toLocaleString()} ha` : '—'}
          </div>
          <div className="stat-label">Estimated changed area</div>
        </div>

        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--c-accent2)' }}>
            {Number.isFinite(stats?.max_prob) ? `${(stats.max_prob * 100).toFixed(0)}%` : '—'}
          </div>
          <div className="stat-label">{isAi ? 'Peak predicted probability' : 'Peak spectral score'}</div>
        </div>

        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--c-purple)' }}>
            {threshold ?? stats?.threshold ?? '—'}
          </div>
          <div className="stat-label">Applied {isAi ? 'probability' : 'CVA magnitude'} threshold</div>
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
          <span>🛰 Provider: {meta.provider || 'Unknown'}</span>
          {model_version && <span>🧠 Model: {model_version}</span>}
          <span>🎯 BBox: [{meta.bbox?.map((n) => Number(n).toFixed(3)).join(', ')}]</span>
          {stats?.pixel_area_m2_estimate != null && <span>▦ Approx. pixel area: {stats.pixel_area_m2_estimate.toLocaleString()} m²</span>}
          {threshold_source && <span>Threshold source: {threshold_source}</span>}
          {meta.resolution_note && <span>{meta.resolution_note}</span>}
        </div>
      )}
      <p className="results-metric-note">Changed-area percentage and hectares are estimates for this output raster; they are not evaluation accuracy. {isAi ? 'The displayed probability is the model sigmoid output and is not calibrated confidence.' : 'Spectral scores are heuristic change magnitudes, not AI confidence.'}</p>

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
