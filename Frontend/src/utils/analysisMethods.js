export const DEFAULT_ANALYSIS_ENGINE = 'geonexus_ai';

export const ANALYSIS_METHODS = Object.freeze({
  geonexus_ai: Object.freeze({
    id: 'geonexus_ai',
    label: 'Geo-Nexus AI Model',
    algorithm: 'MH-DAPT-CD v3.2 Few-Shot k=30',
    endpoint: '/api/detect/start',
  }),
  spectral_scva: Object.freeze({
    id: 'spectral_scva',
    label: 'Spectral Index Analysis (S-CVA)',
    algorithm: 'Spectral Change Vector Analysis (NDVI, NDWI, NDBI, BSI)',
    endpoint: '/api/detect/spectral/start',
  }),
});

export function getEngineLabel(result) {
  const engine = result?.engine;
  return result?.engine_label || ANALYSIS_METHODS[engine]?.label || 'Source unknown';
}

export function findPredictionForEngine(items, bbox, dateT1, dateT2, engine, tolerance = 0.008) {
  if (!ANALYSIS_METHODS[engine] || !bbox || !dateT1 || !dateT2) return null;
  return items.find((item) => {
    const itemEngine = item.engine || item.result?.engine || 'unknown';
    const bounds = item.bbox;
    if (itemEngine !== engine || item.dateT1 !== dateT1 || item.dateT2 !== dateT2 || !bounds) return false;
    return ['lon_min', 'lat_min', 'lon_max', 'lat_max'].every((key) =>
      Number.isFinite(Number(bounds[key])) && Math.abs(Number(bounds[key]) - Number(bbox[key])) <= tolerance
    );
  }) || null;
}
