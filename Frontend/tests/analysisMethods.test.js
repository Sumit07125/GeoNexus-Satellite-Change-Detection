import test from 'node:test';
import assert from 'node:assert/strict';
import { ANALYSIS_METHODS, DEFAULT_ANALYSIS_ENGINE, findPredictionForEngine, getEngineLabel } from '../src/utils/analysisMethods.js';

test('trained AI is the default and the engines use separate API endpoints', () => {
  assert.equal(DEFAULT_ANALYSIS_ENGINE, 'geonexus_ai');
  assert.equal(ANALYSIS_METHODS.geonexus_ai.endpoint, '/api/detect/start');
  assert.equal(ANALYSIS_METHODS.spectral_scva.endpoint, '/api/detect/spectral/start');
});

test('result labels distinguish AI, S-CVA, and legacy unknown sources', () => {
  assert.equal(getEngineLabel({ engine: 'geonexus_ai' }), 'Geo-Nexus AI Model');
  assert.equal(getEngineLabel({ engine: 'spectral_scva' }), 'Spectral Index Analysis (S-CVA)');
  assert.equal(getEngineLabel({}), 'Analysis source unknown');
});

test('cache matching never returns a result from another engine or guesses legacy engine', () => {
  const bbox = { lon_min: 73, lat_min: 18, lon_max: 73.1, lat_max: 18.1 };
  const common = { bbox, dateT1: '2021-01-01', dateT2: '2022-01-01', result: {} };
  const items = [
    { ...common, id: 'spectral', engine: 'spectral_scva' },
    { ...common, id: 'legacy' },
    { ...common, id: 'ai', engine: 'geonexus_ai' },
  ];
  assert.equal(findPredictionForEngine(items, bbox, '2021-01-01', '2022-01-01', 'geonexus_ai').id, 'ai');
  assert.equal(findPredictionForEngine(items, bbox, '2021-01-01', '2022-01-01', 'spectral_scva').id, 'spectral');
  assert.equal(findPredictionForEngine(items, bbox, '2021-01-01', '2022-01-01', 'unknown'), null);
});
