import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('warning separation and deduplication isolates Quality and SAR notices cleanly', () => {
  const mixedWarnings = [
    {
      code: 'quality_signal_low_spatial_variance',
      date: '2021-03-15',
      label: 'T1',
      channel: 'Q',
      channel_index: 16,
      raw_n_clear_std: 0.12,
      encoded_min: 0.85,
      encoded_max: 0.88,
    },
    {
      code: 'quality_signal_low_spatial_variance',
      date: '2024-03-15',
      label: 'T2',
      channel: 'Q',
      channel_index: 16,
      raw_n_clear_std: 0.15,
      encoded_min: 0.85,
      encoded_max: 0.88,
    },
    // Duplicate of T1 Q warning
    {
      code: 'quality_signal_low_spatial_variance',
      date: '2021-03-15',
      label: 'T1',
      channel: 'Q',
      channel_index: 16,
      raw_n_clear_std: 0.12,
      encoded_min: 0.85,
      encoded_max: 0.88,
    },
    // SAR notices
    {
      code: 'sar_vv_urban_reflection_notice',
      channel: 'VV',
      channel_index: 13,
      label: 'T1',
      date: '2021-03-15',
      observed_min: -2500,
      observed_max: 1631,
      outside_count: 36,
      interpretation: 'High co-polarization backscatter characteristic of urban double-bounce.',
    },
    {
      code: 'sar_vh_distribution_notice',
      channel: 'VH',
      channel_index: 14,
      label: 'T2',
      date: '2024-03-15',
      observed_min: -3200,
      observed_max: 600,
      outside_count: 12,
      interpretation: 'VH has values outside typical training quantiles.',
    },
    {
      code: 'sar_vh_vv_contrast_notice',
      channel: 'VH_VV',
      channel_index: 15,
      label: 'T2',
      date: '2024-03-15',
      observed_min: -3600,
      observed_max: -200,
      outside_count: 5,
      interpretation: 'Strong cross-polarization contrast.',
    },
  ];

  // Deduplication
  const seen = new Set();
  const deduped = mixedWarnings.filter((w) => {
    const k = `${w.code || ''}_${w.label || ''}_${w.date || ''}_${w.channel || ''}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });

  assert.equal(deduped.length, 5, 'Duplicate warning should be filtered out');

  const qWarnings = deduped.filter(
    (w) => w.code === 'quality_signal_low_spatial_variance' || (typeof w.code === 'string' && w.code.includes('quality'))
  );
  const sarWarnings = deduped.filter(
    (w) => typeof w.code === 'string' && w.code.startsWith('sar_')
  );

  assert.equal(qWarnings.length, 2, 'Should have exactly 2 distinct Q warnings');
  assert.equal(sarWarnings.length, 3, 'Should have exactly 3 distinct SAR warnings');

  // SAR warnings must not have Q-channel fields
  for (const s of sarWarnings) {
    assert.equal(s.raw_n_clear_std, undefined);
    assert.equal(s.encoded_min, undefined);
    assert.equal(s.encoded_max, undefined);
    assert.ok(s.channel === 'VV' || s.channel === 'VH' || s.channel === 'VH_VV');
    assert.ok(typeof s.observed_min === 'number');
    assert.ok(typeof s.observed_max === 'number');
  }
});

test('App.jsx renders distinct Quality and SAR warning sections without broken fields', async () => {
  const app = await readFile(new URL('../src/App.jsx', import.meta.url), 'utf8');
  assert.match(app, /Radar Backscatter Distribution Notice/);
  assert.match(app, /Quality Distribution Notice/);
  assert.match(app, /seen\.has\(k\)/);
  assert.match(app, /qWarnings/);
  assert.match(app, /sarWarnings/);
  assert.match(app, /Sentinel-1 C-band backscatter values passed Geo-Nexus project physical guardrails/);
  assert.match(app, /Spatially uniform Q does not guarantee prediction accuracy/);
});

test('ResultsPanel explicitly labels surface dynamics decomposition as rule-based', async () => {
  const panel = await readFile(new URL('../src/components/ResultsPanel.jsx', import.meta.url), 'utf8');
  assert.match(panel, /Surface Dynamics Interpretation \(Rule-Based Decomposition\)/);
  assert.match(panel, /Deterministic spectral rule-based post-processing breakdown/);
});
