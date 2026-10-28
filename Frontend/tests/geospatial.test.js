import test from 'node:test';
import assert from 'node:assert/strict';
import { bboxDimensionsKm, bboxDimensionsMeters, createSquareBbox } from '../src/utils/geospatial.js';

test('2 km center-box control produces a real approximately 2 km by 2 km extent near Mumbai', () => {
  const bbox = createSquareBbox(19.1158, 72.84795, 2);
  const dimensions = bboxDimensionsKm(bbox);
  assert.ok(Math.abs(dimensions.widthKm - 2) < 0.01);
  assert.ok(Math.abs(dimensions.heightKm - 2) < 0.01);
  assert.ok(Math.abs((bbox.lat_max - bbox.lat_min) - 0.018) < 0.001);
});

test('the reported screenshot bounds measure about 1 km and are below AI grid minimum', () => {
  const bbox = { lon_min: 72.8432, lat_min: 19.1113, lon_max: 72.8527, lat_max: 19.1203 };
  const dimensions = bboxDimensionsMeters(bbox);
  assert.ok(dimensions.width > 950 && dimensions.width < 1050);
  assert.ok(dimensions.height > 950 && dimensions.height < 1050);
  assert.equal(bboxDimensionsKm(bbox).isSubModelMinimum, true);
});

test('enlarging the same center to 1.4 km clears the minimum on both axes', () => {
  const bbox = createSquareBbox(19.1158, 72.84795, 1.4);
  const dimensions = bboxDimensionsMeters(bbox);
  assert.ok(dimensions.width >= 1280);
  assert.ok(dimensions.height >= 1280);
});
