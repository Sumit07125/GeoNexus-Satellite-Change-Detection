/**** Geo-Nexus v3.2 | Auto-label EVIDENCE export | run once per ZONE ********
 * Exports ONE 10-band Int16 GeoTIFF of raw evidence per zone.
 * NO thresholds are applied here -- fusion happens in Colab (Part 27).
 *
 * Bands (all Int16, scale factors documented per band):
 *   0 ob_p2020      building_presence 2020        x10000  [0..10000]
 *   1 ob_p2023      building_presence 2023        x10000  [0..10000]
 *   2 ob_h2023      building_height   2023 (m)    x100    [0..10000]
 *   3 hansen_ly     lossyear                      x1      [0..24]
 *   4 hansen_tc00   treecover2000 (%)             x1      [0..100]
 *   5 dw_built_t1   DW P(built) T1                x10000
 *   6 dw_built_t2   DW P(built) T2                x10000
 *   7 dw_trees_t1   DW P(trees) T1                x10000
 *   8 dw_trees_t2   DW P(trees) T2                x10000
 *   9 jrc_occ       JRC water occurrence (%)      x1      [0..100]
 ***************************************************************************/

// ============================ CONFIG =====================================
var ZONE = 'A';                       // <<<< 'A' | 'B' | 'C', run three times

var CFG = {
  A: {name:'pune',     crs:'EPSG:32643',
      geom: ee.Geometry.Rectangle([73.70, 18.30, 74.20, 18.80])},
  B: {name:'satara',   crs:'EPSG:32643',
      geom: ee.Geometry.Rectangle([73.50, 17.50, 74.00, 18.00])},
  C: {name:'vidarbha', crs:'EPSG:32644',
      geom: ee.Geometry.Rectangle([78.90, 20.90, 79.30, 21.30])}
};
var AOI = CFG[ZONE].geom, NAME = CFG[ZONE].name, CRS = CFG[ZONE].crs;
var SCALE = 10, FOLDER = 'geonexus_v3_raw', TILE = 5632;
var T1 = ['2020-01-01','2020-04-01'], T2 = ['2024-01-01','2024-04-01'];

// ==================== 1. OPEN BUILDINGS TEMPORAL =========================
// Annual snapshots are many 0.5 m tiles -> mosaic per year, then average
// down to the 10 m grid. reduceResolution with a MEAN reducer, never
// nearest-neighbour: NN on a 20x downsample produces aliased speckle.
var obCol = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')
              .filterBounds(AOI);

function obYear(year, band) {
  var col = obCol.filter(ee.Filter.calendarRange(year, year, 'year'))
                 .select(band);
  // Downsample at the individual tile level (each tile is small, ~4M px).
  // This avoids assigning a 0.5 m projection to a giant 55 km mosaic,
  // preventing the 2^31 pixel limit and user memory limit errors.
  var downsampled = col.map(function(img) {
    return img.setDefaultProjection(img.projection())
              .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4096})
              .reproject({crs: CRS, scale: SCALE});
  });
  return downsampled.mosaic();
}

// COVERAGE LIMIT L1: the collection ENDS 2023-06-30. There is no 2024 snapshot.
// Construction between Jul 2023 and Mar 2024 is invisible here and is caught
// only by the spectral evidence in Colab. Do not silently substitute 2023
// for 2024 without recording the gap.
var ob_p2020 = obYear(2020, 'building_presence').unmask(0);
var ob_p2023 = obYear(2023, 'building_presence').unmask(0);
var ob_h2023 = obYear(2023, 'building_height').unmask(0);

// =========================== 2. HANSEN ===================================
// 30.92 m. lossyear 1..24 == 2001..2024.
// Unmask to 0 so un-deforested pixels are strictly 0 rather than masked/null.
var hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var hansen_ly   = hansen.select('lossyear').unmask(0);
var hansen_tc00 = hansen.select('treecover2000').unmask(0);

// ======================= 3. DYNAMIC WORLD ================================
// Median of per-scene probabilities across the whole dry quarter.
// ONLY 'built' and 'trees'. crops / grass / bare / shrub / flooded are
// transient in Maharashtra and differencing them manufactures change.
function dwProb(period, band) {
  return ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(AOI).filterDate(period[0], period[1])
    .select(band).median().unmask(0);
}
var dw_built_t1 = dwProb(T1, 'built'),  dw_built_t2 = dwProb(T2, 'built');
var dw_trees_t1 = dwProb(T1, 'trees'),  dw_trees_t2 = dwProb(T2, 'trees');

print('DW scene count T1:', ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterBounds(AOI).filterDate(T1[0], T1[1]).size());

// ===================== 4. JRC WATER OCCURRENCE ===========================
// Static prior only. JRC/GSW1_4/YearlyHistory ENDS IN 2021 and is therefore
// useless for a 2024 comparison -- water change comes from your own MNDWI.
var jrc_occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
                .select('occurrence').unmask(0);

// ========================= 5. STACK & EXPORT =============================
// One consistent Int16 container (the Error-code-3 lesson from §4 of progress.md).
var evidence = ee.Image.cat([
  ob_p2020.multiply(10000).rename('ob_p2020'),
  ob_p2023.multiply(10000).rename('ob_p2023'),
  ob_h2023.multiply(100).rename('ob_h2023'),
  hansen_ly.rename('hansen_ly'),
  hansen_tc00.rename('hansen_tc00'),
  dw_built_t1.multiply(10000).rename('dw_built_t1'),
  dw_built_t2.multiply(10000).rename('dw_built_t2'),
  dw_trees_t1.multiply(10000).rename('dw_trees_t1'),
  dw_trees_t2.multiply(10000).rename('dw_trees_t2'),
  jrc_occ.rename('jrc_occ')
]).toInt16().clip(AOI);

Export.image.toDrive({
  image: evidence,
  description: NAME + '_autolabel_evidence',
  folder: FOLDER, fileNamePrefix: NAME + '_autolabel_evidence',
  region: AOI, scale: SCALE, crs: CRS,
  maxPixels: 1e10, fileDimensions: TILE,
  fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}
});

// ===================== 6. PRE-EXPORT SANITY CHECKS =======================
// Run these and READ them before starting the task.
print('--- OB temporal coverage in this AOI ---');
print('available years:', obCol.aggregate_array('system:time_start').distinct()
        .map(function(t){ return ee.Date(t).format('YYYY-MM-dd'); }));

// Fast interactive sanity check over a 5 km sample around the AOI center.
// Using tileScale: 4 prevents browser interactive memory limits while verifying statistics.
var sampleArea = AOI.centroid().buffer(2500).bounds();
print('--- evidence layer statistics (5 km sample area) ---');
print(evidence.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.minMax(), '', true),
  geometry: sampleArea, scale: 10, maxPixels: 1e7, tileScale: 4, bestEffort: true}));

// Hansen lossyear QA diagnostic: check if any deforestation pixels exist in the 5 km sample
print('Hansen raw lossyear valid pixel count in sample:', hansen.select('lossyear').reduceRegion({
  reducer: ee.Reducer.count(), geometry: sampleArea, scale: 30, maxPixels: 1e7, bestEffort: true}));

// GATE: if ob_p2020 and ob_p2023 are both ~0 everywhere, Open Buildings does
// not cover this AOI and construction evidence must come from spectral
// sources alone. Check before building the pipeline around it.

// ========================== 7. VISUAL QA =================================
Map.centerObject(AOI, 11);
Map.addLayer(ob_p2020, {min:0, max:1, palette:['black','white']}, 'OB presence 2020', false);
Map.addLayer(ob_p2023, {min:0, max:1, palette:['black','white']}, 'OB presence 2023', false);

// Quick-look construction candidate -- the SAME triple rule used in Colab
var newBuild = ob_p2023.gte(0.60)
                 .and(ob_p2020.lte(0.20))
                 .and(ob_p2023.subtract(ob_p2020).gte(0.45));
Map.addLayer(newBuild.selfMask(), {palette:['#FFC800']}, 'NEW BUILDING candidate');

var vegLoss = hansen_ly.gte(20).and(hansen_ly.lte(23));
Map.addLayer(vegLoss.selfMask(), {palette:['#DC3232']}, 'Hansen loss 2020-2023');
Map.addLayer(jrc_occ, {min:0, max:100, palette:['white','blue']}, 'JRC occurrence', false);
