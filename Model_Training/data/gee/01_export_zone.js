/**** Geo-Nexus v3.1 | Main Export Script | Run once per ZONE **************
 * Exports:
 *   1. Optical S2 L2A Harmonized (11 bands + n_clear quality band)
 *   2. SAR S1 GRD IW (VV, VH) composited in linear power with locked relative orbit
 *   3. AlphaEarth Annual Embedding Change Score (cosine distance)
 *   4. JRC Global Surface Water Mask (seasonality band)
 ***************************************************************************/

// ============================ CONFIG =====================================
// CHANGE THIS: 'A' | 'B' | 'C' and click Run
var ZONE = 'A';

var CFG = {
  A: {name:'pune',     crs:'EPSG:32643',
      geom: ee.Geometry.Rectangle([73.70, 18.30, 74.20, 18.80])},
  B: {name:'satara',   crs:'EPSG:32643',
      geom: ee.Geometry.Rectangle([73.50, 17.50, 74.00, 18.00])},
  C: {name:'vidarbha', crs:'EPSG:32644',      // 79 E is UTM zone 44N, NOT 43N!
      geom: ee.Geometry.Rectangle([78.90, 20.90, 79.30, 21.30])}
};

var AOI    = CFG[ZONE].geom;
var NAME   = CFG[ZONE].name;
var CRS    = CFG[ZONE].crs;
var SCALE  = 10;
var FOLDER = 'geonexus_v3_raw';
var CLEAR_THRESHOLD = 0.60;
var TILE = 5632;      // 22 * 256 -> export tiles align to patch grid

var PERIODS = {T1: ['2020-01-01','2020-04-01'], T2: ['2024-01-01','2024-04-01']};
var S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'];

// =========================== OPTICAL =====================================
function opticalComposite(d0, d1) {
  var s2  = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')   // D1 FIX
              .filterBounds(AOI).filterDate(d0, d1);
  var csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

  var masked = s2.linkCollection(csp, ['cs_cdf']).map(function (img) {
    return img.updateMask(img.select('cs_cdf').gte(CLEAR_THRESHOLD))
              .select(S2_BANDS).divide(10000)
              .copyProperties(img, ['system:time_start']);
  });

  // D3 FIX: per-pixel clear-observation count -> the quality signal q
  var nClear = masked.select('B4').count().rename('n_clear').unmask(0);

  print('S2 scenes ' + d0 + ' -> ' + d1 + ':', s2.size());
  return masked.median().addBands(nClear).clip(AOI);
}

// ============================= SAR =======================================
function s1Base(d0, d1) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(AOI).filterDate(d0, d1)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
}

var ORBIT_MAP = {
  A: 136, // Zone A (Pune): 10 scenes T1, 5 scenes T2 (DESCENDING)
  B: 136, // Zone B (Satara): 14 scenes T1, 10 scenes T2 (DESCENDING)
  C: 165  // Zone C (Vidarbha): 13 scenes T1, 12 scenes T2 (DESCENDING)
};

var ORBIT = ORBIT_MAP[ZONE];

function reportOrbits() {
  ['T1','T2'].forEach(function (t) {
    var c = s1Base(PERIODS[t][0], PERIODS[t][1]);
    print('Zone ' + ZONE + ' ' + t + ' S1 orbit numbers:', c.aggregate_array('relativeOrbitNumber_start').distinct());
    print('Zone ' + ZONE + ' ' + t + ' S1 directions:', c.aggregate_array('orbitProperties_pass').distinct());
    print('Zone ' + ZONE + ' ' + t + ' S1 scenes:', c.size());
  });
}
// reportOrbits(); // Verified: A=136, B=136, C=165

function toNatural(img) { return ee.Image(10).pow(img.divide(10)); }
function toDB(img)      { return img.log10().multiply(10); }

function sarComposite(d0, d1) {
  var col = s1Base(d0, d1)
              .filter(ee.Filter.eq('relativeOrbitNumber_start', ORBIT))
              .select(['VV','VH']);
  var n = col.size();
  print('S1 scenes ' + d0 + ' on orbit ' + ORBIT + ':', n);

  // D5 FIX: Composite in LINEAR power, not in dB
  var med = toDB(col.map(toNatural).median());

  // D5 FIX: Spatial despeckle ONLY when temporal median has too few looks (<5)
  return ee.Image(ee.Algorithms.If(n.lt(5),
             med.focal_median(1.5, 'circle', 'pixels'),   // ~3x3 = 30 m
             med)).clip(AOI);
}

// =========================== EXPORTS =====================================
function exportImg(img, suffix) {
  Export.image.toDrive({
    image: img, description: NAME + '_' + suffix,
    folder: FOLDER, fileNamePrefix: NAME + '_' + suffix,
    region: AOI, scale: SCALE, crs: CRS,
    maxPixels: 1e10, fileDimensions: TILE,
    fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}
  });
}

['T1','T2'].forEach(function (t) {
  var opt = opticalComposite(PERIODS[t][0], PERIODS[t][1]);
  // Reflectance x10000 -> int16 (rho <= 3.2 is safe, max int16 = 32767)
  // All bands in a multi-band GeoTIFF must share a single consistent data type (Int16)
  exportImg(opt.select(S2_BANDS).multiply(10000).toInt16()
               .addBands(opt.select('n_clear').toInt16()), t + '_optical');

  // dB x100 -> int16 ([-35,+5] dB -> [-3500,+500]. NEVER x10000 here to avoid overflow!)
  exportImg(sarComposite(PERIODS[t][0], PERIODS[t][1]).multiply(100).toInt16(),
            t + '_sar');
});

// ---- AlphaEarth: Baseline + Active Stratified Sampling ----
var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
var e20 = emb.filterDate('2020-01-01','2021-01-01').filterBounds(AOI).mosaic().clip(AOI);
var e24 = emb.filterDate('2024-01-01','2025-01-01').filterBounds(AOI).mosaic().clip(AOI);
// Embeddings are unit-length -> dot product IS cosine similarity
var chg = ee.Image(1).subtract(e20.multiply(e24).reduce(ee.Reducer.sum()))
            .rename('change_score'); // in [0, 2]
exportImg(chg.multiply(10000).toInt16(), 'alphaearth_change');

// ---- JRC Water Mask (Zone B Reservoir Confound Masking) ----
exportImg(ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            .select('seasonality').gte(1).unmask(0).toUint8(), 'watermask');

// ---- Visual inspection ----
Map.centerObject(AOI, 10);
var t1 = opticalComposite(PERIODS.T1[0], PERIODS.T1[1]);
var t2 = opticalComposite(PERIODS.T2[0], PERIODS.T2[1]);
Map.addLayer(t1, {bands:['B4','B3','B2'], min:0, max:0.3}, 'T1 RGB');
Map.addLayer(t2, {bands:['B4','B3','B2'], min:0, max:0.3}, 'T2 RGB');
Map.addLayer(chg, {min:0, max:0.8, palette:['black','orange','red']}, 'AlphaEarth change');
Map.addLayer(t1.select('n_clear'), {min:0,max:15,palette:['red','yellow','green']}, 'n_clear');
