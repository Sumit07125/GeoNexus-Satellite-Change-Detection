/**** Geo-Nexus v3.1 | Single-Date Monsoon Export (Zone A) *****************
 * Exports:
 *   1. Single-date Sentinel-2 scene with 30-70% real cloud cover (NOT composite!)
 *   2. Cloud Score+ continuous cs_cdf quality band (q255)
 *   3. Nearest Sentinel-1 pass on same locked orbit
 *
 * This test set is the ONLY place where q genuinely varies spatially,
 * so it is the ONLY place where Hypothesis H2 (SAR helping under clouds)
 * can be falsified!
 ***************************************************************************/

var AOI    = ee.Geometry.Rectangle([73.70, 18.30, 74.20, 18.80]); // Zone A (Pune)
var CRS    = 'EPSG:32643';
var FOLDER = 'geonexus_v3_raw';
var S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'];
var ORBIT  = 136; // Locked common relative orbit (DESCENDING)

function candidates(d0, d1) {
  var c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(AOI).filterDate(d0, d1)
    .filter(ee.Filter.rangeContains('CLOUDY_PIXEL_PERCENTAGE', 30, 70))
    .sort('CLOUDY_PIXEL_PERCENTAGE');
  print(d0 + ' candidates:', c.size(),
        c.aggregate_array('system:index'),
        c.aggregate_array('CLOUDY_PIXEL_PERCENTAGE'));
  return c;
}

// Candidates checked:
// 2020: 20200911T052651_20200911T053945_T43QCA (41.2% cloud)
// 2024: 20240920T052651_20240920T053347_T43QCA (38.0% cloud)
var ID20 = '20200911T052651_20200911T053945_T43QCA';
var ID24 = '20240920T052651_20240920T053347_T43QCA';

var csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

function sceneWithQ(id) {
  var img = ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + id);
  var cs  = ee.Image(csp.filter(ee.Filter.eq('system:index', id)).first());
  // All bands in a multi-band GeoTIFF must share a single consistent data type (Int16)
  return img.select(S2_BANDS).divide(10000).multiply(10000).toInt16()
            .addBands(cs.select('cs_cdf').multiply(255).toInt16().rename('q255'))
            .clip(AOI);
}

function toNatural(i){ return ee.Image(10).pow(i.divide(10)); }
function toDB(i){ return i.log10().multiply(10); }

// Architectural Fix: Select the literal SINGLE nearest S1 pass on the locked orbit,
// NOT a multi-day temporal median, ensuring a genuine single-date test pair.
function nearestSar(dateStr) {
  var targetDate = ee.Date(dateStr);
  var col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(AOI)
    .filterDate(targetDate.advance(-14, 'day'), targetDate.advance(14, 'day'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', ORBIT))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);

  // Add absolute time difference in milliseconds to sort by nearest
  var withTimeDiff = col.map(function (img) {
    var diff = ee.Number(img.get('system:time_start')).subtract(targetDate.millis()).abs();
    return img.set('time_diff', diff);
  });

  var nearest = ee.Image(withTimeDiff.sort('time_diff').first());
  return nearest.multiply(100).toInt16().clip(AOI);
}

[['T1', ID20, '2020-09-11'], ['T2', ID24, '2024-09-20']].forEach(function (a) {
  Export.image.toDrive({
    image: sceneWithQ(a[1]),
    description: 'pune_monsoon_' + a[0] + '_optical', folder: FOLDER,
    fileNamePrefix: 'pune_monsoon_' + a[0] + '_optical',
    region: AOI, scale: 10, crs: CRS, maxPixels: 1e10, fileDimensions: 5632,
    fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}
  });
  Export.image.toDrive({
    image: nearestSar(a[2]),
    description: 'pune_monsoon_' + a[0] + '_sar', folder: FOLDER,
    fileNamePrefix: 'pune_monsoon_' + a[0] + '_sar',
    region: AOI, scale: 10, crs: CRS, maxPixels: 1e10, fileDimensions: 5632,
    fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}
  });
});

function getNearestSarInfo(dateStr) {
  var targetDate = ee.Date(dateStr);
  var col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(AOI)
    .filterDate(targetDate.advance(-14, 'day'), targetDate.advance(14, 'day'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', ORBIT))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
  var withTimeDiff = col.map(function (img) {
    var diff = ee.Number(img.get('system:time_start')).subtract(targetDate.millis()).abs();
    return img.set('time_diff', diff);
  });
  var nearest = ee.Image(withTimeDiff.sort('time_diff').first());
  return ee.Dictionary({
    id: nearest.get('system:index'),
    datetime: ee.Date(nearest.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss')
  });
}

// ================= Console Output & Map Preview =================
print('=============================================');
print('MONSOON EXPORT INITIALIZED SUCCESSFULLY!');
print('T1 (2020) S2 Scene ID:', ID20);
print('T1 (2020) S1 SAR Nearest:', getNearestSarInfo('2020-09-11'));
print('T2 (2024) S2 Scene ID:', ID24);
print('T2 (2024) S1 SAR Nearest:', getNearestSarInfo('2024-09-20'));
print('Orbit:', ORBIT);
print('>>> CHECK THE "TASKS" TAB ON THE RIGHT TO RUN THE 4 EXPORT TASKS! <<<');
print('=============================================');

Map.centerObject(AOI, 10);
Map.addLayer(sceneWithQ(ID20), {bands:['B4','B3','B2'], min:0, max:3000}, '2020 Monsoon Optical (RGB)');
Map.addLayer(sceneWithQ(ID24), {bands:['B4','B3','B2'], min:0, max:3000}, '2024 Monsoon Optical (RGB)');

