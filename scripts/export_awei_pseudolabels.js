/*
 * export_awei_pseudolabels.js
 *
 * Exports AWEI_sh-based water masks for UrbanSARFloods TRAINING events.
 * AWEI_sh suppresses both shadows and built-up area false positives,
 * making it more reliable than NDWI in urban scenes.
 *
 * AWEI_sh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
 * Threshold: AWEI_sh > 0 → water
 *
 * Strategy:
 *   - Only export for training events (not test events — no leakage)
 *   - Use clearest available Sentinel-2 image within ±14 days of flood peak
 *   - Apply minimum 8-pixel connected component filter to remove speckle
 *   - Export as binary water mask (1=water, 0=non-water)
 *
 * These labels are used ONLY for 02_FO (open flood) chips where
 * large contiguous water bodies fill 10m pixels cleanly.
 * 03_FU chips are excluded — mixed pixels in dense urban cores
 * make optical labels unreliable there.
 *
 * Output: Drive → water_detection_pseudolabels/
 *   {EventName}_awei_water_mask.tif
 */

var TRAIN_EVENTS = [
  // Events in training split only — no test events
  { name: 'Houston',   lon: -95.3698, lat:  29.7604, buffer_km: 60,
    flood_start: '2017-08-25', flood_end: '2017-09-10' },
  { name: 'Beira',     lon:  34.8389, lat: -19.8436, buffer_km: 40,
    flood_start: '2019-03-10', flood_end: '2019-04-05' },
  { name: 'Japan',     lon: 133.0000, lat:  34.2000, buffer_km: 60,
    flood_start: '2019-07-01', flood_end: '2019-08-15' },
  { name: 'Canada',    lon:-116.0000, lat:  55.0000, buffer_km: 60,
    flood_start: '2019-04-20', flood_end: '2019-05-20' },
  { name: 'Iran',      lon:  48.0000, lat:  31.5000, buffer_km: 60,
    flood_start: '2019-03-20', flood_end: '2019-04-15' },
  { name: 'Lumberton', lon: -79.0742, lat:  34.6182, buffer_km: 30,
    flood_start: '2016-10-01', flood_end: '2016-10-25' },
  { name: 'Somalia',   lon:  43.0000, lat:   5.0000, buffer_km: 40,
    flood_start: '2018-04-25', flood_end: '2018-05-20' },
];

print('Queueing AWEI_sh pseudo-label exports for ' + TRAIN_EVENTS.length + ' training events...');

TRAIN_EVENTS.forEach(function(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);

  // Get least-cloudy Sentinel-2 SR image during flood period
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.flood_start, event.flood_end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .first();

  // Fallback: wider cloud threshold if nothing found
  var s2_fallback = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.flood_start, event.flood_end)
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .first();

  var s2_image = ee.Image(ee.Algorithms.If(s2, s2, s2_fallback));

  // Compute AWEI_sh
  // Bands: B2=Blue, B3=Green, B8=NIR, B11=SWIR1, B12=SWIR2
  // Values are scaled 0-10000 in SR; divide by 10000 for reflectance
  var blue  = s2_image.select('B2').divide(10000);
  var green = s2_image.select('B3').divide(10000);
  var nir   = s2_image.select('B8').divide(10000);
  var swir1 = s2_image.select('B11').divide(10000);
  var swir2 = s2_image.select('B12').divide(10000);

  // AWEI_sh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
  var awei_sh = blue
    .add(green.multiply(2.5))
    .subtract(nir.add(swir1).multiply(1.5))
    .subtract(swir2.multiply(0.25));

  // Binary water mask: AWEI_sh > 0 → water
  var water_mask = awei_sh.gt(0).rename('water');

  // Remove isolated pixels (min 8 connected pixels = 800m² at 10m)
  // This suppresses speckle and shadow false positives
  var water_clean = water_mask
    .selfMask()
    .connectedPixelCount(25, true)
    .gte(8)
    .unmask(0)
    .toByte();

  // Also export the raw AWEI_sh value for threshold tuning if needed
  Export.image.toDrive({
    image:          water_clean,
    description:    event.name + '_awei_water_mask',
    folder:         'water_detection_pseudolabels',
    fileNamePrefix: event.name + '_awei_water_mask',
    region:         aoi.bounds(),
    scale:          10,     // 10m to match Sentinel-1 chip resolution
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  print('Queued: ' + event.name);
});

print('Done. Run all tasks in the Tasks tab.');
print('Download to: data/pseudolabels/');
print('');
print('NOTE: Only use these labels for 02_FO chips (open flood).');
print('Do NOT apply to 03_FU chips — mixed pixels in dense urban cores');
print('make optical labels unreliable there.');
