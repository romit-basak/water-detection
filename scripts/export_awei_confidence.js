/*
 * export_awei_confidence.js
 *
 * Exports AWEI_sh-based CONTINUOUS confidence maps for UrbanSARFloods
 * training events. Instead of a binary water mask, each pixel gets a
 * confidence value in [0, 1] representing how likely it is to be water.
 *
 * AWEI_sh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
 *
 * Confidence mapping:
 *   AWEI_sh >= +0.3  → confidence = 1.0  (clearly water)
 *   AWEI_sh = 0      → confidence = 0.5  (uncertain)
 *   AWEI_sh <= -0.3  → confidence = 0.0  (clearly non-water)
 *   Linear interpolation between these anchors.
 *
 * Shadow masking:
 *   Pixels with low NIR and low Blue are likely in building shadow.
 *   Shadow pixels get confidence = -1 (ignore completely — not water,
 *   but also not confidently non-water).
 *
 * Why confidence maps instead of binary masks?
 *   - Pixels near the AWEI threshold (mixed pixels in urban cores)
 *     contribute low gradient rather than corrupting labels.
 *   - Shadow pixels are excluded rather than misclassified as water.
 *   - Works for both 02_FO (open flood) and 03_FU (urban flood) chips.
 *
 * Output: Drive → water_detection_pseudolabels/
 *   {EventName}_awei_confidence.tif  (float32, [0,1] + -1 for shadow)
 *
 * Training integration:
 *   confidence > 0.7  → treat as flood label (1), weight = confidence
 *   confidence < 0.3  → treat as non-flood label (0), weight = 1-confidence
 *   0.3 <= conf <= 0.7 → ignore (-1), too uncertain
 *   confidence = -1   → ignore (-1), shadow
 */

var TRAIN_EVENTS = [
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

print('Queueing AWEI_sh confidence map exports for ' + TRAIN_EVENTS.length + ' events...');

TRAIN_EVENTS.forEach(function(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);

  // ── Get least-cloudy Sentinel-2 SR during flood ──────────────────────
  var s2col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.flood_start, event.flood_end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));

  // Use the least cloudy image; median composite as fallback
  var s2_best = s2col.sort('CLOUDY_PIXEL_PERCENTAGE').first();
  var s2_median = s2col.median();
  var s2_image = ee.Image(ee.Algorithms.If(
    s2col.size().gt(0), s2_best, s2_median
  ));

  // ── Compute AWEI_sh ───────────────────────────────────────────────────
  // Sentinel-2 SR bands scaled 0-10000; divide by 10000 for reflectance
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

  // ── Shadow mask ───────────────────────────────────────────────────────
  // Building shadows have: low NIR (no vegetation/surface reflectance),
  // low Blue (shaded), but can have moderate SWIR from indirect illumination.
  // Conservative shadow definition: NIR < 0.1 AND Blue < 0.05
  // These thresholds are validated for Sentinel-2 SR in urban scenes.
  var in_shadow = nir.lt(0.10).and(blue.lt(0.05));

  // ── Cloud mask ────────────────────────────────────────────────────────
  // Use Sentinel-2 SCL band (Scene Classification Layer) if available
  // SCL: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus
  var scl = s2_image.select('SCL');
  var is_cloud = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));

  // ── Map AWEI_sh to confidence ─────────────────────────────────────────
  // Linear ramp: AWEI_sh in [-0.3, +0.3] → confidence in [0, 1]
  // Clamp outside this range.
  var confidence = awei_sh
    .add(0.3)        // shift so 0 → 0.3, +0.3 → 0.6
    .divide(0.6)     // scale so [-0.3,+0.3] → [0,1]
    .clamp(0, 1)
    .toFloat();

  // Apply shadow mask: set to -1 (ignore)
  confidence = confidence.where(in_shadow, -1.0);

  // Apply cloud mask: set to -1 (ignore)
  confidence = confidence.where(is_cloud, -1.0);

  // Fill nodata with -1 (ignore)
  confidence = confidence.unmask(-1.0);

  Export.image.toDrive({
    image:          confidence,
    description:    event.name + '_awei_confidence',
    folder:         'water_detection_pseudolabels',
    fileNamePrefix: event.name + '_awei_confidence',
    region:         aoi.bounds(),
    scale:          10,        // 10m to match Sentinel-1 chip resolution
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  print('Queued: ' + event.name);
});

print('');
print('All ' + TRAIN_EVENTS.length + ' tasks queued.');
print('Run all tasks in the Tasks tab.');
print('Download to: data/pseudolabels/');
print('');
print('Confidence map interpretation:');
print('  -1.0 = ignore (shadow / cloud / nodata)');
print('   0.0 = confidently non-water (AWEI_sh <= -0.3)');
print('   0.5 = uncertain (AWEI_sh ≈ 0, near threshold)');
print('   1.0 = confidently water (AWEI_sh >= +0.3)');
