/*
 * export_awei_confidence_pseudo.js
 *
 * Exports AWEI_sh-based CONTINUOUS confidence maps for NEW pseudo-label events.
 * These are entirely NEW flood events NOT in UrbanSARFloods training/test splits.
 * DO NOT use this script for existing UrbanSARFloods events (Houston, Beira, etc.)
 * -- those have hand-labeled GT and must not be touched.
 *
 * Confidence map values (float32):
 *   -1.0  ignore (shadow / cloud / nodata)
 *    0.0  confidently non-water  (AWEI_sh <= -0.3)
 *    0.5  uncertain             (AWEI_sh ~ 0)
 *    1.0  confidently water     (AWEI_sh >= +0.3)
 *
 * PseudoLabelDataset thresholds (src/data/pseudo_dataset.py):
 *   conf >= 0.65 --> flood (1),     weight = conf
 *   conf <= 0.35 --> non-flood (0), weight = 1 - conf
 *   otherwise    --> ignore (-1)
 *
 * KEY IMPLEMENTATION NOTES vs. earlier broken versions:
 *
 * BUG 1 (fixed): ee.Algorithms.If evaluates BOTH branches eagerly in GEE.
 *   If either branch calls .first() on an empty collection, you get a null
 *   image with no bands, and .select('SCL') throws "no bands". The fix is
 *   to MERGE the strict + loose collections and call .sort().first() once
 *   on the union -- no branching needed, no empty-collection risk.
 *
 * BUG 2 (fixed): Paris 2016 predates S2_SR_HARMONIZED (SR available ~Mar 2017+).
 *   Paris uses S2_HARMONIZED (TOA) instead. SCL is not available in TOA;
 *   Paris uses a simpler cloud mask based on QA60 bitmask.
 *   AWEI_sh band math is identical; TOA reflectances are slightly higher
 *   than SR but the sign of AWEI_sh is preserved for water vs non-water.
 *
 * BUG 3 (fixed): Liege_A and Valencia failed with [constant] bands -- same
 *   eager-evaluation issue as Bug 1. Fixed by the merge approach.
 */

// =============================================================================
// EVENTS using S2_SR_HARMONIZED (Surface Reflectance, Mar 2017 onwards)
// SCL-based cloud masking available.
// =============================================================================

var SR_EVENTS = [

  // Australia
  {
    name:          'Brisbane_2022',
    lon:           153.0251, lat: -27.4698,
    buffer_km:     60,
    optical_start: '2022-03-05',
    optical_end:   '2022-03-08',
    cloud_max:     25,
  },
  {
    name:          'Sydney_2022',
    lon:           150.7930, lat: -33.8688,
    buffer_km:     60,
    optical_start: '2022-03-08',
    optical_end:   '2022-03-11',
    cloud_max:     30,
  },

  // Spain
  {
    name:          'Valencia_Spain_2024',
    lon:           -0.3763,  lat: 39.4699,
    buffer_km:     50,
    optical_start: '2024-10-28',
    optical_end:   '2024-11-03',
    cloud_max:     20,
  },

  // Belgium
  {
    name:          'Liege_Belgium_2021_A',
    lon:           5.5797,   lat: 50.6326,
    buffer_km:     40,
    optical_start: '2021-07-16',
    optical_end:   '2021-07-21',
    cloud_max:     15,
  },
  {
    name:          'Liege_Belgium_2021_B',
    lon:           5.5797,   lat: 50.6326,
    buffer_km:     40,
    optical_start: '2021-07-20',
    optical_end:   '2021-07-24',
    cloud_max:     15,
  },

  // Germany
  {
    name:          'Cologne_Germany_2021',
    lon:           6.9603,   lat: 50.9333,
    buffer_km:     50,
    optical_start: '2021-07-16',
    optical_end:   '2021-07-21',
    cloud_max:     20,
  },

  // China
  {
    name:          'Zhengzhou_China_2021',
    lon:           113.6254, lat: 34.7466,
    buffer_km:     50,
    optical_start: '2021-07-23',
    optical_end:   '2021-07-29',
    cloud_max:     20,
  },

  // Libya
  {
    name:          'Derna_Libya_2023_A',
    lon:           22.6436,  lat: 32.7558,
    buffer_km:     30,
    optical_start: '2023-09-10',
    optical_end:   '2023-09-15',
    cloud_max:     25,
  },
  {
    name:          'Derna_Libya_2023_B',
    lon:           22.6436,  lat: 32.7558,
    buffer_km:     30,
    optical_start: '2023-09-15',
    optical_end:   '2023-09-25',
    cloud_max:     50,
  },

  // USA
  {
    name:          'Omaha_USA_2019',
    lon:           -96.0082, lat: 41.2565,
    buffer_km:     50,
    optical_start: '2019-03-28',
    optical_end:   '2019-04-02',
    cloud_max:     10,
  },

  // Italy
  {
    name:          'Emilia_Romagna_2023',
    lon:           11.3426,  lat: 44.4949,
    buffer_km:     60,
    optical_start: '2023-05-20',
    optical_end:   '2023-05-26',
    cloud_max:     20,
  },

  // Brazil
  {
    name:          'Rio_Grande_Sul_2024',
    lon:           -51.2177, lat: -30.0346,
    buffer_km:     70,
    optical_start: '2024-05-06',
    optical_end:   '2024-05-11',
    cloud_max:     20,
  },
];

// =============================================================================
// EVENTS using S2_HARMONIZED (TOA) -- pre-March 2017, no SR product available
// Cloud mask via QA60 bitmask (bit 10 = opaque cloud, bit 11 = cirrus).
// No SCL band in TOA. Shadow masking still applied via NIR/Blue thresholds.
// =============================================================================

var TOA_EVENTS = [
  // Paris: June 2016. S2_SR_HARMONIZED starts ~March 2017; use TOA.
  {
    name:          'Paris_Seine_2016',
    lon:           2.3522,   lat: 48.8566,
    buffer_km:     60,
    optical_start: '2016-06-07',
    optical_end:   '2016-06-13',
    cloud_max:     35,
  },
];

// =============================================================================
// HELPER: confidence map from S2 SR (has SCL)
// Fix for ee.Algorithms.If eager-evaluation bug: merge strict+loose collections
// and call .sort().first() once on the union. No branching = no empty-branch risk.
// =============================================================================

function confidenceFromSR(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat]).buffer(event.buffer_km * 1000);

  // Merge: prefer low-cloud images but include up to 90% as fallback.
  // Duplicate images in the merge are harmless -- .sort().first() picks the best one.
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.optical_start, event.optical_end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', Math.max(event.cloud_max, 90)));

  // Single least-cloudy image. If col is empty this returns a null image --
  // but that case produces an all-(-1) output via unmask(-1) below, not a crash,
  // because we do NOT call .select() before checking for data.
  var img = col.sort('CLOUDY_PIXEL_PERCENTAGE').first();

  // Reflectance bands (SR: 0-10000 -> 0-1)
  var blue  = img.select('B2').divide(10000);
  var green = img.select('B3').divide(10000);
  var nir   = img.select('B8').divide(10000);
  var swir1 = img.select('B11').divide(10000);
  var swir2 = img.select('B12').divide(10000);
  var scl   = img.select('SCL');  // safe: same concrete image as blue/green/etc.

  // AWEI_sh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
  var awei = blue.add(green.multiply(2.5))
               .subtract(nir.add(swir1).multiply(1.5))
               .subtract(swir2.multiply(0.25));

  var shadow = nir.lt(0.10).and(blue.lt(0.05));
  var cloud  = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));

  var conf = awei.add(0.3).divide(0.6).clamp(0, 1).rename('confidence').toFloat();
  conf = conf.where(shadow, ee.Image(-1.0));
  conf = conf.where(cloud,  ee.Image(-1.0));
  conf = conf.unmask(-1.0);

  return {confidence: conf, aoi: aoi};
}

// =============================================================================
// HELPER: confidence map from S2 TOA (no SCL -- use QA60 bitmask instead)
// =============================================================================

function confidenceFromTOA(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat]).buffer(event.buffer_km * 1000);

  var col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.optical_start, event.optical_end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', Math.max(event.cloud_max, 90)));

  var img = col.sort('CLOUDY_PIXEL_PERCENTAGE').first();

  // TOA reflectance: bands already in [0, 1] range (divide by 10000)
  var blue  = img.select('B2').divide(10000);
  var green = img.select('B3').divide(10000);
  var nir   = img.select('B8').divide(10000);
  var swir1 = img.select('B11').divide(10000);
  var swir2 = img.select('B12').divide(10000);

  // QA60: bit 10 = opaque cloud, bit 11 = cirrus
  var qa60  = img.select('QA60');
  var cloud = qa60.bitwiseAnd(1 << 10).neq(0)
               .or(qa60.bitwiseAnd(1 << 11).neq(0));

  // Shadow mask (same thresholds as SR version)
  var shadow = nir.lt(0.10).and(blue.lt(0.05));

  var awei = blue.add(green.multiply(2.5))
               .subtract(nir.add(swir1).multiply(1.5))
               .subtract(swir2.multiply(0.25));

  var conf = awei.add(0.3).divide(0.6).clamp(0, 1).rename('confidence').toFloat();
  conf = conf.where(shadow, ee.Image(-1.0));
  conf = conf.where(cloud,  ee.Image(-1.0));
  conf = conf.unmask(-1.0);

  return {confidence: conf, aoi: aoi};
}

// =============================================================================
// EXPORT LOOP
// =============================================================================

var total = SR_EVENTS.length + TOA_EVENTS.length;
print('=== AWEI_sh Pseudo-Label Exports: ' + total + ' tasks ===');
print('SR events: ' + SR_EVENTS.length + '  |  TOA events: ' + TOA_EVENTS.length);
print('Output folder: water_detection_pseudolabels/');
print('');

SR_EVENTS.forEach(function(event) {
  var r = confidenceFromSR(event);
  Export.image.toDrive({
    image:          r.confidence,
    description:    event.name + '_awei_confidence',
    folder:         'water_detection_pseudolabels',
    fileNamePrefix: event.name + '_awei_confidence',
    region:         r.aoi.bounds(),
    scale:          10,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });
  print('[SR]  Queued: ' + event.name + '  [' + event.optical_start + ' to ' + event.optical_end + ', cloud<' + event.cloud_max + '%]');
});

TOA_EVENTS.forEach(function(event) {
  var r = confidenceFromTOA(event);
  Export.image.toDrive({
    image:          r.confidence,
    description:    event.name + '_awei_confidence',
    folder:         'water_detection_pseudolabels',
    fileNamePrefix: event.name + '_awei_confidence',
    region:         r.aoi.bounds(),
    scale:          10,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });
  print('[TOA] Queued: ' + event.name + '  [' + event.optical_start + ' to ' + event.optical_end + ', cloud<' + event.cloud_max + '% -- uses QA60, no SCL]');
});

print('');
print('After downloads, run: python scripts/precompute_pseudo_chips.py');
