/*
 * 05_batch_pseudo_label_export.js  (v2 — full loop, no getInfo)
 *
 * Queues all 14 pseudo-label export pairs in a single run.
 * No EVENT_INDEX needed — runs all events automatically.
 *
 * Because we skip .getInfo() calls, GEE evaluates everything
 * server-side and queues all 28 tasks (14 SAR + 14 LABEL) at once.
 * Script completes in ~30 seconds. Exports run in parallel on GEE servers.
 *
 * After running:
 *   Tasks tab → click Run on each task (or use "Run all" if available)
 *   Files land in Drive → water_detection_pseudolabels/
 *
 * To inspect a specific event visually before exporting, use
 * 05b_single_event_qc.js (set EVENT_INDEX there).
 */

// ============================================================
// CONFIGURATION
// ============================================================

var CFG = {
  temporal_window_days:  5,
  cloud_threshold:       40,
  urban_buffer_m:        500,
  ndwi_flood_thresh:     0.10,
  ndwi_dry_thresh:      -0.15,
  building_height_m:     10,
  shadow_max_m:          80,
  resolution_m:          10,
  drive_folder:          'water_detection_pseudolabels',
  crs:                   'EPSG:4326',
};

var WORLDCOVER = ee.Image('ESA/WorldCover/v200/2021');
var BUILT_UP   = WORLDCOVER.eq(50);

var URBAN_ZONE_GLOBAL = BUILT_UP
  .focal_max({ radius: CFG.urban_buffer_m, units: 'meters',
               kernelType: 'circle' })
  .unmask(0);

// ============================================================
// EVENTS
// ============================================================

var EVENTS = [
  { name: 'UK_Yorkshire_2015',
    lon: -1.0815,   lat: 53.9590,  buffer_km: 50,
    start: '2015-12-26', end: '2016-01-05', grade: 'A' },
  { name: 'Zhengzhou_China_2021',
    lon: 113.6249,  lat: 34.7472,  buffer_km: 40,
    start: '2021-07-17', end: '2021-07-31', grade: 'A' },
  { name: 'Pakistan_Sindh_2022',
    lon: 68.7732,   lat: 27.5260,  buffer_km: 80,
    start: '2022-08-10', end: '2022-09-15', grade: 'A' },
  { name: 'Cyclone_Gabrielle_NZ_2023',
    lon: 176.9120,  lat: -39.4928, buffer_km: 50,
    start: '2023-02-12', end: '2023-02-22', grade: 'A' },
  { name: 'Jeddah_Saudi_2022',
    lon: 39.1925,   lat: 21.4858,  buffer_km: 30,
    start: '2022-11-24', end: '2022-12-03', grade: 'A' },
  { name: 'Bucharest_Romania_2024',
    lon: 26.1025,   lat: 44.4268,  buffer_km: 40,
    start: '2024-09-13', end: '2024-09-22', grade: 'A' },
  { name: 'Seville_Spain_2025',
    lon: -5.9845,   lat: 37.3891,  buffer_km: 40,
    start: '2025-01-28', end: '2025-02-05', grade: 'B' },
  { name: 'Cologne_Rhine_2021',
    lon: 6.9603,    lat: 50.9375,  buffer_km: 30,
    start: '2021-07-14', end: '2021-07-22', grade: 'B' },
  { name: 'Western_Europe_2021',
    lon: 6.8183,    lat: 50.5437,  buffer_km: 60,
    start: '2021-07-14', end: '2021-07-22', grade: 'B' },
  { name: 'Perth_Australia_2021',
    lon: 115.8605,  lat: -31.9505, buffer_km: 40,
    start: '2021-05-28', end: '2021-06-10', grade: 'B' },
  { name: 'Libya_Derna_2023',
    lon: 22.6389,   lat: 32.7570,  buffer_km: 20,
    start: '2023-09-10', end: '2023-09-20', grade: 'B' },
  { name: 'Seoul_South_Korea_2022',
    lon: 126.9780,  lat: 37.5665,  buffer_km: 40,
    start: '2022-08-08', end: '2022-08-15', grade: 'B' },
  { name: 'Typhoon_Vamco_Manila_2020',
    lon: 121.0437,  lat: 14.6760,  buffer_km: 40,
    start: '2020-11-11', end: '2020-11-18', grade: 'C' },
  { name: 'Hurricane_Florence_2018',
    lon: -77.9447,  lat: 34.2257,  buffer_km: 60,
    start: '2018-09-14', end: '2018-09-28', grade: 'C' },
];

// ============================================================
// PIPELINE FUNCTIONS (all server-side — no .getInfo())
// ============================================================

function findBestPair(aoi, event) {
  var windowMs = CFG.temporal_window_days * 24 * 3600 * 1000;

  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(event.start, event.end)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'));

  var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.start, event.end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CFG.cloud_threshold));

  var joined = ee.Join.saveAll({ matchesKey: 's2_matches' })
    .apply(
      s1Col, s2Col,
      ee.Filter.and(
        ee.Filter.intersects({ leftField: '.geo', rightField: '.geo' }),
        ee.Filter.maxDifference({
          difference:  windowMs,
          leftField:   'system:time_start',
          rightField:  'system:time_start'
        })
      )
    );

  var withBest = joined.map(function(s1img) {
    var matches = ee.ImageCollection.fromImages(s1img.get('s2_matches'));
    var bestS2  = matches.sort('CLOUDY_PIXEL_PERCENTAGE').first();
    return s1img
      .set('best_s2',      bestS2)
      .set('best_cloud',   bestS2.get('CLOUDY_PIXEL_PERCENTAGE'))
      .set('n_s2_matches', matches.size());
  });

  // Return the single best S1 image (lowest cloud S2 match)
  return ee.Image(
    withBest
      .filter(ee.Filter.gt('n_s2_matches', 0))
      .sort('best_cloud')
      .first()
  );
}

function computeInvalidMask(s2Image, exportRegion) {
  var scl = s2Image.select('SCL');
  var cloudMask = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));

  var solarElevDeg = ee.Number(90).subtract(
    ee.Number(s2Image.get('MEAN_SOLAR_ZENITH_ANGLE'))
  );
  var solarElevRad = solarElevDeg.multiply(Math.PI / 180);
  var shadowLen = ee.Number(CFG.building_height_m)
                    .divide(solarElevRad.tan().max(0.1))
                    .min(CFG.shadow_max_m);

  var shadowMask = BUILT_UP
    .focal_max({ radius: shadowLen, units: 'meters', kernelType: 'circle' })
    .clip(exportRegion)
    .unmask(0);

  return cloudMask.or(shadowMask).rename('invalid_mask');
}

function computePseudoLabel(s2Image, exportRegion, urbanZone, invalidMask) {
  var ndwi    = s2Image.normalizedDifference(['B3', 'B8']);
  var builtUp = BUILT_UP.clip(exportRegion).unmask(0);

  var isFlood    = ndwi.gt(CFG.ndwi_flood_thresh);
  var floodOpen  = isFlood.and(builtUp.not()).and(urbanZone).multiply(1);
  var floodUrban = isFlood.and(builtUp).multiply(2);
  var floodLabel = floodOpen.add(floodUrban);

  var isDry    = ndwi.lt(CFG.ndwi_dry_thresh);
  var nonFlood = isDry.and(urbanZone).and(isFlood.not()).multiply(3);

  var rawLabel = floodLabel
    .where(floodLabel.eq(0).and(nonFlood.gt(0)), nonFlood)
    .uint8();

  return rawLabel
    .where(invalidMask, 0)
    .where(urbanZone.not(), 0)
    .uint8()
    .rename('pseudo_label');
}

function exportEvent(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);

  // Best pair — fully server-side
  var bestS1 = findBestPair(aoi, event);
  var bestS2 = ee.Image(bestS1.get('best_s2'));

  // Export region: S1 ∩ S2 ∩ AOI
  var exportRegion = bestS1.geometry()
    .intersection(bestS2.geometry(), ee.ErrorMargin(10))
    .intersection(aoi, ee.ErrorMargin(10));

  var urbanZone   = URBAN_ZONE_GLOBAL.clip(exportRegion).unmask(0);
  var invalidMask = computeInvalidMask(bestS2, exportRegion);
  var pseudoLabel = computePseudoLabel(
    bestS2, exportRegion, urbanZone, invalidMask
  );
  var sarBands = bestS1
    .select(['VV', 'VH'])
    .clip(exportRegion)
    .updateMask(urbanZone)
    .toFloat();

  var bounds = exportRegion.bounds();

  Export.image.toDrive({
    image:          sarBands,
    description:    event.name + '_SAR',
    folder:         CFG.drive_folder,
    fileNamePrefix: event.name + '_SAR',
    region:         bounds,
    scale:          CFG.resolution_m,
    crs:            CFG.crs,
    maxPixels:      1e13,
    fileFormat:     'GeoTIFF',
    formatOptions:  { cloudOptimized: true }
  });

  Export.image.toDrive({
    image:          pseudoLabel,
    description:    event.name + '_LABEL',
    folder:         CFG.drive_folder,
    fileNamePrefix: event.name + '_LABEL',
    region:         bounds,
    scale:          CFG.resolution_m,
    crs:            CFG.crs,
    maxPixels:      1e13,
    fileFormat:     'GeoTIFF',
    formatOptions:  { cloudOptimized: true }
  });

  print('Queued [' + event.grade + ']: ' + event.name);
}

// ============================================================
// RUN ALL EVENTS
// ============================================================

print('Queueing exports for ' + EVENTS.length + ' events...');
print('(No progress bar — GEE evaluates server-side)');
print('');

EVENTS.forEach(function(event) {
  exportEvent(event);
});

print('');
print('Done. ' + (EVENTS.length * 2) + ' tasks queued.');
print('→ Tasks tab → Run all tasks');
print('→ Files will appear in Drive: ' + CFG.drive_folder + '/');
print('');
print('Expected output files:');
EVENTS.forEach(function(event) {
  print('  ' + event.name + '_SAR.tif');
  print('  ' + event.name + '_LABEL.tif');
});
