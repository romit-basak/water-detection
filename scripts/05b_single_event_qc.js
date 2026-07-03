/*
 * 05b_single_event_qc.js
 *
 * Visual QC for a single event before or after export.
 * Set EVENT_NAME to any event from the EVENTS list in
 * 05_batch_pseudo_label_export.js.
 *
 * Layers shown:
 *   - SAR VV (dB)
 *   - S2 RGB (true color)
 *   - Urban zone (500m buffer)
 *   - Pseudo-labels (0=masked, 1=open, 2=urban, 3=dry)
 *   - Invalid mask (cloud + shadow)
 */

// Paste config and pipeline functions from 05_batch_pseudo_label_export.js
// then set EVENT_NAME below and run.

var CFG = {
  temporal_window_days:  5,
  cloud_threshold:       40,
  urban_buffer_m:        500,
  ndwi_flood_thresh:     0.10,
  ndwi_dry_thresh:      -0.15,
  building_height_m:     10,
  shadow_max_m:          80,
  resolution_m:          10,
};

var WORLDCOVER = ee.Image('ESA/WorldCover/v200/2021');
var BUILT_UP   = WORLDCOVER.eq(50);
var URBAN_ZONE_GLOBAL = BUILT_UP
  .focal_max({ radius: CFG.urban_buffer_m, units: 'meters',
               kernelType: 'circle' })
  .unmask(0);

// ── Change this to inspect any event ──
var EVENT = {
  name: 'UK_Yorkshire_2015',
  lon: -1.0815, lat: 53.9590, buffer_km: 50,
  start: '2015-12-26', end: '2016-01-05', grade: 'A'
};
// ──────────────────────────────────────

var windowMs = CFG.temporal_window_days * 24 * 3600 * 1000;
var aoi = ee.Geometry.Point([EVENT.lon, EVENT.lat]).buffer(EVENT.buffer_km * 1000);

var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi).filterDate(EVENT.start, EVENT.end)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'));

var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate(EVENT.start, EVENT.end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CFG.cloud_threshold));

var joined = ee.Join.saveAll({ matchesKey: 's2_matches' })
  .apply(s1Col, s2Col, ee.Filter.and(
    ee.Filter.intersects({ leftField: '.geo', rightField: '.geo' }),
    ee.Filter.maxDifference({
      difference: windowMs,
      leftField: 'system:time_start', rightField: 'system:time_start'
    })
  ));

var withBest = joined.map(function(img) {
  var matches = ee.ImageCollection.fromImages(img.get('s2_matches'));
  var bestS2  = matches.sort('CLOUDY_PIXEL_PERCENTAGE').first();
  return img.set('best_s2', bestS2)
            .set('best_cloud', bestS2.get('CLOUDY_PIXEL_PERCENTAGE'))
            .set('n_s2_matches', matches.size());
});

var bestS1 = ee.Image(withBest.filter(ee.Filter.gt('n_s2_matches', 0))
                               .sort('best_cloud').first());
var bestS2 = ee.Image(bestS1.get('best_s2'));

print('S1:', bestS1.date().format('YYYY-MM-dd').getInfo());
print('S2:', bestS2.date().format('YYYY-MM-dd').getInfo());
print('Cloud %:', ee.Number(bestS1.get('best_cloud')).round().getInfo());

var exportRegion = bestS1.geometry()
  .intersection(bestS2.geometry(), ee.ErrorMargin(10))
  .intersection(aoi, ee.ErrorMargin(10));

print('Export region km²:', exportRegion.area(10).divide(1e6).round().getInfo());

var urbanZone = URBAN_ZONE_GLOBAL.clip(exportRegion).unmask(0);

var scl = bestS2.select('SCL');
var cloudMask = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
var solarElev = ee.Number(90).subtract(ee.Number(bestS2.get('MEAN_SOLAR_ZENITH_ANGLE')))
                  .multiply(Math.PI / 180);
var shadowLen = ee.Number(CFG.building_height_m).divide(solarElev.tan().max(0.1)).min(80);
var shadowMask = BUILT_UP.focal_max({ radius: shadowLen, units: 'meters', kernelType: 'circle' })
                          .clip(exportRegion).unmask(0);
var invalidMask = cloudMask.or(shadowMask);

var ndwi    = bestS2.normalizedDifference(['B3', 'B8']);
var builtUp = BUILT_UP.clip(exportRegion).unmask(0);
var isFlood = ndwi.gt(CFG.ndwi_flood_thresh);
var isDry   = ndwi.lt(CFG.ndwi_dry_thresh);
var pseudoLabel = ee.Image(0).uint8()
  .where(isFlood.and(builtUp.not()).and(urbanZone).and(invalidMask.not()), 1)
  .where(isFlood.and(builtUp).and(invalidMask.not()), 2)
  .where(isDry.and(urbanZone).and(isFlood.not()).and(invalidMask.not()), 3)
  .where(urbanZone.not(), 0)
  .rename('pseudo_label');

var dist = pseudoLabel.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: exportRegion, scale: 100,
  maxPixels: 1e12, bestEffort: true
});
print('Label distribution (0=masked, 1=open, 2=urban, 3=dry):', dist);

Map.centerObject(exportRegion, 11);
Map.addLayer(exportRegion, { color: '0000FF', fillColor: '00000000' }, 'Export region');
Map.addLayer(urbanZone.selfMask(), { palette: ['yellow'] }, 'Urban zone', false);
Map.addLayer(bestS1.select('VV').clip(exportRegion),
  { min: -25, max: 0, palette: ['black', 'white'] }, 'SAR VV');
Map.addLayer(bestS2.select(['B4','B3','B2']).clip(exportRegion),
  { min: 0, max: 3000 }, 'S2 RGB', false);
Map.addLayer(pseudoLabel.clip(exportRegion),
  { min: 0, max: 3, palette: ['black', 'cyan', 'orange', 'green'] },
  'Labels (0=masked 1=open 2=urban 3=dry)');
Map.addLayer(invalidMask.clip(exportRegion).selfMask(),
  { palette: ['red'] }, 'Invalid mask', false);
