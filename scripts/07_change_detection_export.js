/*
 * 07_change_detection_export.js
 *
 * Pseudo-label generation using NDWI change detection.
 *
 * Previous approach (WRONG):
 *   flood_label = NDWI_during > 0.10
 *   Problem: labels permanent rivers and coastlines, not actual flooding.
 *
 * Correct approach:
 *   baseline_NDWI = median composite of dry-season S2 (same area, prior year)
 *   flood_label   = (NDWI_during - baseline_NDWI) > DELTA_THRESHOLD
 *                   AND NDWI_during > ABS_THRESHOLD
 *
 * This only labels pixels where water APPEARED — permanent water bodies
 * show near-zero delta and are excluded. Newly flooded land shows
 * strongly positive delta.
 *
 * Thresholds:
 *   DELTA_THRESHOLD: 0.15 — NDWI must increase by at least 0.15 from baseline
 *   ABS_THRESHOLD:   0.05 — during-event NDWI must be positive (actual water)
 *   DRY_DELTA:      -0.10 — NDWI decreased from baseline → confident non-flood
 *
 * Baseline period:
 *   Dry season composite 90-365 days before flood event.
 *   Uses median of all cloud-free S2 pixels to get stable baseline.
 *   Cloud threshold for baseline: 20% (stricter — we want clean dry pixels)
 *
 * Final 9 confirmed events:
 *   UK_Yorkshire_2015, Pakistan_Sindh_2022, Seville_Spain_2025,
 *   Libya_Derna_2023, Hurricane_Florence_2018, Cologne_Rhine_2021,
 *   Western_Europe_2021, Seoul_South_Korea_2022,
 *   Cyclone_Gabrielle_NZ_2023
 */

// ============================================================
// CONFIGURATION
// ============================================================

var CFG = {
  // Pair selection
  cloud_threshold_flood:   60,   // generous during storm events
  cloud_threshold_baseline: 20,  // strict for dry baseline
  temporal_window_days:     3,   // max S1-S2 gap

  // Change detection thresholds
  ndwi_delta_flood:   0.15,  // NDWI must increase by this much from baseline
  ndwi_abs_flood:     0.05,  // during-event NDWI must be at least this
  ndwi_delta_dry:    -0.10,  // NDWI decreased → confident non-flood

  // Baseline window
  baseline_days_min:   90,   // baseline starts at least 90 days before flood
  baseline_days_max:  365,   // baseline ends 365 days before flood

  // Urban constraint
  urban_buffer_m:     500,

  // Shadow masking
  building_height_m:   10,
  shadow_max_m:        80,

  // Export
  resolution_m:        10,
  drive_folder:       'water_detection_pseudolabels_v2',
  crs:                'EPSG:4326',
};

var WORLDCOVER = ee.Image('ESA/WorldCover/v200/2021');
var BUILT_UP   = WORLDCOVER.eq(50);
var URBAN_ZONE_GLOBAL = BUILT_UP
  .focal_max({ radius: CFG.urban_buffer_m, units: 'meters',
               kernelType: 'circle' })
  .unmask(0);

// Permanent water mask from JRC — exclude from training entirely
// These pixels are water regardless of flood and add no signal
var JRC_WATER = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence')
  .gt(50)  // water present >50% of observations = permanent water
  .unmask(0);

// ============================================================
// EVENTS — final 9 confirmed with good timing
// baseline_start/end define the dry-season composite window
// ============================================================

var EVENTS = [
  {
    name: 'UK_Yorkshire_2015',
    lon: -1.0815, lat: 53.9590, buffer_km: 50,
    flood_start: '2015-12-26', flood_end: '2016-01-05',
    // S2 not operational over UK before Dec 2015 — use post-event summer
    baseline_windows: [
      { start: '2016-06-01', end: '2016-09-30' },  // first summer after flood
      { start: '2017-06-01', end: '2017-09-30' },  // fallback
    ],
    notes: 'S2 launched mid-2015; no pre-event baseline available'
  },
  {
    name: 'Pakistan_Sindh_2022',
    lon: 68.7732, lat: 27.5260, buffer_km: 80,
    flood_start: '2022-08-10', flood_end: '2022-09-15',
    baseline_windows: [
      { start: '2022-02-01', end: '2022-05-31' },  // dry season before flood
      { start: '2021-10-01', end: '2022-01-31' },  // fallback: post-monsoon
    ],
    notes: 'Indus overflow; dry season Feb-May reliable in Sindh'
  },
  {
    name: 'Seville_Spain_2025',
    lon: -5.9845, lat: 37.3891, buffer_km: 40,
    flood_start: '2025-01-28', flood_end: '2025-02-05',
    baseline_windows: [
      { start: '2024-06-01', end: '2024-09-30' },  // dry Andalusian summer
      { start: '2023-06-01', end: '2023-09-30' },  // fallback
    ],
    notes: 'Mediterranean climate; summer baseline very clean'
  },
  {
    name: 'Libya_Derna_2023',
    lon: 22.6389, lat: 32.7570, buffer_km: 20,
    flood_start: '2023-09-10', flood_end: '2023-09-20',
    baseline_windows: [
      { start: '2023-03-01', end: '2023-07-31' },  // arid — any non-storm period
      { start: '2022-03-01', end: '2022-07-31' },  // fallback
    ],
    notes: 'Arid baseline; delta will be very clear over dam-breach water'
  },
  {
    name: 'Hurricane_Florence_2018',
    lon: -77.9447, lat: 34.2257, buffer_km: 60,
    flood_start: '2018-09-14', flood_end: '2018-09-28',
    baseline_windows: [
      { start: '2018-05-01', end: '2018-08-31' },  // pre-hurricane dry summer
      { start: '2017-05-01', end: '2017-08-31' },  // fallback prior year
    ],
    notes: 'Carolinas; weeks of riverine flooding post-landfall'
  },
  {
    name: 'Cologne_Rhine_2021',
    lon: 6.9603, lat: 50.9375, buffer_km: 30,
    flood_start: '2021-07-14', flood_end: '2021-07-22',
    baseline_windows: [
      { start: '2021-03-01', end: '2021-06-30' },  // spring before flood
      { start: '2020-06-01', end: '2020-09-30' },  // fallback prior year
    ],
    notes: 'Rhine at peak Jul 15-17'
  },
  {
    name: 'Western_Europe_2021',
    lon: 6.8183, lat: 50.5437, buffer_km: 60,
    flood_start: '2021-07-14', flood_end: '2021-07-22',
    baseline_windows: [
      { start: '2021-03-01', end: '2021-06-30' },
      { start: '2020-06-01', end: '2020-09-30' },
    ],
    notes: 'Ahr Valley; same event as Cologne'
  },
  {
    name: 'Seoul_South_Korea_2022',
    lon: 126.9780, lat: 37.5665, buffer_km: 40,
    flood_start: '2022-08-07', flood_end: '2022-08-13',
    baseline_windows: [
      { start: '2022-04-01', end: '2022-06-30' },  // pre-monsoon spring
      { start: '2021-04-01', end: '2021-06-30' },  // fallback prior year
    ],
    notes: 'Peak-window pair confirmed; Aug 9 S1 / Aug 10 S2'
  },
  {
    name: 'Cyclone_Gabrielle_NZ_2023',
    lon: 176.9120, lat: -39.4928, buffer_km: 50,
    flood_start: '2023-02-13', flood_end: '2023-02-22',
    baseline_windows: [
      { start: '2022-11-01', end: '2023-01-31' },  // NZ summer — dry for Hawkes Bay
      { start: '2021-11-01', end: '2022-01-31' },  // fallback prior year
    ],
    notes: 'Riverine component persisted; Feb 20/21 pair'
  },
];

// ============================================================
// STEP 1: FIND BEST FLOOD PAIR
// ============================================================

function findFloodPair(aoi, event) {
  var windowMs = CFG.temporal_window_days * 24 * 3600 * 1000;

  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(event.flood_start, event.flood_end)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'));

  var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.flood_start, event.flood_end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CFG.cloud_threshold_flood));

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
    return img
      .set('best_s2',      bestS2)
      .set('best_cloud',   bestS2.get('CLOUDY_PIXEL_PERCENTAGE'))
      .set('n_s2_matches', matches.size());
  });

  return withBest
    .filter(ee.Filter.gt('n_s2_matches', 0))
    .sort('best_cloud');
}

// ============================================================
// STEP 2: BUILD DRY BASELINE
// Median composite of cloud-free S2 in dry season
// ============================================================

function buildBaseline(aoi, event) {
  var MIN_IMAGES = 3;
  var chosen = null;
  var nChosen = 0;

  // Try each baseline window in order until we find enough images
  for (var i = 0; i < event.baseline_windows.length; i++) {
    var w = event.baseline_windows[i];
    var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi)
      .filterDate(w.start, w.end)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50));

    var n = col.size().getInfo();
    print('Baseline window ' + (i+1) + ' (' + w.start + ' – ' + w.end + '): ' + n + ' images');

    if (n >= MIN_IMAGES) {
      chosen   = col;
      nChosen  = n;
      print('Using window ' + (i+1) + ' for baseline.');
      break;
    }
  }

  if (!chosen) {
    // Last resort: combine all windows together
    print('WARNING: No window had ' + MIN_IMAGES + '+ images. Merging all windows.');
    var all = ee.ImageCollection([]);
    for (var j = 0; j < event.baseline_windows.length; j++) {
      var w = event.baseline_windows[j];
      all = all.merge(
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(aoi)
          .filterDate(w.start, w.end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
      );
    }
    chosen  = all;
    nChosen = chosen.size().getInfo();
    print('Merged baseline images: ' + nChosen);
  }

  // Per-pixel cloud masking before median
  var masked = chosen.map(function(img) {
    var scl = img.select('SCL');
    var cloud = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
    return img.updateMask(cloud.not());
  });

  return masked.median();
}

// ============================================================
// STEP 3: CHANGE DETECTION LABEL
//
// delta_NDWI = NDWI_flood - NDWI_baseline
//
// flood_label:
//   delta > DELTA_THRESHOLD AND NDWI_flood > ABS_THRESHOLD
//   → water appeared here during the flood
//
// non_flood_label:
//   delta < DRY_DELTA AND within urban zone
//   → area was already dry and got drier (or stayed dry)
//
// masked (label=0):
//   permanent water body (JRC)
//   cloud or optical shadow
//   outside urban buffer
//   ambiguous delta
// ============================================================

function computeChangeLabel(s2Flood, baseline, exportRegion,
                             urbanZone, invalidMask) {
  var ndwiFlood    = s2Flood.normalizedDifference(['B3', 'B8']);
  var ndwiBaseline = baseline.normalizedDifference(['B3', 'B8']);

  // Where baseline has no data (cloud gaps), use 0 as conservative estimate
  ndwiBaseline = ndwiBaseline.unmask(0);

  var delta = ndwiFlood.subtract(ndwiBaseline).rename('delta_NDWI');

  // Flood: water appeared (large positive delta + positive absolute NDWI)
  var isFlood = delta.gt(CFG.ndwi_delta_flood)
                     .and(ndwiFlood.gt(CFG.ndwi_abs_flood))
                     .and(JRC_WATER.not());  // exclude permanent water

  // Non-flood dry: NDWI didn't increase (stayed dry or got drier)
  var isDry = delta.lt(CFG.ndwi_delta_dry)
                   .and(ndwiFlood.lt(0.1))
                   .and(urbanZone);

  var builtUp = BUILT_UP.clip(exportRegion).unmask(0);

  // Split flood by land cover (for compatibility with UrbanSARFloods)
  var floodOpen  = isFlood.and(builtUp.not()).and(urbanZone).multiply(1);
  var floodUrban = isFlood.and(builtUp).multiply(2);
  var floodLabel = floodOpen.add(floodUrban);

  var nonFlood = isDry.and(isFlood.not()).multiply(3);

  var rawLabel = floodLabel
    .where(floodLabel.eq(0).and(nonFlood.gt(0)), nonFlood)
    .uint8();

  return rawLabel
    .where(invalidMask, 0)
    .where(urbanZone.not(), 0)
    .where(JRC_WATER, 0)     // always mask permanent water
    .uint8()
    .rename('pseudo_label');
}

// ============================================================
// STEP 4: INVALID MASK (cloud + optical shadow)
// ============================================================

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
    .clip(exportRegion).unmask(0);
  return cloudMask.or(shadowMask).rename('invalid_mask');
}

// ============================================================
// MAIN — change EVENT_INDEX to process each event
// ============================================================

var EVENT_INDEX = 0;
var event = EVENTS[EVENT_INDEX];

print('================================================');
print('Change-detection pseudo-label export');
print('Event: ' + event.name);
print('Flood window: ' + event.flood_start + ' → ' + event.flood_end);
print('Baseline windows: ' + event.baseline_windows.length + ' configured');
print('================================================');

var aoi = ee.Geometry.Point([event.lon, event.lat])
            .buffer(event.buffer_km * 1000);

// Build baseline with automatic fallback
var baseline = buildBaseline(aoi, event);

// Find flood pair
var pairedCol = findFloodPair(aoi, event);
var nPairs    = pairedCol.size().getInfo();
print('Flood pairs found: ' + nPairs);

if (nPairs === 0) {
  print('ERROR: No flood pairs. Increase cloud_threshold_flood.');
} else {
  var bestS1 = ee.Image(pairedCol.first());
  var bestS2 = ee.Image(bestS1.get('best_s2'));

  print('S1 date:    ' + bestS1.date().format('YYYY-MM-dd').getInfo());
  print('S2 date:    ' + bestS2.date().format('YYYY-MM-dd').getInfo());
  print('Cloud %:    ' + ee.Number(bestS1.get('best_cloud')).round().getInfo());

  var exportRegion = bestS1.geometry()
    .intersection(bestS2.geometry(), ee.ErrorMargin(10))
    .intersection(aoi, ee.ErrorMargin(10));

  var areaKm2 = exportRegion.area(10).divide(1e6).round().getInfo();
  print('Export region km²: ' + areaKm2);

  if (areaKm2 < 1) {
    print('ERROR: Export region too small — no spatial overlap.');
  } else {
    var urbanZone   = URBAN_ZONE_GLOBAL.clip(exportRegion).unmask(0);
    var invalidMask = computeInvalidMask(bestS2, exportRegion);
    var pseudoLabel = computeChangeLabel(
      bestS2, baseline, exportRegion, urbanZone, invalidMask
    );
    var sarBands = bestS1.select(['VV', 'VH'])
      .clip(exportRegion).updateMask(urbanZone).toFloat();

    // Label distribution diagnostic
    var dist = pseudoLabel.reduceRegion({
      reducer: ee.Reducer.frequencyHistogram(),
      geometry: exportRegion, scale: 100,
      maxPixels: 1e12, bestEffort: true
    });
    print('Label distribution (0=masked 1=flood_open 2=flood_urban 3=dry):');
    print(dist);

    // Delta NDWI stats — confirms change detection is working
    var ndwiFlood    = bestS2.normalizedDifference(['B3', 'B8']);
    var ndwiBaseline = baseline.normalizedDifference(['B3', 'B8']).unmask(0);
    var delta        = ndwiFlood.subtract(ndwiBaseline);
    var deltaStats   = delta.reduceRegion({
      reducer: ee.Reducer.percentile([5, 50, 95]),
      geometry: exportRegion, scale: 100,
      maxPixels: 1e12, bestEffort: true
    });
    print('Delta NDWI (p5 / median / p95):');
    print(deltaStats);
    print('(p95 >> 0.15 means significant new water detected)');

    // Export
    Export.image.toDrive({
      image: sarBands,
      description: event.name + '_SAR',
      folder: CFG.drive_folder,
      fileNamePrefix: event.name + '_SAR',
      region: exportRegion.bounds(),
      scale: CFG.resolution_m, crs: CFG.crs,
      maxPixels: 1e13, fileFormat: 'GeoTIFF',
      formatOptions: { cloudOptimized: true }
    });
    Export.image.toDrive({
      image: pseudoLabel,
      description: event.name + '_LABEL',
      folder: CFG.drive_folder,
      fileNamePrefix: event.name + '_LABEL',
      region: exportRegion.bounds(),
      scale: CFG.resolution_m, crs: CFG.crs,
      maxPixels: 1e13, fileFormat: 'GeoTIFF',
      formatOptions: { cloudOptimized: true }
    });

    print('Tasks queued → Tasks tab → Run');

    // Visualization
    Map.centerObject(exportRegion, 11);
    Map.addLayer(exportRegion,
      { color: '0000FF', fillColor: '00000000' }, 'Export region');
    Map.addLayer(bestS1.select('VV').clip(exportRegion),
      { min: -25, max: 0, palette: ['black', 'white'] }, 'SAR VV');
    Map.addLayer(bestS2.select(['B4','B3','B2']).clip(exportRegion),
      { min: 0, max: 3000 }, 'S2 RGB', false);
    Map.addLayer(ndwiFlood.clip(exportRegion),
      { min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue'] },
      'NDWI flood', false);
    Map.addLayer(delta.clip(exportRegion),
      { min: -0.3, max: 0.3, palette: ['red', 'white', 'blue'] },
      'Delta NDWI (change)');
    Map.addLayer(pseudoLabel.clip(exportRegion),
      { min: 0, max: 3, palette: ['black', 'cyan', 'orange', 'green'] },
      'Labels (0=masked 1=flood_open 2=flood_urban 3=dry)');
    Map.addLayer(JRC_WATER.clip(exportRegion).selfMask(),
      { palette: ['navy'] }, 'Permanent water (JRC) — excluded', false);
  }
}

/*
 * EVENT INDEX REFERENCE:
 *   0  UK_Yorkshire_2015
 *   1  Pakistan_Sindh_2022
 *   2  Seville_Spain_2025
 *   3  Libya_Derna_2023
 *   4  Hurricane_Florence_2018
 *   5  Cologne_Rhine_2021
 *   6  Western_Europe_2021
 *   7  Seoul_South_Korea_2022
 *   8  Cyclone_Gabrielle_NZ_2023
 *
 * QC CHECKLIST:
 * 1. Toggle 'Delta NDWI' layer — blue areas = new water appeared.
 *    These should be in floodplains, streets, parks — NOT in the river channel.
 *    The river channel should appear WHITE (near-zero delta — was always there).
 * 2. Toggle 'Permanent water (JRC)' — navy = masked out.
 *    River channels and coastlines should be navy, confirming they're excluded.
 * 3. Toggle 'Labels' — cyan/orange should be OUTSIDE the river channel,
 *    in areas that are normally dry land.
 * 4. Check console: p95 of delta NDWI should be > 0.15 for flood events.
 *    If p95 < 0.10, flood signal is weak — consider adjusting dates.
 * 5. If QC passes → Tasks tab → Run.
 * 6. Change EVENT_INDEX and repeat.
 */
