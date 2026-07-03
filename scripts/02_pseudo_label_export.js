/*
 * 02_pseudo_label_export.js  (v3 — urban-constrained)
 *
 * Exports co-registered SAR + pseudo-label pairs for:
 *   1. UK_Yorkshire_2015      — Grade A, simultaneous S1+S2, riverine
 *   2. Zhengzhou_2021         — Grade A, 365 urban km², Chinese megacity
 *   3. Western_Europe_2021    — Grade B, 103 urban km², Ahr Valley
 *   4. Typhoon_Vamco_2020     — Grade C, 804 urban km², Metro Manila
 *
 * Urban constraint (v3 addition):
 *   All exported pixels are within URBAN_BUFFER_M of WorldCover built-up.
 *   This ensures we're training on urban SAR signatures, not farmland.
 *   Non-urban pixels are masked to label=0 and suppressed in SAR export.
 *
 * Label encoding (matches UrbanSARFloods exactly):
 *   0 = background / masked
 *       (clouds, optical shadow, non-urban, ambiguous NDWI, no-data)
 *   1 = flooded open area    (NDWI > thresh, non-built-up but urban-adjacent)
 *   2 = flooded urban area   (NDWI > thresh, built-up land cover)
 *
 * Why urban-adjacent open areas get label 1 (not masked):
 *   Flooded parks, car parks, and open lots adjacent to buildings are
 *   a real part of urban flood scenes and appear in UrbanSARFloods labels.
 *   We keep them but only within the urban buffer zone.
 *
 * Non-flood labels:
 *   Pixels within the urban buffer, not clouded, not shadowed, with
 *   NDWI < ndwi_dry_thresh are labeled as explicit non-flood.
 *   These are label=1 in the non-flooded chips (01_NF equivalent) —
 *   important for giving the model a negative training signal.
 *   Without this, the model sees almost no non-flood urban examples
 *   in pseudo-label training.
 *
 * Run: set EVENT_INDEX, run script, do visual QC, then queue Tasks.
 */

// ============================================================
// CONFIGURATION
// ============================================================

var CFG = {
  // Pair selection
  temporal_window_days:  5,
  cloud_threshold:       40,

  // Urban constraint
  urban_buffer_m:        500,   // export pixels within 500m of built-up
                                // captures urban fringe flooding

  // NDWI thresholds
  ndwi_flood_thresh:     0.15,  // > this → flood
  ndwi_dry_thresh:      -0.15,  // < this → confident non-flood (explicit label)
  // Between -0.15 and 0.15 → ambiguous → label 0

  // Optical shadow
  building_height_m:     10,
  shadow_max_m:          80,

  // Export
  resolution_m:          10,
  drive_folder:          'water_detection_pseudolabels',
  crs:                   'EPSG:4326',
};

// WorldCover 2021
var WORLDCOVER = ee.Image('ESA/WorldCover/v200/2021');
var BUILT_UP   = WORLDCOVER.eq(50);  // built-up class

// Urban mask: built-up pixels dilated by urban_buffer_m
// This is the spatial constraint applied to ALL exports.
// Pre-computed once at the top level for efficiency.
var URBAN_ZONE = BUILT_UP
  .focal_max({ radius: CFG.urban_buffer_m, units: 'meters',
               kernelType: 'circle' })
  .unmask(0);

// ============================================================
// EVENTS
// ============================================================

var EVENTS = [
  {
    index: 0,
    name:        'UK_Yorkshire_2015',
    lon:         -1.0815, lat: 53.9590, buffer_km: 50,
    start:       '2015-12-26', end: '2016-01-05',
    notes:       'Grade A — simultaneous S1+S2 Dec 29 at peak flooding'
  },
  {
    index: 1,
    name:        'Zhengzhou_2021',
    lon:         113.6249, lat: 34.7472, buffer_km: 40,
    start:       '2021-07-17', end: '2021-07-31',
    notes:       'Grade A — 365 urban km², Chinese megacity, 36% urban fraction'
  },
  {
    index: 2,
    name:        'Western_Europe_2021',
    lon:         6.8183, lat: 50.5437, buffer_km: 60,
    start:       '2021-07-14', end: '2021-07-22',
    notes:       'Grade B — 103 urban km², Ahr Valley, EMSR517-520 validated'
  },
  {
    index: 3,
    name:        'Typhoon_Vamco_2020',
    lon:         121.0437, lat: 14.6760, buffer_km: 40,
    start:       '2020-11-11', end: '2020-11-18',
    notes:       'Grade C — 804 urban km², Metro Manila, 29% cloud acceptable'
  },
];

// ============================================================
// STEP 1: FIND BEST SPATIALLY-OVERLAPPING S1+S2 PAIR
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
    var gapDays = ee.Number(s1img.get('system:time_start'))
                    .subtract(ee.Number(bestS2.get('system:time_start')))
                    .abs().divide(86400000);
    return s1img
      .set('best_s2',      bestS2)
      .set('best_cloud',   bestS2.get('CLOUDY_PIXEL_PERCENTAGE'))
      .set('gap_days',     gapDays)
      .set('n_s2_matches', matches.size());
  });

  return withBest
    .filter(ee.Filter.gt('n_s2_matches', 0))
    .sort('best_cloud');
}

// ============================================================
// STEP 2: URBAN ZONE MASK
// Returns a binary mask: 1 = within urban_buffer_m of built-up, 0 = rural
// This is the primary spatial quality gate.
// ============================================================

function getUrbanZone(exportRegion) {
  return URBAN_ZONE.clip(exportRegion).unmask(0);
}

// ============================================================
// STEP 3: CLOUD + OPTICAL SHADOW MASK
// ============================================================

function computeInvalidMask(s2Image, exportRegion) {
  // Cloud mask from SCL
  var scl = s2Image.select('SCL');
  var cloudMask = scl.eq(3)
    .or(scl.eq(8))
    .or(scl.eq(9))
    .or(scl.eq(10))
    .rename('cloud_mask');

  // Optical shadow: dilate built-up in anti-solar direction
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
    .unmask(0)
    .rename('shadow_mask');

  return cloudMask.or(shadowMask).rename('invalid_mask');
}

// ============================================================
// STEP 4: PSEUDO-LABEL GENERATION
//
// Label logic applied in order of priority:
//   1. Outside urban zone                    → 0 (masked, rural, skip)
//   2. Cloud or optical shadow               → 0 (masked, unreliable NDWI)
//   3. NDWI > flood_thresh + built-up        → 2 (flooded urban)
//   4. NDWI > flood_thresh + non-built-up    → 1 (flooded open, urban-adjacent)
//   5. NDWI < dry_thresh (any urban pixel)   → 3 (non-flood — explicit negative)
//   6. Ambiguous NDWI (-0.15 to 0.15)        → 0 (masked, uncertain)
//
// Label 3 (non-flood) is remapped to match UrbanSARFloods encoding later.
// In UrbanSARFloods, non-flooded pixels in 01_NF chips have label 0 (bg).
// We use a temporary label 3 here to distinguish "confident non-flood"
// from "ambiguous/masked" during generation, then decide in Python
// whether to keep it or collapse to 0.
// ============================================================

function computePseudoLabel(s2Image, exportRegion, urbanZone, invalidMask) {
  var ndwi = s2Image.normalizedDifference(['B3', 'B8']);

  var builtUp = BUILT_UP.clip(exportRegion).unmask(0);

  // Flood detections
  var isFlood     = ndwi.gt(CFG.ndwi_flood_thresh);
  var floodOpen   = isFlood.and(builtUp.not()).and(urbanZone).multiply(1);
  var floodUrban  = isFlood.and(builtUp).multiply(2);
  var floodLabel  = floodOpen.add(floodUrban);  // 0, 1, or 2

  // Non-flood: confident dry, within urban zone
  var isDry       = ndwi.lt(CFG.ndwi_dry_thresh);
  var nonFlood    = isDry.and(urbanZone).and(isFlood.not()).multiply(3);

  // Combine: flood takes priority over non-flood
  var rawLabel = floodLabel
    .where(floodLabel.eq(0).and(nonFlood.gt(0)), nonFlood)
    .uint8();

  // Apply masks (cloud + shadow → 0; outside urban zone → 0)
  var finalLabel = rawLabel
    .where(invalidMask, 0)
    .where(urbanZone.not(), 0)
    .uint8()
    .rename('pseudo_label');

  return finalLabel;
}

// ============================================================
// STEP 5: PREPARE SAR
// Mask SAR to urban zone — no point exporting rural SAR pixels
// ============================================================

function prepareSAR(s1Image, exportRegion, urbanZone) {
  return s1Image
    .select(['VV', 'VH'])
    .clip(exportRegion)
    // Set rural pixels to NaN — they'll be filtered in Python chipping
    .updateMask(urbanZone)
    .toFloat();
}

// ============================================================
// STEP 6: DIAGNOSTICS
// ============================================================

function printDiagnostics(pseudoLabel, exportRegion, urbanKm2) {
  print('Urban area in export region (km²):', urbanKm2);

  var dist = pseudoLabel.reduceRegion({
    reducer:   ee.Reducer.frequencyHistogram(),
    geometry:  exportRegion,
    scale:     CFG.resolution_m,
    maxPixels: 1e12
  });
  print('Label distribution:', dist);
  print('  0 = masked/rural/ambiguous');
  print('  1 = flood open   (urban-adjacent)');
  print('  2 = flood urban  (built-up)');
  print('  3 = non-flood    (confident dry, urban zone)');
}

// ============================================================
// STEP 7: EXPORT
// ============================================================

function exportPair(sarImage, labelImage, exportRegion, eventName) {
  var bounds = exportRegion.bounds();

  Export.image.toDrive({
    image:          sarImage,
    description:    eventName + '_SAR',
    folder:         CFG.drive_folder,
    fileNamePrefix: eventName + '_SAR',
    region:         bounds,
    scale:          CFG.resolution_m,
    crs:            CFG.crs,
    maxPixels:      1e13,
    fileFormat:     'GeoTIFF',
    formatOptions:  { cloudOptimized: true }
  });

  Export.image.toDrive({
    image:          labelImage,
    description:    eventName + '_LABEL',
    folder:         CFG.drive_folder,
    fileNamePrefix: eventName + '_LABEL',
    region:         bounds,
    scale:          CFG.resolution_m,
    crs:            CFG.crs,
    maxPixels:      1e13,
    fileFormat:     'GeoTIFF',
    formatOptions:  { cloudOptimized: true }
  });

  print('Queued: ' + eventName + '_SAR  +  ' + eventName + '_LABEL');
}

// ============================================================
// MAIN
// ============================================================

var EVENT_INDEX = 0;
var event = EVENTS[EVENT_INDEX];

print('================================================');
print('Pseudo-label export (urban-constrained): ' + event.name);
print('Period: ' + event.start + ' → ' + event.end);
print('Urban buffer: ' + CFG.urban_buffer_m + 'm from built-up');
print('Notes: ' + event.notes);
print('================================================');

var aoi = ee.Geometry.Point([event.lon, event.lat])
            .buffer(event.buffer_km * 1000);

var pairedCol = findBestPair(aoi, event);
var nPairs = pairedCol.size().getInfo();
print('Valid S1+S2 pairs found:', nPairs);

if (nPairs === 0) {
  print('ERROR: No valid pairs. Increase cloud_threshold or temporal_window_days.');
} else {
  var bestS1 = ee.Image(pairedCol.first());
  var bestS2 = ee.Image(bestS1.get('best_s2'));

  print('S1 date:    ' + bestS1.date().format('YYYY-MM-dd').getInfo());
  print('S2 date:    ' + bestS2.date().format('YYYY-MM-dd').getInfo());
  print('Cloud %:    ' + ee.Number(bestS1.get('best_cloud')).round().getInfo());
  print('Gap (days): ' + ee.Number(bestS1.get('gap_days')).round().getInfo());

  // Export region: S1 ∩ S2 ∩ AOI
  var exportRegion = bestS1.geometry()
    .intersection(bestS2.geometry(), ee.ErrorMargin(10))
    .intersection(aoi, ee.ErrorMargin(10));

  var overlapKm2 = exportRegion.area(10).divide(1e6).round().getInfo();
  print('S1 ∩ S2 ∩ AOI (km²): ' + overlapKm2);

  // Urban zone within export region
  var urbanZone = getUrbanZone(exportRegion);
  var urbanKm2  = urbanZone.reduceRegion({
    reducer:   ee.Reducer.sum(),
    geometry:  exportRegion,
    scale:     100,
    maxPixels: 1e11
  }).get('Map');
  var urbanKm2Val = ee.Number(urbanKm2).multiply(0.01).round().getInfo();

  // Build outputs
  var invalidMask  = computeInvalidMask(bestS2, exportRegion);
  var pseudoLabel  = computePseudoLabel(bestS2, exportRegion,
                                        urbanZone, invalidMask);
  var sarBands     = prepareSAR(bestS1, exportRegion, urbanZone);

  printDiagnostics(pseudoLabel, exportRegion, urbanKm2Val);
  exportPair(sarBands, pseudoLabel, exportRegion, event.name);

  // ── Visualization ────────────────────────────────────────────────────

  Map.centerObject(exportRegion, 11);

  Map.addLayer(exportRegion,
    { color: '0000FF', fillColor: '00000000' }, 'Export region (S1∩S2∩AOI)');

  Map.addLayer(urbanZone.clip(exportRegion).selfMask(),
    { palette: ['yellow'] }, 'Urban zone (500m buffer)', false);

  Map.addLayer(bestS1.select('VV').clip(exportRegion),
    { min: -25, max: 0, palette: ['black', 'white'] }, 'SAR VV (dB)');

  Map.addLayer(bestS2.select(['B4', 'B3', 'B2']).clip(exportRegion),
    { min: 0, max: 3000 }, 'S2 RGB', false);

  Map.addLayer(pseudoLabel.clip(exportRegion),
    { min: 0, max: 3,
      palette: ['black', 'cyan', 'orange', 'green'] },
    'Labels (0=masked, 1=flood_open, 2=flood_urban, 3=non_flood)');

  Map.addLayer(invalidMask.clip(exportRegion).selfMask(),
    { palette: ['red'] }, 'Invalid (cloud+shadow)', false);
}

/*
 * QC CHECKLIST
 * 1. Toggle 'Urban zone' layer — yellow should cover city areas,
 *    not farmland. Adjust urban_buffer_m if needed.
 * 2. Toggle 'S2 RGB' — locate visually flooded streets/districts.
 * 3. Toggle 'Labels' — cyan/orange should sit over flooded urban areas.
 *    Green (non-flood) should cover dry streets and rooftops.
 *    Black (masked) over clouds, shadows, and rural areas is correct.
 * 4. Check console: label 2 (flood urban) count should be > 0.
 * 5. If QC passes → Tasks tab → Run both export tasks.
 * 6. Change EVENT_INDEX (0→1→2→3) and repeat.
 */
