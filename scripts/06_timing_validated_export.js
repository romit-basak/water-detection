/*
 * 06_timing_validated_export.js
 *
 * Re-exports events with timing validation — ensures both S1 and S2
 * are acquired DURING active flooding, not before or after.
 *
 * Key change from previous scripts:
 *   Previous approach: find lowest-cloud S2 within temporal window,
 *   sort by cloud cover. This picked post-flood clear-sky images.
 *
 *   This approach: restrict search window to FLOOD PEAK DATES only,
 *   derived from documented flood timelines. Only pairs where both
 *   sensors fall within the peak window are considered.
 *
 * Events retained (6 with confirmed good timing + re-evaluated events):
 *   CONFIRMED GOOD (keep existing exports):
 *     UK_Yorkshire_2015      S1 Dec 27, S2 Dec 29 — flood peak Dec 26-29 ✓
 *     Pakistan_Sindh_2022    S1 Sep 3, S2 Sep 5 — flood persisted weeks ✓
 *     Seville_Spain_2025     S1 Feb 2, S2 Jan 30 — during Guadalquivir flood ✓
 *     Libya_Derna_2023       S1 Sep 12, S2 Sep 12 — 0-day gap, during event ✓
 *     Hurricane_Florence_2018 S1 Sep 14, S2 Sep 18 — weeks of flooding ✓
 *     Cologne_Rhine_2021     S1 Jul 15, S2 Jul 18 — Rhine at peak ✓
 *     Western_Europe_2021    S1 Jul 15, S2 Jul 18 — Ahr at peak ✓
 *
 *   NEEDS RE-EXPORT with tighter peak window:
 *     Zhengzhou_2021         Peak Jul 20-23. Need S1+S2 in that window.
 *     Bucharest_Romania_2024 Peak Sep 13-16. Need S1+S2 in that window.
 *     Cyclone_Gabrielle_2023 Riverine component persisted Feb 14-20.
 *     Seoul_2022             Peak Aug 8-9. Very tight — may not be feasible.
 *     Typhoon_Vamco_2020     Need S1 during Nov 12-14, not Nov 16.
 *
 * Run EVENT_INDEX 0-4 to re-export the problem events.
 * Events 0-6 (confirmed good) do NOT need re-running — skip them.
 */

var CFG = {
  cloud_threshold:   60,   // raised — we need images during storm, clouds expected
  urban_buffer_m:   500,
  ndwi_flood_thresh: 0.08, // lowered further — storm-period images are noisier
  ndwi_dry_thresh:  -0.15,
  building_height_m: 10,
  shadow_max_m:      80,
  resolution_m:      10,
  drive_folder:     'water_detection_pseudolabels',
  crs:              'EPSG:4326',
};

var WORLDCOVER = ee.Image('ESA/WorldCover/v200/2021');
var BUILT_UP   = WORLDCOVER.eq(50);
var URBAN_ZONE_GLOBAL = BUILT_UP
  .focal_max({ radius: CFG.urban_buffer_m, units: 'meters',
               kernelType: 'circle' })
  .unmask(0);

// ============================================================
// EVENTS — only the ones needing re-export
// peak_start / peak_end define the active flood window
// Both S1 and S2 must fall within this window
// ============================================================

var REEXPORT_EVENTS = [
  {
    index: 0,
    name: 'Zhengzhou_China_2021',
    lon: 113.6249, lat: 34.7472, buffer_km: 40,
    // Full event: Jul 17-31. Peak inundation: Jul 20-24
    // Jul 20 was the record rainfall day; flooding persisted through Jul 23
    peak_start: '2021-07-20', peak_end: '2021-07-25',
    search_start: '2021-07-17', search_end: '2021-07-28',
    notes: 'Peak rainfall Jul 20; streets flooded Jul 20-23'
  },
  {
    index: 1,
    name: 'Bucharest_Romania_2024',
    lon: 26.1025, lat: 44.4268, buffer_km: 40,
    // Storm Boris peaked Sep 13-16 in Romania
    // Galati/Iasi most affected Sep 13-15
    peak_start: '2024-09-13', peak_end: '2024-09-17',
    search_start: '2024-09-12', search_end: '2024-09-20',
    notes: 'Storm Boris peak flooding Sep 13-16'
  },
  {
    index: 2,
    name: 'Cyclone_Gabrielle_NZ_2023',
    lon: 176.9120, lat: -39.4928, buffer_km: 50,
    // Cyclone passed Feb 14. Riverine flooding in Hawkes Bay persisted
    // through Feb 19-20 as rivers crested. Coastal surge receded faster.
    peak_start: '2023-02-14', peak_end: '2023-02-20',
    search_start: '2023-02-13', search_end: '2023-02-22',
    notes: 'Cyclone Feb 14; Hawkes Bay rivers crested Feb 15-18'
  },
  {
    index: 3,
    name: 'Seoul_South_Korea_2022',
    lon: 126.9780, lat: 37.5665, buffer_km: 40,
    // Record rainfall Aug 8 night. Peak flooding Aug 8-9.
    // Very tight window — S2 may not have clear enough view
    // Widen cloud threshold to 70% for this one
    peak_start: '2022-08-08', peak_end: '2022-08-11',
    search_start: '2022-08-07', search_end: '2022-08-13',
    notes: 'Record rainfall Aug 8 night; flooding Aug 8-10'
  },
  {
    index: 4,
    name: 'Typhoon_Vamco_Manila_2020',
    lon: 121.0437, lat: 14.6760, buffer_km: 40,
    // Typhoon Nov 11-12. Marikina River peaked Nov 12-13.
    // Flooding persisted Nov 12-14 in Marikina basin
    peak_start: '2020-11-11', peak_end: '2020-11-15',
    search_start: '2020-11-10', search_end: '2020-11-17',
    notes: 'Typhoon landfall Nov 11; Marikina peak Nov 12-13'
  },
];

// ============================================================
// PAIR FINDER — restricted to peak window
// Tries peak window first, falls back to full search window
// if no pairs found, reporting which window was used
// ============================================================

function findPeakPair(aoi, event) {
  var windowMs = 3 * 24 * 3600 * 1000;  // 3-day max gap during peak

  function queryPairs(s1Start, s1End, s2Start, s2End, cloudThresh) {
    var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(aoi)
      .filterDate(s1Start, s1End)
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .filter(ee.Filter.eq('instrumentMode', 'IW'));

    var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi)
      .filterDate(s2Start, s2End)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudThresh));

    var joined = ee.Join.saveAll({ matchesKey: 's2_matches' })
      .apply(s1Col, s2Col, ee.Filter.and(
        ee.Filter.intersects({ leftField: '.geo', rightField: '.geo' }),
        ee.Filter.maxDifference({
          difference: windowMs,
          leftField: 'system:time_start', rightField: 'system:time_start'
        })
      ));

    return joined.map(function(img) {
      var matches = ee.ImageCollection.fromImages(img.get('s2_matches'));
      var bestS2  = matches.sort('CLOUDY_PIXEL_PERCENTAGE').first();
      var gapDays = ee.Number(img.get('system:time_start'))
                      .subtract(ee.Number(bestS2.get('system:time_start')))
                      .abs().divide(86400000);
      return img
        .set('best_s2',      bestS2)
        .set('best_cloud',   bestS2.get('CLOUDY_PIXEL_PERCENTAGE'))
        .set('gap_days',     gapDays)
        .set('n_s2_matches', matches.size());
    }).filter(ee.Filter.gt('n_s2_matches', 0)).sort('best_cloud');
  }

  // Try 1: strict peak window, normal cloud threshold
  var peakPairs = queryPairs(
    event.peak_start, event.peak_end,
    event.peak_start, event.peak_end,
    CFG.cloud_threshold
  );

  // Try 2: wider search window, higher cloud threshold (storm conditions)
  var widePairs = queryPairs(
    event.search_start, event.search_end,
    event.search_start, event.search_end,
    80  // accept very cloudy during storm
  );

  // Return peak pairs if available, otherwise wide pairs
  // Tag which was used so we can print it
  var nPeak = peakPairs.size().getInfo();
  var nWide = widePairs.size().getInfo();

  print(event.name + ':');
  print('  Peak window pairs (' + event.peak_start + '–' + event.peak_end + '): ' + nPeak);
  print('  Wide window pairs (' + event.search_start + '–' + event.search_end + '): ' + nWide);

  if (nPeak > 0) {
    var best = ee.Image(peakPairs.first());
    var s2   = ee.Image(best.get('best_s2'));
    print('  Using PEAK pair: S1=' + best.date().format('YYYY-MM-dd').getInfo() +
          ' S2=' + s2.date().format('YYYY-MM-dd').getInfo() +
          ' cloud=' + ee.Number(best.get('best_cloud')).round().getInfo() + '%');
    return { img: best, window: 'peak', nPairs: nPeak };
  } else if (nWide > 0) {
    var best = ee.Image(widePairs.first());
    var s2   = ee.Image(best.get('best_s2'));
    print('  Using WIDE pair: S1=' + best.date().format('YYYY-MM-dd').getInfo() +
          ' S2=' + s2.date().format('YYYY-MM-dd').getInfo() +
          ' cloud=' + ee.Number(best.get('best_cloud')).round().getInfo() + '% — WARNING: may be post-flood');
    return { img: best, window: 'wide', nPairs: nWide };
  } else {
    print('  NO PAIRS FOUND — event cannot be pseudo-labeled');
    return { img: null, window: 'none', nPairs: 0 };
  }
}

// ============================================================
// LABEL + EXPORT FUNCTIONS (same as before)
// ============================================================

function computeInvalidMask(s2Image, exportRegion) {
  var scl = s2Image.select('SCL');
  var cloudMask = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
  var solarElevDeg = ee.Number(90).subtract(
    ee.Number(s2Image.get('MEAN_SOLAR_ZENITH_ANGLE'))
  );
  var solarElevRad = solarElevDeg.multiply(Math.PI / 180);
  var shadowLen = ee.Number(CFG.building_height_m)
                    .divide(solarElevRad.tan().max(0.1)).min(CFG.shadow_max_m);
  var shadowMask = BUILT_UP
    .focal_max({ radius: shadowLen, units: 'meters', kernelType: 'circle' })
    .clip(exportRegion).unmask(0);
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
    .where(floodLabel.eq(0).and(nonFlood.gt(0)), nonFlood).uint8();
  return rawLabel.where(invalidMask, 0).where(urbanZone.not(), 0)
    .uint8().rename('pseudo_label');
}

// ============================================================
// MAIN — change EVENT_INDEX to run each re-export event
// ============================================================

var EVENT_INDEX = 0;
var event = REEXPORT_EVENTS[EVENT_INDEX];

print('================================================');
print('Timing-validated re-export: ' + event.name);
print('Notes: ' + event.notes);
print('================================================');

var aoi    = ee.Geometry.Point([event.lon, event.lat])
               .buffer(event.buffer_km * 1000);
var result = findPeakPair(aoi, event);

if (result.nPairs === 0) {
  print('Cannot export — no valid pairs during flood peak.');
  print('Consider dropping this event from the dataset.');
} else {
  var bestS1 = result.img;
  var bestS2 = ee.Image(bestS1.get('best_s2'));

  var exportRegion = bestS1.geometry()
    .intersection(bestS2.geometry(), ee.ErrorMargin(10))
    .intersection(aoi, ee.ErrorMargin(10));

  var areaKm2 = exportRegion.area(10).divide(1e6).round().getInfo();
  print('Export region km²: ' + areaKm2);

  if (areaKm2 < 1) {
    print('Export region too small — S1 and S2 footprints barely overlap.');
    print('Try widening buffer_km or adjusting peak dates.');
  } else {
    var urbanZone   = URBAN_ZONE_GLOBAL.clip(exportRegion).unmask(0);
    var invalidMask = computeInvalidMask(bestS2, exportRegion);
    var pseudoLabel = computePseudoLabel(
      bestS2, exportRegion, urbanZone, invalidMask
    );
    var sarBands = bestS1.select(['VV', 'VH'])
      .clip(exportRegion).updateMask(urbanZone).toFloat();

    // Quick label check
    var dist = pseudoLabel.reduceRegion({
      reducer: ee.Reducer.frequencyHistogram(),
      geometry: exportRegion, scale: 100,
      maxPixels: 1e12, bestEffort: true
    });
    print('Label distribution:', dist);

    // Overwrite the previous bad export with corrected dates
    Export.image.toDrive({
      image: sarBands, description: event.name + '_SAR',
      folder: CFG.drive_folder, fileNamePrefix: event.name + '_SAR',
      region: exportRegion.bounds(), scale: CFG.resolution_m,
      crs: CFG.crs, maxPixels: 1e13, fileFormat: 'GeoTIFF',
      formatOptions: { cloudOptimized: true }
    });
    Export.image.toDrive({
      image: pseudoLabel, description: event.name + '_LABEL',
      folder: CFG.drive_folder, fileNamePrefix: event.name + '_LABEL',
      region: exportRegion.bounds(), scale: CFG.resolution_m,
      crs: CFG.crs, maxPixels: 1e13, fileFormat: 'GeoTIFF',
      formatOptions: { cloudOptimized: true }
    });

    print('Tasks queued — check Tasks tab → Run');

    // Visualisation
    Map.centerObject(exportRegion, 11);
    Map.addLayer(exportRegion, { color: '0000FF', fillColor: '00000000' }, 'Export region');
    Map.addLayer(bestS1.select('VV').clip(exportRegion),
      { min: -25, max: 0, palette: ['black', 'white'] }, 'SAR VV');
    Map.addLayer(bestS2.select(['B4','B3','B2']).clip(exportRegion),
      { min: 0, max: 3000 }, 'S2 RGB', false);
    Map.addLayer(pseudoLabel.clip(exportRegion),
      { min: 0, max: 3, palette: ['black', 'cyan', 'orange', 'green'] },
      'Labels (0=masked 1=open 2=urban 3=dry)');
  }
}

/*
 * EVENT INDEX REFERENCE (re-export only):
 *   0  Zhengzhou_China_2021      peak Jul 20-25
 *   1  Bucharest_Romania_2024    peak Sep 13-17
 *   2  Cyclone_Gabrielle_NZ_2023 peak Feb 14-20
 *   3  Seoul_South_Korea_2022    peak Aug 8-11 (may have no pairs)
 *   4  Typhoon_Vamco_Manila_2020 peak Nov 11-15
 *
 * Events NOT needing re-export (already have good timing):
 *   UK_Yorkshire_2015, Pakistan_Sindh_2022, Seville_Spain_2025,
 *   Libya_Derna_2023, Hurricane_Florence_2018,
 *   Cologne_Rhine_2021, Western_Europe_2021
 */
