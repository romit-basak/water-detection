/*
 * 03_event_screening.js
 *
 * Screens all 25 candidate flood events for pseudo-label suitability.
 *
 * For each event, computes:
 *   - Number of S1 acquisitions during flood period
 *   - Number of usable S2 acquisitions (< cloud threshold)
 *   - Best S1+S2 pair: dates, cloud %, temporal gap
 *   - Overlap area (km²) between best S1 and S2 footprints
 *   - Urban area (km²) within that overlap (WorldCover built-up)
 *   - Urban fraction within overlap
 *   - Quality grade: A/B/C/D
 *
 * Grade criteria:
 *   A: urban_overlap_km2 ≥ 100 AND cloud ≤ 10% AND gap ≤ 2 days
 *   B: urban_overlap_km2 ≥ 50  AND cloud ≤ 20% AND gap ≤ 3 days
 *   C: urban_overlap_km2 ≥ 20  AND cloud ≤ 30% AND gap ≤ 5 days
 *   D: everything else — not suitable for pseudo-labeling
 *
 * Output:
 *   - Console table (copy-paste to spreadsheet)
 *   - CSV export to Drive: water_detection_screening/event_screening.csv
 *
 * Runtime: ~3-5 min. Run as-is — no configuration needed.
 * All .getInfo() calls are batched per event to stay within GEE limits.
 */

// ============================================================
// CONFIGURATION
// ============================================================

var CFG = {
  cloud_threshold:     40,   // max S2 cloud % to consider (generous for screening)
  temporal_window_days: 5,   // max S1-S2 gap to consider a pair
  urban_scale_m:       100,  // scale for urban fraction computation
  overlap_scale_m:     100,  // scale for overlap area computation
  drive_folder:        'water_detection_screening',

  // Grade thresholds
  grade_A: { urban_km2: 100, cloud: 10,  gap_days: 2 },
  grade_B: { urban_km2: 50,  cloud: 20,  gap_days: 3 },
  grade_C: { urban_km2: 20,  cloud: 30,  gap_days: 5 },
};

// WorldCover 2021 built-up class
var WORLDCOVER = ee.Image('ESA/WorldCover/v200/2021');
var BUILT_UP   = WORLDCOVER.eq(50);

// ============================================================
// EVENT CATALOGUE
// All 25 events from the research document
// ============================================================

var EVENTS = [
  // Americas
  {
    id: 1, name: 'Hurricane_Florence_2018',
    lat: 34.2257, lon: -77.9447, buffer_km: 60,
    start: '2018-09-14', end: '2018-09-25',
    type: 'Coastal+Riverine', country: 'USA',
    tier: 2, notes: 'Carolinas; NASA ARIA FPM+DPM'
  },
  {
    id: 2, name: 'Hurricane_Ian_2022',
    lat: 26.6406, lon: -81.8723, buffer_km: 50,
    start: '2022-09-28', end: '2022-10-05',
    type: 'Coastal', country: 'USA',
    tier: 2, notes: 'Fort Myers; S2 usable Sep 30 (2 days post)'
  },
  {
    id: 3, name: 'Hurricane_Ida_2021',
    lat: 29.6499, lon: -90.7040, buffer_km: 60,
    start: '2021-08-29', end: '2021-09-08',
    type: 'Coastal+Riverine', country: 'USA',
    tier: 2, notes: 'New Orleans; NASA ARIA DPM'
  },
  {
    id: 4, name: 'Rio_Grande_do_Sul_2024',
    lat: -30.0346, lon: -51.2177, buffer_km: 60,
    start: '2024-05-01', end: '2024-05-20',
    type: 'Riverine', country: 'Brazil',
    tier: 2, notes: 'Porto Alegre; EMSR720; 76km² urban flooded'
  },
  {
    id: 5, name: 'Hurricane_Dorian_2019',
    lat: 26.5420, lon: -77.0639, buffer_km: 40,
    start: '2019-09-01', end: '2019-09-10',
    type: 'Coastal', country: 'Bahamas',
    tier: 1, notes: 'Freeport; ARIA FPM; EMSR385'
  },
  {
    id: 6, name: 'Hurricanes_Eta_Iota_2020',
    lat: 15.5000, lon: -88.0000, buffer_km: 80,
    start: '2020-11-03', end: '2020-11-25',
    type: 'Coastal+Riverine', country: 'Honduras',
    tier: 2, notes: 'San Pedro Sula; 6 CEMS activations'
  },
  {
    id: 7, name: 'Midwest_Floods_2019',
    lat: 41.2565, lon: -95.9345, buffer_km: 60,
    start: '2019-03-13', end: '2019-04-15',
    type: 'Riverine', country: 'USA',
    tier: 1, notes: 'Nebraska; IN Sen1Floods11 benchmark'
  },
  {
    id: 8, name: 'Hurricane_Maria_2017',
    lat: 18.2208, lon: -66.5901, buffer_km: 80,
    start: '2017-09-20', end: '2017-10-05',
    type: 'Coastal+Riverine', country: 'Puerto Rico',
    tier: 2, notes: 'Island-wide; ARIA DPM; heavy cloud cover'
  },

  // Europe
  {
    id: 9, name: 'Western_Europe_Floods_2021',
    lat: 50.5437, lon: 6.8183, buffer_km: 60,
    start: '2021-07-14', end: '2021-07-22',
    type: 'Riverine', country: 'Germany/Belgium',
    tier: 2, notes: 'Ahr Valley; EMSR517-520; F1=0.9 RAPID'
  },
  {
    id: 10, name: 'Emilia_Romagna_2023',
    lat: 44.2222, lon: 11.8765, buffer_km: 60,
    start: '2023-05-16', end: '2023-05-25',
    type: 'Riverine', country: 'Italy',
    tier: 2, notes: 'Faenza/Forli; EMSR659+664; building-level GT'
  },
  {
    id: 11, name: 'UK_Yorkshire_Floods_2015',
    lat: 53.9590, lon: -1.0815, buffer_km: 50,
    start: '2015-12-26', end: '2016-01-05',
    type: 'Riverine', country: 'UK',
    tier: 1, notes: 'York/Leeds; SIMULTANEOUS S1+S2 Dec 29 2015'
  },

  // Africa
  {
    id: 12, name: 'Cyclone_Idai_2019',
    lat: -19.8436, lon: 34.8389, buffer_km: 50,
    start: '2019-03-14', end: '2019-03-28',
    type: 'Coastal+Riverine', country: 'Mozambique',
    tier: 1, notes: 'Beira; IN UrbanSARFloods; EMSR348; GPS validation'
  },
  {
    id: 13, name: 'KwaZulu_Natal_2022',
    lat: -29.8587, lon: 30.9810, buffer_km: 50,
    start: '2022-04-09', end: '2022-04-20',
    type: 'Riverine', country: 'South Africa',
    tier: 3, notes: 'Durban; UNOSAT; FAO multi-sensor'
  },
  {
    id: 14, name: 'Nigeria_Floods_2022',
    lat: 7.7965, lon: 6.7395, buffer_km: 60,
    start: '2022-09-01', end: '2022-11-01',
    type: 'Riverine', country: 'Nigeria',
    tier: 2, notes: 'Lokoja; IN UrbanSARFloods'
  },

  // South Asia
  {
    id: 15, name: 'Chennai_Floods_2015',
    lat: 13.0827, lon: 80.2707, buffer_km: 50,
    start: '2015-11-25', end: '2015-12-10',
    type: 'Riverine+Coastal', country: 'India',
    tier: 3, notes: 'Chennai metro; ISRO/NRSC maps'
  },
  {
    id: 16, name: 'Kerala_Floods_2018',
    lat: 9.9816, lon: 76.2999, buffer_km: 60,
    start: '2018-08-09', end: '2018-08-25',
    type: 'Riverine', country: 'India',
    tier: 3, notes: 'Kochi; 10+ SAR papers; heavy cloud'
  },
  {
    id: 17, name: 'Cyclone_Amphan_2020',
    lat: 22.5726, lon: 88.3639, buffer_km: 70,
    start: '2020-05-20', end: '2020-06-05',
    type: 'Coastal', country: 'India/Bangladesh',
    tier: 3, notes: 'Kolkata; S1 May 22 (2d post); S2 Jun 3'
  },
  {
    id: 18, name: 'Pakistan_Floods_2022',
    lat: 27.5260, lon: 68.7732, buffer_km: 100,
    start: '2022-08-10', end: '2022-09-23',
    type: 'Riverine', country: 'Pakistan',
    tier: 1, notes: 'Sindh; TU Wien open dataset CC BY 4.0'
  },

  // Southeast Asia
  {
    id: 19, name: 'Jakarta_Floods_2020',
    lat: -6.2088, lon: 106.8456, buffer_km: 40,
    start: '2020-01-01', end: '2020-01-10',
    type: 'Riverine', country: 'Indonesia',
    tier: 3, notes: 'Jakarta; NASA ARIA FPM; 17% city flooded'
  },
  {
    id: 20, name: 'Typhoon_Vamco_2020',
    lat: 14.6760, lon: 121.0437, buffer_km: 40,
    start: '2020-11-11', end: '2020-11-18',
    type: 'Coastal+Riverine', country: 'Philippines',
    tier: 3, notes: 'Metro Manila; Marikina River; 75km² inundated'
  },

  // East Asia
  {
    id: 21, name: 'Western_Japan_2018',
    lat: 34.6168, lon: 133.7153, buffer_km: 30,
    start: '2018-07-05', end: '2018-07-15',
    type: 'Riverine', country: 'Japan',
    tier: 1, notes: 'Mabi/Kurashiki; IN UrbanSARFloods; GSI polygons'
  },
  {
    id: 22, name: 'Typhoon_Hagibis_2019',
    lat: 35.6762, lon: 139.6503, buffer_km: 80,
    start: '2019-10-12', end: '2019-10-20',
    type: 'Coastal+Riverine', country: 'Japan',
    tier: 1, notes: 'Tokyo/Fukushima; ARIA open Figshare dataset'
  },
  {
    id: 23, name: 'Zhengzhou_Floods_2021',
    lat: 34.7472, lon: 113.6249, buffer_km: 40,
    start: '2021-07-17', end: '2021-07-31',
    type: 'Riverine+Pluvial', country: 'China',
    tier: 3, notes: 'Zhengzhou megacity; cloud-free S2 Jul 31'
  },

  // Oceania
  {
    id: 24, name: 'Lismore_2022',
    lat: -28.8133, lon: 153.2750, buffer_km: 30,
    start: '2022-02-28', end: '2022-03-15',
    type: 'Riverine', country: 'Australia',
    tier: 2, notes: 'Lismore CBD; EMSR570+637; SAR only viable sensor'
  },
  {
    id: 25, name: 'Cyclone_Gabrielle_2023',
    lat: -39.4928, lon: 176.9120, buffer_km: 50,
    start: '2023-02-12', end: '2023-02-22',
    type: 'Coastal+Riverine', country: 'New Zealand',
    tier: 2, notes: 'Hawkes Bay/Napier; Dragonfly SAR polygons'
  },
];

// ============================================================
// CORE SCREENING FUNCTION
// ============================================================

function screenEvent(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);

  var windowMs = CFG.temporal_window_days * 24 * 3600 * 1000;

  // --- S1 collection ---
  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(event.start, event.end)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'));

  // --- S2 collection ---
  var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(event.start, event.end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CFG.cloud_threshold));

  // --- Spatiotemporal join: S1 must spatially overlap S2 ---
  var spatialFilter  = ee.Filter.intersects({
    leftField: '.geo', rightField: '.geo'
  });
  var temporalFilter = ee.Filter.maxDifference({
    difference: windowMs,
    leftField:  'system:time_start',
    rightField: 'system:time_start'
  });

  var joined = ee.Join.saveAll({ matchesKey: 's2_matches' })
    .apply(s1Col, s2Col, ee.Filter.and(spatialFilter, temporalFilter));

  // For each S1, attach best (lowest cloud) spatially overlapping S2
  var withBest = joined.map(function(s1img) {
    var matches = ee.ImageCollection.fromImages(s1img.get('s2_matches'));
    var bestS2  = matches.sort('CLOUDY_PIXEL_PERCENTAGE').first();
    var gapDays = ee.Number(s1img.get('system:time_start'))
                    .subtract(ee.Number(bestS2.get('system:time_start')))
                    .abs().divide(86400000);  // ms → days
    return s1img
      .set('best_s2',       bestS2)
      .set('best_cloud',    bestS2.get('CLOUDY_PIXEL_PERCENTAGE'))
      .set('gap_days',      gapDays)
      .set('n_s2_matches',  matches.size());
  });

  // Best overall pair = lowest cloud among all S1 candidates
  var best = ee.Image(withBest
    .filter(ee.Filter.gt('n_s2_matches', 0))
    .sort('best_cloud')
    .first());

  // --- Compute overlap and urban stats for best pair ---
  var bestS2       = ee.Image(best.get('best_s2'));
  var s1Footprint  = best.geometry();
  var s2Footprint  = bestS2.geometry();
  var overlap      = s1Footprint
                       .intersection(s2Footprint, ee.ErrorMargin(10))
                       .intersection(aoi,          ee.ErrorMargin(10));

  // Urban pixels within overlap
  var urbanInOverlap = BUILT_UP.clip(overlap);
  var urbanStats = urbanInOverlap.reduceRegion({
    reducer:   ee.Reducer.sum(),
    geometry:  overlap,
    scale:     CFG.urban_scale_m,
    maxPixels: 1e11
  });

  // Overlap area (m²)
  var overlapAreaM2 = overlap.area(10);
  var overlapKm2    = overlapAreaM2.divide(1e6);

  // Urban area: each built-up pixel at 100m scale = 10,000 m² = 0.01 km²
  var urbanKm2 = ee.Number(urbanStats.get('Map')).multiply(0.01);
  var urbanFrac = urbanKm2.divide(overlapKm2.max(0.001));

  // Pack result
  return {
    best:        best,
    bestS2:      bestS2,
    overlapKm2:  overlapKm2,
    urbanKm2:    urbanKm2,
    urbanFrac:   urbanFrac,
    s1Count:     s1Col.size(),
    s2Count:     s2Col.size(),
    nPairs:      withBest.filter(ee.Filter.gt('n_s2_matches', 0)).size()
  };
}

function assignGrade(cloudPct, gapDays, urbanKm2) {
  var A = CFG.grade_A, B = CFG.grade_B, C = CFG.grade_C;
  if (urbanKm2 >= A.urban_km2 && cloudPct <= A.cloud  && gapDays <= A.gap_days) return 'A';
  if (urbanKm2 >= B.urban_km2 && cloudPct <= B.cloud  && gapDays <= B.gap_days) return 'B';
  if (urbanKm2 >= C.urban_km2 && cloudPct <= C.cloud  && gapDays <= C.gap_days) return 'C';
  return 'D';
}

// ============================================================
// RUN SCREENING — one event at a time to avoid GEE timeouts
// Change EVENT_INDEX to screen a different event.
// After screening all, collect results and export CSV.
// ============================================================

// To screen ALL events at once, set RUN_ALL = true.
// Warning: with 25 events this may hit GEE memory limits.
// Recommended: run in batches of 5 by setting BATCH_START/END.
var RUN_ALL    = true;
var BATCH_START = 0;   // inclusive
var BATCH_END   = 25;  // exclusive

var results = [];

var eventsToRun = EVENTS.slice(BATCH_START, BATCH_END);

print('================================================');
print('Urban Flood Event Screening — ' + eventsToRun.length + ' events');
print('Cloud threshold: ' + CFG.cloud_threshold + '%');
print('Temporal window: ' + CFG.temporal_window_days + ' days');
print('================================================');
print('');
print('ID | Event                         | S1 | S2 | Pairs | S1 Date    | S2 Date    | Cloud% | Gap(d) | Overlap km² | Urban km² | Urban% | Grade | Tier | Notes');
print('---|-------------------------------|----|----|-------|------------|------------|--------|--------|-------------|-----------|--------|-------|------|------');

eventsToRun.forEach(function(event) {
  var r = screenEvent(event);

  // Extract scalar values — these are the GEE server-side calls
  var s1Count    = r.s1Count.getInfo();
  var s2Count    = r.s2Count.getInfo();
  var nPairs     = r.nPairs.getInfo();

  if (nPairs === 0) {
    print(
      pad(event.id, 3) + ' | ' +
      pad(event.name, 29) + ' | ' +
      pad(s1Count, 2) + ' | ' +
      pad(s2Count, 2) + ' | ' +
      pad(0, 5) + ' | ' +
      pad('N/A', 10) + ' | ' +
      pad('N/A', 10) + ' | ' +
      pad('N/A', 6) + ' | ' +
      pad('N/A', 6) + ' | ' +
      pad('N/A', 11) + ' | ' +
      pad('N/A', 9) + ' | ' +
      pad('N/A', 6) + ' | ' +
      pad('D', 5) + ' | ' +
      pad(event.tier, 4) + ' | ' +
      'No valid SAR+optical pairs found'
    );
    results.push(ee.Feature(null, {
      id: event.id, name: event.name, country: event.country,
      type: event.type, tier: event.tier,
      s1_count: s1Count, s2_count: s2Count, n_pairs: 0,
      s1_date: 'N/A', s2_date: 'N/A',
      cloud_pct: -1, gap_days: -1,
      overlap_km2: 0, urban_km2: 0, urban_frac: 0,
      grade: 'D', notes: event.notes
    }));
    return;
  }

  var cloudPct   = ee.Number(r.best.get('best_cloud')).round().getInfo();
  var gapDays    = ee.Number(r.best.get('gap_days')).round().getInfo();
  var overlapKm2 = r.overlapKm2.round().getInfo();
  var urbanKm2   = r.urbanKm2.round().getInfo();
  var urbanFrac  = r.urbanFrac.multiply(100).round().getInfo(); // as %
  var s1Date     = r.best.date().format('YYYY-MM-dd').getInfo();
  var s2Date     = r.bestS2.date().format('YYYY-MM-dd').getInfo();
  var grade      = assignGrade(cloudPct, gapDays, urbanKm2);

  print(
    pad(event.id, 3) + ' | ' +
    pad(event.name, 29) + ' | ' +
    pad(s1Count, 2) + ' | ' +
    pad(s2Count, 2) + ' | ' +
    pad(nPairs, 5) + ' | ' +
    pad(s1Date, 10) + ' | ' +
    pad(s2Date, 10) + ' | ' +
    pad(cloudPct, 6) + ' | ' +
    pad(gapDays, 6) + ' | ' +
    pad(overlapKm2, 11) + ' | ' +
    pad(urbanKm2, 9) + ' | ' +
    pad(urbanFrac + '%', 6) + ' | ' +
    pad(grade, 5) + ' | ' +
    pad(event.tier, 4) + ' | ' +
    event.notes
  );

  results.push(ee.Feature(null, {
    id: event.id, name: event.name, country: event.country,
    type: event.type, tier: event.tier,
    s1_count: s1Count, s2_count: s2Count, n_pairs: nPairs,
    s1_date: s1Date, s2_date: s2Date,
    cloud_pct: cloudPct, gap_days: gapDays,
    overlap_km2: overlapKm2, urban_km2: urbanKm2,
    urban_frac_pct: urbanFrac,
    grade: grade, notes: event.notes
  }));
});

// ============================================================
// EXPORT CSV TO DRIVE
// ============================================================

Export.table.toDrive({
  collection: ee.FeatureCollection(results),
  description: 'urban_flood_event_screening',
  folder: CFG.drive_folder,
  fileNamePrefix: 'event_screening_results',
  fileFormat: 'CSV'
});

print('');
print('================================================');
print('CSV export queued → Tasks tab → Run');
print('Drive folder: ' + CFG.drive_folder);
print('================================================');

// ============================================================
// SUMMARY: GRADE COUNTS
// ============================================================

print('');
print('--- GRADE SUMMARY ---');
var gradeCounts = {A: 0, B: 0, C: 0, D: 0};
results.forEach(function(f) {
  var g = ee.Feature(f).get('grade').getInfo();
  gradeCounts[g] = (gradeCounts[g] || 0) + 1;
});
print('Grade A (best — ready for pseudo-labeling):', gradeCounts.A);
print('Grade B (good — usable with caveats):',       gradeCounts.B);
print('Grade C (marginal — proceed carefully):',     gradeCounts.C);
print('Grade D (unsuitable — skip):',                gradeCounts.D);

// ============================================================
// HELPER: string padding for table alignment
// ============================================================

function pad(val, width) {
  var s = String(val);
  while (s.length < width) s = s + ' ';
  return s.substring(0, width);
}
