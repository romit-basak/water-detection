/*
 * Urban Flood Event Data Discovery Script
 * 
 * Purpose: Survey satellite data availability for urban flood events
 * Sensors: Sentinel-1, Sentinel-2, Landsat 8/9, ALOS-2 PALSAR-2
 * 
 * For each flood event, we identify:
 * - Available SAR imagery (with orbit, incidence angle)
 * - Available optical imagery (with cloud %)
 * - Coincident pairs (optical + SAR within temporal window)
 * - Urban pixel coverage
 * 
 * Focus: Mid-latitude cities (better polar-orbiting satellite coverage)
 */

// ============================================================
// CONFIGURATION
// ============================================================

var TEMPORAL_WINDOW_DAYS = 2;  // Max days between optical and SAR for "coincident"
var CLOUD_THRESHOLD = 30;      // Max cloud % for usable optical imagery
var URBAN_MIN_FRACTION = 0.1;  // Minimum urban fraction in AOI

// Urban land cover from ESA WorldCover 2021
var worldCover = ee.Image("ESA/WorldCover/v200/2021");
var urbanMask = worldCover.eq(50);  // Class 50 = Built-up

// ============================================================
// FLOOD EVENTS DATABASE
// ============================================================
// Focus on mid-latitude cities (30-60 deg N/S) for better satellite coverage
// Each event: name, center coords, buffer (km), flood start, flood end, event type

var floodEvents = [
  // === NORTH AMERICA (Mid-latitude) ===
  {
    name: 'Houston_Harvey_2017',
    lat: 29.7604,
    lon: -95.3698,
    buffer_km: 50,
    flood_start: '2017-08-25',
    flood_end: '2017-09-10',
    type: 'Hurricane',
    country: 'USA',
    notes: 'Baseline event from paper'
  },
  {
    name: 'Miami_Irma_2017',
    lat: 25.7617,
    lon: -80.1918,
    buffer_km: 30,
    flood_start: '2017-09-09',
    flood_end: '2017-09-20',
    type: 'Hurricane',
    country: 'USA',
    notes: 'Quick hurricane dispersal expected'
  },
  {
    name: 'Midwest_USA_Omaha_2019',
    lat: 41.2565,
    lon: -95.9345,
    buffer_km: 40,
    flood_start: '2019-03-13',
    flood_end: '2019-04-15',
    type: 'Fluvial',
    country: 'USA',
    notes: 'Extended Missouri River flooding, good for clear sky overlap'
  },
  {
    name: 'New_Orleans_Ida_2021',
    lat: 29.9511,
    lon: -90.0715,
    buffer_km: 35,
    flood_start: '2021-08-29',
    flood_end: '2021-09-10',
    type: 'Hurricane',
    country: 'USA',
    notes: 'Category 4, extensive urban flooding'
  },
  {
    name: 'Fort_Lauderdale_2023',
    lat: 26.1224,
    lon: -80.1373,
    buffer_km: 25,
    flood_start: '2023-04-12',
    flood_end: '2023-04-20',
    type: 'Flash_flood',
    country: 'USA',
    notes: 'April 2023 floods, 25 inches in 24hrs'
  },
  
  // === EUROPE (Excellent mid-latitude coverage) ===
  {
    name: 'Ahr_Valley_Germany_2021',
    lat: 50.4337,
    lon: 7.1183,
    buffer_km: 25,
    flood_start: '2021-07-14',
    flood_end: '2021-07-25',
    type: 'Fluvial',
    country: 'Germany',
    notes: 'Catastrophic flooding, well-documented'
  },
  {
    name: 'Liege_Belgium_2021',
    lat: 50.6326,
    lon: 5.5797,
    buffer_km: 20,
    flood_start: '2021-07-14',
    flood_end: '2021-07-25',
    type: 'Fluvial',
    country: 'Belgium',
    notes: 'Same event as Ahr Valley, simultaneous'
  },
  {
    name: 'Cologne_Germany_2021',
    lat: 50.9375,
    lon: 6.9603,
    buffer_km: 20,
    flood_start: '2021-07-14',
    flood_end: '2021-07-20',
    type: 'Fluvial',
    country: 'Germany',
    notes: 'Rhine flooding, large urban area'
  },
  {
    name: 'Valencia_Spain_2024',
    lat: 39.4699,
    lon: -0.3763,
    buffer_km: 25,
    flood_start: '2024-10-29',
    flood_end: '2024-11-10',
    type: 'Flash_flood',
    country: 'Spain',
    notes: 'October 2024 DANA floods'
  },
  {
    name: 'Emilia_Romagna_Italy_2023',
    lat: 44.4949,
    lon: 11.3426,
    buffer_km: 40,
    flood_start: '2023-05-16',
    flood_end: '2023-05-25',
    type: 'Fluvial',
    country: 'Italy',
    notes: 'Bologna region, extended flooding'
  },
  {
    name: 'Paris_Seine_2016',
    lat: 48.8566,
    lon: 2.3522,
    buffer_km: 30,
    flood_start: '2016-05-30',
    flood_end: '2016-06-10',
    type: 'Fluvial',
    country: 'France',
    notes: 'Seine flooding, major urban center'
  },
  {
    name: 'Prague_2002',
    lat: 50.0755,
    lon: 14.4378,
    buffer_km: 25,
    flood_start: '2002-08-07',
    flood_end: '2002-08-20',
    type: 'Fluvial',
    country: 'Czechia',
    notes: 'Historic flood - predates S1/S2, Landsat 7 only'
  },
  {
    name: 'Dresden_2013',
    lat: 51.0504,
    lon: 13.7373,
    buffer_km: 25,
    flood_start: '2013-06-01',
    flood_end: '2013-06-15',
    type: 'Fluvial',
    country: 'Germany',
    notes: 'Elbe flooding - Landsat 8 available, no S1/S2'
  },
  
  // === EAST ASIA (Good mid-latitude coverage) ===
  {
    name: 'Tokyo_Hagibis_2019',
    lat: 35.6762,
    lon: 139.6503,
    buffer_km: 40,
    flood_start: '2019-10-12',
    flood_end: '2019-10-20',
    type: 'Typhoon',
    country: 'Japan',
    notes: 'Typhoon Hagibis, rapid clearing typical'
  },
  {
    name: 'Zhengzhou_China_2021',
    lat: 34.7472,
    lon: 113.6249,
    buffer_km: 30,
    flood_start: '2021-07-17',
    flood_end: '2021-07-28',
    type: 'Flash_flood',
    country: 'China',
    notes: '1000-year rainfall event'
  },
  {
    name: 'Seoul_2022',
    lat: 37.5665,
    lon: 126.9780,
    buffer_km: 30,
    flood_start: '2022-08-08',
    flood_end: '2022-08-15',
    type: 'Flash_flood',
    country: 'South Korea',
    notes: 'August 2022 Seoul floods'
  },
  {
    name: 'Osaka_2018',
    lat: 34.6937,
    lon: 135.5023,
    buffer_km: 30,
    flood_start: '2018-07-05',
    flood_end: '2018-07-15',
    type: 'Fluvial',
    country: 'Japan',
    notes: 'July 2018 Japan floods'
  },
  
  // === AUSTRALIA (Southern mid-latitude) ===
  {
    name: 'Sydney_2022',
    lat: -33.8688,
    lon: 151.2093,
    buffer_km: 40,
    flood_start: '2022-02-23',
    flood_end: '2022-03-15',
    type: 'Fluvial',
    country: 'Australia',
    notes: 'Extended Hawkesbury-Nepean flooding'
  },
  {
    name: 'Brisbane_2022',
    lat: -27.4698,
    lon: 153.0251,
    buffer_km: 35,
    flood_start: '2022-02-26',
    flood_end: '2022-03-10',
    type: 'Fluvial',
    country: 'Australia',
    notes: 'Same event system as Sydney'
  },
  {
    name: 'Lismore_Australia_2022',
    lat: -28.8133,
    lon: 153.2750,
    buffer_km: 20,
    flood_start: '2022-02-28',
    flood_end: '2022-03-15',
    type: 'Fluvial',
    country: 'Australia',
    notes: 'Record-breaking floods'
  },
  
  // === SOUTH AMERICA ===
  {
    name: 'Rio_Grande_Sul_Brazil_2024',
    lat: -30.0346,
    lon: -51.2177,
    buffer_km: 40,
    flood_start: '2024-04-27',
    flood_end: '2024-05-20',
    type: 'Fluvial',
    country: 'Brazil',
    notes: 'Porto Alegre area, extended flooding'
  },
  {
    name: 'Buenos_Aires_2013',
    lat: -34.6037,
    lon: -58.3816,
    buffer_km: 30,
    flood_start: '2013-04-02',
    flood_end: '2013-04-10',
    type: 'Flash_flood',
    country: 'Argentina',
    notes: 'Pre-Sentinel era, Landsat only'
  },
  
  // === ARID REGIONS (Clear sky advantage) ===
  {
    name: 'Dubai_2024',
    lat: 25.2048,
    lon: 55.2708,
    buffer_km: 30,
    flood_start: '2024-04-15',
    flood_end: '2024-04-20',
    type: 'Flash_flood',
    country: 'UAE',
    notes: 'Rare urban flood, excellent clear sky conditions'
  },
  {
    name: 'Derna_Libya_2023',
    lat: 32.7570,
    lon: 22.6389,
    buffer_km: 15,
    flood_start: '2023-09-10',
    flood_end: '2023-09-25',
    type: 'Dam_breach',
    country: 'Libya',
    notes: 'Catastrophic dam failure, clear arid conditions'
  },
  {
    name: 'Jeddah_2009',
    lat: 21.4858,
    lon: 39.1925,
    buffer_km: 25,
    flood_start: '2009-11-25',
    flood_end: '2009-12-05',
    type: 'Flash_flood',
    country: 'Saudi Arabia',
    notes: 'Pre-Sentinel, rare urban flood in arid region'
  },
  {
    name: 'Jeddah_2022',
    lat: 21.4858,
    lon: 39.1925,
    buffer_km: 25,
    flood_start: '2022-11-24',
    flood_end: '2022-12-01',
    type: 'Flash_flood',
    country: 'Saudi Arabia',
    notes: 'Recent Jeddah flood with full sensor coverage'
  },
  
  // === SOUTH ASIA (Monsoon - challenging but important) ===
  {
    name: 'Kerala_2018',
    lat: 9.9312,
    lon: 76.2673,
    buffer_km: 40,
    flood_start: '2018-08-08',
    flood_end: '2018-08-30',
    type: 'Monsoon',
    country: 'India',
    notes: 'Extended flooding, may have post-monsoon clearing'
  },
  {
    name: 'Chennai_2015',
    lat: 13.0827,
    lon: 80.2707,
    buffer_km: 35,
    flood_start: '2015-11-15',
    flood_end: '2015-12-05',
    type: 'Fluvial',
    country: 'India',
    notes: 'December floods, post-monsoon'
  },
  {
    name: 'Pakistan_Sindh_2022',
    lat: 25.3960,
    lon: 68.3578,
    buffer_km: 50,
    flood_start: '2022-08-01',
    flood_end: '2022-09-30',
    type: 'Monsoon',
    country: 'Pakistan',
    notes: 'Massive extent, some clearing periods possible'
  },
  {
    name: 'Dhaka_2004',
    lat: 23.8103,
    lon: 90.4125,
    buffer_km: 35,
    flood_start: '2004-07-15',
    flood_end: '2004-09-15',
    type: 'Monsoon',
    country: 'Bangladesh',
    notes: 'Pre-Sentinel, benchmark for transfer target'
  }
];

// ============================================================
// HELPER FUNCTIONS
// ============================================================

function createAOI(event) {
  var point = ee.Geometry.Point([event.lon, event.lat]);
  return point.buffer(event.buffer_km * 1000);
}

function getUrbanFraction(aoi) {
  var urbanPixels = urbanMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: aoi,
    scale: 100,
    maxPixels: 1e9
  });
  return urbanPixels.get('Map');
}

// Parse date string to milliseconds for comparison
function dateToMillis(dateStr) {
  return ee.Date(dateStr).millis();
}

// ============================================================
// SENTINEL-1 QUERY
// ============================================================

function querySentinel1(aoi, startDate, endDate) {
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'));
  
  return s1.map(function(img) {
    var incAngle = img.select('angle').reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: aoi,
      scale: 100,
      maxPixels: 1e8
    }).get('angle');
    
    return ee.Feature(null, {
      'sensor': 'Sentinel-1',
      'type': 'SAR',
      'date': img.date().format('YYYY-MM-dd'),
      'datetime': img.date().format('YYYY-MM-dd HH:mm:ss'),
      'millis': img.date().millis(),
      'orbit_direction': img.get('orbitProperties_pass'),
      'relative_orbit': img.get('relativeOrbitNumber_start'),
      'incidence_angle': incAngle,
      'polarization': 'VV+VH',
      'resolution_m': 10,
      'band': 'C-band',
      'image_id': img.id()
    });
  });
}

// ============================================================
// ALOS-2 PALSAR-2 QUERY
// ============================================================

function queryALOS2(aoi, startDate, endDate) {
  var alos = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
    .filterBounds(aoi)
    .filterDate(startDate, endDate);
  
  return alos.map(function(img) {
    return ee.Feature(null, {
      'sensor': 'ALOS-2',
      'type': 'SAR',
      'date': img.date().format('YYYY-MM-dd'),
      'datetime': img.date().format('YYYY-MM-dd HH:mm:ss'),
      'millis': img.date().millis(),
      'orbit_direction': img.get('PassDirection'),
      'relative_orbit': 'N/A',
      'incidence_angle': 'Variable (ScanSAR)',
      'polarization': 'HH+HV',
      'resolution_m': 25,
      'band': 'L-band',
      'image_id': img.id()
    });
  });
}

// ============================================================
// SENTINEL-2 QUERY
// ============================================================

function querySentinel2(aoi, startDate, endDate) {
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(startDate, endDate);
  
  return s2.map(function(img) {
    var cloudPct = img.get('CLOUDY_PIXEL_PERCENTAGE');
    
    return ee.Feature(null, {
      'sensor': 'Sentinel-2',
      'type': 'Optical',
      'date': img.date().format('YYYY-MM-dd'),
      'datetime': img.date().format('YYYY-MM-dd HH:mm:ss'),
      'millis': img.date().millis(),
      'cloud_percent': cloudPct,
      'resolution_m': 10,
      'image_id': img.id(),
      'usable': ee.Algorithms.If(ee.Number(cloudPct).lt(CLOUD_THRESHOLD), 'yes', 'no')
    });
  });
}

// ============================================================
// LANDSAT 8/9 QUERY
// ============================================================

function queryLandsat(aoi, startDate, endDate) {
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(startDate, endDate);
  
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(startDate, endDate);
  
  var landsat = l8.merge(l9);
  
  return landsat.map(function(img) {
    var cloudPct = img.get('CLOUD_COVER');
    var spacecraft = img.get('SPACECRAFT_ID');
    
    return ee.Feature(null, {
      'sensor': spacecraft,
      'type': 'Optical',
      'date': img.date().format('YYYY-MM-dd'),
      'datetime': img.date().format('YYYY-MM-dd HH:mm:ss'),
      'millis': img.date().millis(),
      'cloud_percent': cloudPct,
      'resolution_m': 30,
      'image_id': img.id(),
      'usable': ee.Algorithms.If(ee.Number(cloudPct).lt(CLOUD_THRESHOLD), 'yes', 'no')
    });
  });
}

// ============================================================
// COINCIDENT PAIR FINDER
// ============================================================

function findCoincidentPairs(sarFeatures, opticalFeatures, windowDays) {
  // Find SAR-Optical pairs within temporal window
  // Returns a feature collection of valid pairs
  
  var windowMillis = windowDays * 24 * 60 * 60 * 1000;
  
  // Filter to usable optical only
  var usableOptical = opticalFeatures.filter(ee.Filter.eq('usable', 'yes'));
  
  // For each SAR image, find matching optical images
  var pairs = sarFeatures.map(function(sarFeat) {
    var sarMillis = ee.Number(sarFeat.get('millis'));
    var sarDate = sarFeat.get('date');
    var sarSensor = sarFeat.get('sensor');
    var sarOrbit = sarFeat.get('orbit_direction');
    var sarIncAngle = sarFeat.get('incidence_angle');
    var sarImageId = sarFeat.get('image_id');
    
    // Find optical images within window
    var matchingOptical = usableOptical.filter(
      ee.Filter.and(
        ee.Filter.gte('millis', sarMillis.subtract(windowMillis)),
        ee.Filter.lte('millis', sarMillis.add(windowMillis))
      )
    );
    
    // Get count and best match (lowest cloud)
    var matchCount = matchingOptical.size();
    var bestMatch = ee.Algorithms.If(
      matchCount.gt(0),
      matchingOptical.sort('cloud_percent').first(),
      null
    );
    
    // Create pair feature
    return ee.Feature(null, {
      'sar_sensor': sarSensor,
      'sar_date': sarDate,
      'sar_image_id': sarImageId,
      'sar_orbit': sarOrbit,
      'sar_incidence_angle': sarIncAngle,
      'optical_matches': matchCount,
      'best_optical_sensor': ee.Algorithms.If(bestMatch, ee.Feature(bestMatch).get('sensor'), 'none'),
      'best_optical_date': ee.Algorithms.If(bestMatch, ee.Feature(bestMatch).get('date'), 'none'),
      'best_optical_cloud': ee.Algorithms.If(bestMatch, ee.Feature(bestMatch).get('cloud_percent'), 999),
      'best_optical_id': ee.Algorithms.If(bestMatch, ee.Feature(bestMatch).get('image_id'), 'none'),
      'has_pair': ee.Algorithms.If(matchCount.gt(0), 'yes', 'no')
    });
  });
  
  return pairs;
}

// ============================================================
// MAIN ANALYSIS FUNCTION
// ============================================================

function analyzeFloodEvent(event) {
  var aoi = createAOI(event);
  var urbanFrac = getUrbanFraction(aoi);
  
  // Query all sensors
  var s1 = querySentinel1(aoi, event.flood_start, event.flood_end);
  var alos2 = queryALOS2(aoi, event.flood_start, event.flood_end);
  var s2 = querySentinel2(aoi, event.flood_start, event.flood_end);
  var landsat = queryLandsat(aoi, event.flood_start, event.flood_end);
  
  // Merge optical sources
  var allOptical = s2.merge(landsat);
  
  // Find coincident pairs
  var s1Pairs = findCoincidentPairs(s1, allOptical, TEMPORAL_WINDOW_DAYS);
  var alos2Pairs = findCoincidentPairs(alos2, allOptical, TEMPORAL_WINDOW_DAYS);
  
  // Add event metadata to each feature
  function addEventMeta(fc, eventName, eventType, country, notes) {
    return fc.map(function(f) {
      return f.set({
        'event_name': eventName,
        'event_type': eventType,
        'country': country,
        'notes': notes,
        'urban_fraction': urbanFrac
      });
    });
  }
  
  var s1_meta = addEventMeta(s1, event.name, event.type, event.country, event.notes);
  var alos2_meta = addEventMeta(alos2, event.name, event.type, event.country, event.notes);
  var s2_meta = addEventMeta(s2, event.name, event.type, event.country, event.notes);
  var landsat_meta = addEventMeta(landsat, event.name, event.type, event.country, event.notes);
  var s1Pairs_meta = addEventMeta(s1Pairs, event.name, event.type, event.country, event.notes);
  var alos2Pairs_meta = addEventMeta(alos2Pairs, event.name, event.type, event.country, event.notes);
  
  return {
    s1: s1_meta,
    alos2: alos2_meta,
    s2: s2_meta,
    landsat: landsat_meta,
    s1_pairs: s1Pairs_meta,
    alos2_pairs: alos2Pairs_meta,
    aoi: aoi,
    urban_fraction: urbanFrac
  };
}

// ============================================================
// SUMMARY STATISTICS FUNCTION
// ============================================================

function printEventSummary(event, results) {
  var s2Usable = results.s2.filter(ee.Filter.eq('usable', 'yes')).size();
  var landsatUsable = results.landsat.filter(ee.Filter.eq('usable', 'yes')).size();
  var s1WithPairs = results.s1_pairs.filter(ee.Filter.eq('has_pair', 'yes')).size();
  var alos2WithPairs = results.alos2_pairs.filter(ee.Filter.eq('has_pair', 'yes')).size();
  
  print('========================================');
  print('EVENT: ' + event.name);
  print('Type: ' + event.type + ' | Country: ' + event.country);
  print('Period: ' + event.flood_start + ' to ' + event.flood_end);
  print('Notes: ' + event.notes);
  print('Urban fraction:', results.urban_fraction);
  print('----------------------------------------');
  print('SAR Coverage:');
  print('  Sentinel-1 images:', results.s1.size());
  print('  ALOS-2 images:', results.alos2.size());
  print('Optical Coverage:');
  print('  Sentinel-2 total:', results.s2.size(), '| usable (<' + CLOUD_THRESHOLD + '% cloud):', s2Usable);
  print('  Landsat 8/9 total:', results.landsat.size(), '| usable:', landsatUsable);
  print('COINCIDENT PAIRS (within ' + TEMPORAL_WINDOW_DAYS + ' days):');
  print('  S1 with usable optical:', s1WithPairs);
  print('  ALOS-2 with usable optical:', alos2WithPairs);
  print('========================================');
}

// ============================================================
// RUN ANALYSIS FOR SELECTED EVENT
// ============================================================

// Change this index to analyze different events
var EVENT_INDEX = 0;
var selectedEvent = floodEvents[EVENT_INDEX];

var results = analyzeFloodEvent(selectedEvent);

// Print summary
printEventSummary(selectedEvent, results);

// Print detailed results
print('');
print('--- DETAILED SENTINEL-1 DATA ---');
print(results.s1);

print('');
print('--- DETAILED ALOS-2 DATA ---');
print(results.alos2);

print('');
print('--- S1 COINCIDENT PAIRS ---');
print(results.s1_pairs);

print('');
print('--- ALOS-2 COINCIDENT PAIRS ---');
print(results.alos2_pairs);

print('');
print('--- SENTINEL-2 DATA ---');
print(results.s2);

print('');
print('--- LANDSAT DATA ---');
print(results.landsat);

// Visualize AOI
Map.centerObject(results.aoi, 10);
Map.addLayer(results.aoi, {color: 'blue'}, 'AOI');
Map.addLayer(urbanMask.clip(results.aoi), {palette: ['white', 'red']}, 'Urban areas');

// ============================================================
// EXPORT INDIVIDUAL EVENT DATA
// ============================================================

// Export coincident pairs (most useful)
Export.table.toDrive({
  collection: results.s1_pairs.merge(results.alos2_pairs),
  description: selectedEvent.name + '_coincident_pairs',
  fileFormat: 'CSV'
});

// Export all sensor data
var allResults = results.s1
  .merge(results.alos2)
  .merge(results.s2)
  .merge(results.landsat);

Export.table.toDrive({
  collection: allResults,
  description: selectedEvent.name + '_all_satellite_data',
  fileFormat: 'CSV'
});

// ============================================================
// BATCH ANALYSIS - ALL EVENTS SUMMARY
// ============================================================

print('');
print('==========================================================');
print('QUICK REFERENCE: ALL FLOOD EVENTS (' + floodEvents.length + ' total)');
print('==========================================================');
print('');

// Categorize by region/latitude
print('--- MID-LATITUDE FOCUS (30-50°N/S) - Best Coverage ---');
floodEvents.forEach(function(event, index) {
  var absLat = Math.abs(event.lat);
  if (absLat >= 30 && absLat <= 50) {
    print(index + ': ' + event.name + ' (' + event.type + ') [' + event.lat.toFixed(1) + '°]');
  }
});

print('');
print('--- HIGH LATITUDE (>50°N/S) ---');
floodEvents.forEach(function(event, index) {
  var absLat = Math.abs(event.lat);
  if (absLat > 50) {
    print(index + ': ' + event.name + ' (' + event.type + ') [' + event.lat.toFixed(1) + '°]');
  }
});

print('');
print('--- TROPICAL/SUBTROPICAL (<30°) ---');
floodEvents.forEach(function(event, index) {
  var absLat = Math.abs(event.lat);
  if (absLat < 30) {
    print(index + ': ' + event.name + ' (' + event.type + ') [' + event.lat.toFixed(1) + '°]');
  }
});

// ============================================================
// BATCH EXPORT (uncomment to run all events)
// ============================================================

/*
// Warning: This generates many export tasks

floodEvents.forEach(function(event, index) {
  var results = analyzeFloodEvent(event);
  
  // Export coincident pairs
  var pairs = results.s1_pairs.merge(results.alos2_pairs);
  Export.table.toDrive({
    collection: pairs,
    description: event.name + '_coincident_pairs',
    fileFormat: 'CSV',
    folder: 'water_detection_discovery'
  });
  
  // Export all data
  var allResults = results.s1
    .merge(results.alos2)
    .merge(results.s2)
    .merge(results.landsat);
  Export.table.toDrive({
    collection: allResults,
    description: event.name + '_all_satellite_data',
    fileFormat: 'CSV',
    folder: 'water_detection_discovery'
  });
});
*/
