/*
 * export_aux_pseudo.js
 *
 * Exports auxiliary context features for new pseudo-label events.
 * Mirrors export_aux_features.js exactly (same datasets, same normalisation)
 * but targets the new pseudo-label AOIs.
 *
 * Exports per event (4 files):
 *   {EventName}_buildings.tif    -- Google Open Buildings count density at 100m
 *   {EventName}_worldcover.tif   -- ESA WorldCover built-up fraction [0,1] at 100m
 *   {EventName}_hand.tif         -- HAND elevation clipped [0,30m] at 100m
 *   {EventName}_angle.tif        -- S1 incidence angle [20,46deg]->[0,1] at 100m
 *
 * Output folder: water_detection_aux_pseudo/
 * Download to:   data/pseudolabels/aux/
 *
 * These are optional but important -- precompute_pseudo_chips.py falls back
 * to neutral values (0.0 / 0.5) if missing, but that creates a domain gap
 * at fine-tuning time where real aux features ARE present.
 */

var WORLDCOVER = ee.ImageCollection('ESA/WorldCover/v200').first();
var HAND       = ee.Image('MERIT/Hydro/v1_0_1').select('hnd');
var BUILDINGS  = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons');

var EVENTS = [
  { name: 'Brisbane_2022',        lon:  153.0251, lat:  -27.4698, buffer_km: 60,
    sar_date: '2022-03-07' },
  { name: 'Sydney_2022',          lon:  150.7930, lat:  -33.8688, buffer_km: 60,
    sar_date: '2022-03-10' },
  { name: 'Valencia_Spain_2024',  lon:   -0.3763, lat:   39.4699, buffer_km: 50,
    sar_date: '2024-10-31' },
  { name: 'Liege_Belgium_2021_A', lon:    5.5797, lat:   50.6326, buffer_km: 40,
    sar_date: '2021-07-18' },
  { name: 'Liege_Belgium_2021_B', lon:    5.5797, lat:   50.6326, buffer_km: 40,
    sar_date: '2021-07-22' },
  { name: 'Cologne_Germany_2021', lon:    6.9603, lat:   50.9333, buffer_km: 50,
    sar_date: '2021-07-18' },
  { name: 'Zhengzhou_China_2021', lon:  113.6254, lat:   34.7466, buffer_km: 50,
    sar_date: '2021-07-27' },
  { name: 'Derna_Libya_2023_A',   lon:   22.6436, lat:   32.7558, buffer_km: 30,
    sar_date: '2023-09-12' },
  { name: 'Derna_Libya_2023_B',   lon:   22.6436, lat:   32.7558, buffer_km: 30,
    sar_date: '2023-09-20' },
  { name: 'Omaha_USA_2019',       lon:  -96.0082, lat:   41.2565, buffer_km: 50,
    sar_date: '2019-03-29' },
  { name: 'Emilia_Romagna_2023',  lon:   11.3426, lat:   44.4949, buffer_km: 60,
    sar_date: '2023-05-22' },
  { name: 'Rio_Grande_Sul_2024',  lon:  -51.2177, lat:  -30.0346, buffer_km: 70,
    sar_date: '2024-05-08' },
];

print('=== Aux feature exports for pseudo-label events ===');
print(EVENTS.length + ' events x 4 features = ' + (EVENTS.length * 4) + ' tasks');
print('');

EVENTS.forEach(function(event) {
  var aoi    = ee.Geometry.Point([event.lon, event.lat]).buffer(event.buffer_km * 1000);
  var region = aoi.bounds();

  // ── 1. Google Open Buildings density ─────────────────────────────────
  var buildings = BUILDINGS
    .filterBounds(aoi)
    .reduceToImage(['confidence'], ee.Reducer.count())
    .rename('buildings')
    .unmask(0)
    .toFloat();

  Export.image.toDrive({
    image:          buildings,
    description:    event.name + '_buildings',
    folder:         'water_detection_aux_pseudo',
    fileNamePrefix: event.name + '_buildings',
    region:         region,
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  // ── 2. WorldCover built-up fraction ──────────────────────────────────
  var builtup = WORLDCOVER.eq(50)
    .reduceResolution({
      reducer:    ee.Reducer.mean(),
      bestEffort: true,
      maxPixels:  1024,
    })
    .reproject({ crs: 'EPSG:4326', scale: 100 })
    .rename('builtup_fraction')
    .unmask(0)
    .toFloat();

  Export.image.toDrive({
    image:          builtup,
    description:    event.name + '_worldcover',
    folder:         'water_detection_aux_pseudo',
    fileNamePrefix: event.name + '_worldcover',
    region:         region,
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  // ── 3. HAND elevation (clipped at 30m) ────────────────────────────────
  var hand = HAND.unmask(30).min(30).toFloat();

  Export.image.toDrive({
    image:          hand,
    description:    event.name + '_hand',
    folder:         'water_detection_aux_pseudo',
    fileNamePrefix: event.name + '_hand',
    region:         region,
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  // ── 4. Sentinel-1 incidence angle ─────────────────────────────────────
  // Incidence angle is stable per location+orbit, so +-7 days is fine.
  var start = ee.Date(event.sar_date).advance(-7, 'day');
  var end   = ee.Date(event.sar_date).advance( 7, 'day');

  var angle = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select('angle')
    .first()
    .subtract(20).divide(26)   // normalise [20, 46] -> [0, 1]
    .clamp(0, 1)
    .unmask(0.5)               // 0.5 = 33 degrees midrange fallback
    .toFloat();

  Export.image.toDrive({
    image:          angle,
    description:    event.name + '_angle',
    folder:         'water_detection_aux_pseudo',
    fileNamePrefix: event.name + '_angle',
    region:         region,
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  print('Queued: ' + event.name + ' (buildings + worldcover + hand + angle)');
});

print('');
print('Total: ' + (EVENTS.length * 4) + ' tasks queued.');
print('After download, move files to: data/pseudolabels/aux/');
