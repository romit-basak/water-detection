/*
 * export_aux_features.js
 *
 * Exports two additional auxiliary feature rasters for all UrbanSARFloods
 * events:
 *
 *  1. ESA WorldCover v200 (2021) — 10m land cover
 *     Resampled to 100m, exported as built-up fraction per pixel
 *     (fraction of 10m pixels classified as class 50 = built-up)
 *
 *  2. HAND (Height Above Nearest Drainage) from MERIT Hydro
 *     At 90m resolution, resampled to 100m
 *     Key hydrological prior: low HAND → more likely to flood
 *
 * Output: Drive → water_detection_aux_features/
 *   {EventName}_worldcover.tif   — built-up fraction [0, 1] at 100m
 *   {EventName}_hand.tif         — HAND elevation (metres) at 100m
 *
 * Run this script once. All 13 events are queued in a single execution.
 */

var EVENTS = [
  { name: 'Hagibis',       lon: 139.6503, lat:  35.6762, buffer_km: 60 },
  { name: 'Sydney',        lon: 150.9000, lat: -33.8688, buffer_km: 50 },
  { name: 'Coraki',        lon: 153.2750, lat: -28.8133, buffer_km: 25 },
  { name: 'Niger',         lon:   6.3667, lat:   7.8333, buffer_km: 40 },
  { name: 'Hebei',         lon: 115.4000, lat:  39.3054, buffer_km: 60 },
  { name: 'Beledweyne',    lon:  45.2000, lat:   4.7361, buffer_km: 30 },
  { name: 'PortMacquarie', lon: 152.9086, lat: -31.4298, buffer_km: 25 },
  { name: 'Houston',       lon: -95.3698, lat:  29.7604, buffer_km: 60 },
  { name: 'Beira',         lon:  34.8389, lat: -19.8436, buffer_km: 40 },
  { name: 'Japan',         lon: 133.0000, lat:  34.2000, buffer_km: 60 },
  { name: 'Canada',        lon:-116.0000, lat:  55.0000, buffer_km: 60 },
  { name: 'Iran',          lon:  48.0000, lat:  31.5000, buffer_km: 60 },
  { name: 'Lumberton',     lon: -79.0742, lat:  34.6182, buffer_km: 30 },
];

// ESA WorldCover 2021 — class 50 = built-up area
var WORLDCOVER = ee.ImageCollection('ESA/WorldCover/v200').first();

// MERIT Hydro HAND — Height Above Nearest Drainage
// Best available globally consistent HAND dataset
var HAND = ee.Image('MERIT/Hydro/v1_0_1').select('hnd');

print('Queueing ' + (EVENTS.length * 2) + ' aux feature exports...');

// Event date lookup for incidence angle — need an acquisition near the flood event
// Using approximate peak flood dates for each event
var EVENT_DATES = {
  'Hagibis':       ['2019-10-12', '2019-10-15'],
  'Sydney':        ['2022-07-05', '2022-07-08'],
  'Coraki':        ['2022-03-02', '2022-03-05'],
  'Niger':         ['2022-10-13', '2022-10-16'],
  'Hebei':         ['2023-08-05', '2023-08-08'],
  'Beledweyne':    ['2023-11-14', '2023-11-17'],
  'PortMacquarie': ['2021-03-19', '2021-03-22'],
  'Houston':       ['2017-08-30', '2017-09-02'],
  'Beira':         ['2019-03-20', '2019-03-23'],
  'Japan':         ['2019-10-12', '2019-10-15'],
  'Canada':        ['2019-05-02', '2019-05-05'],
  'Iran':          ['2019-03-29', '2019-04-01'],
  'Lumberton':     ['2016-10-11', '2016-10-14'],
};

EVENTS.forEach(function(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);

  // ── WorldCover: built-up fraction at 100m ──────────────────────────────
  // Resample 10m WorldCover to 100m by computing the fraction of
  // built-up pixels (class 50) within each 100m cell
  var builtup = WORLDCOVER.eq(50).rename('builtup_fraction');
  var builtup_100m = builtup
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      bestEffort: true,
      maxPixels: 1024,
    })
    .reproject({ crs: 'EPSG:4326', scale: 100 })
    .unmask(0)
    .toFloat();

  Export.image.toDrive({
    image:          builtup_100m,
    description:    event.name + '_worldcover',
    folder:         'water_detection_aux_features',
    fileNamePrefix: event.name + '_worldcover',
    region:         aoi.bounds(),
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  // ── HAND elevation at 100m ─────────────────────────────────────────────
  // Clip to 30m max — pixels above 30m HAND are essentially never flooded
  // and clipping avoids large outlier values distorting normalisation
  var hand_clipped = HAND
    .unmask(30)           // fill nodata (ocean/outside) with 30m (high HAND)
    .min(30)              // clip max at 30m
    .toFloat();

  Export.image.toDrive({
    image:          hand_clipped,
    description:    event.name + '_hand',
    folder:         'water_detection_aux_features',
    fileNamePrefix: event.name + '_hand',
    region:         aoi.bounds(),
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  // ── Incidence angle from Sentinel-1 acquisition ──────────────────────
  // The 'angle' band gives local incidence angle in degrees.
  // Sentinel-1 IW range: ~20° (near range) to ~46° (far range).
  // Normalised to [0, 1] over [20, 46] degrees.
  // A single acquisition near the flood event date is sufficient —
  // incidence angle for a given location is stable across acquisitions
  // from the same orbit direction.
  var dates = EVENT_DATES[event.name];
  if (dates) {
    var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(aoi)
      .filterDate(dates[0], dates[1])
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .select('angle')
      .first();

    // Fallback: if no acquisition in date range, use a wider window
    var s1_fallback = ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(aoi)
      .filterDate(
        ee.Date(dates[0]).advance(-14, 'day').format('YYYY-MM-dd'),
        ee.Date(dates[1]).advance( 14, 'day').format('YYYY-MM-dd'))
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .select('angle')
      .first();

    var angle_image = ee.Image(ee.Algorithms.If(
      s1, s1, s1_fallback
    ));

    // Normalise [20, 46] → [0, 1]
    var angle_norm = angle_image
      .subtract(20).divide(26)
      .clamp(0, 1)
      .unmask(0.5)  // 0.5 = midrange (33°) for missing pixels
      .toFloat();

    Export.image.toDrive({
      image:          angle_norm,
      description:    event.name + '_incidence_angle',
      folder:         'water_detection_aux_features',
      fileNamePrefix: event.name + '_incidence_angle',
      region:         aoi.bounds(),
      scale:          100,
      crs:            'EPSG:4326',
      maxPixels:      1e11,
      fileFormat:     'GeoTIFF',
    });
  }

  print('Queued: ' + event.name + ' (worldcover + hand + incidence_angle)');
});

print('Done — ' + (EVENTS.length * 2) + ' tasks queued.');
print('Go to Tasks tab and click Run All.');
print('Files → Drive: water_detection_aux_features/');
