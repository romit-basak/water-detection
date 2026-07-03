/*
 * export_building_density.js
 *
 * Exports Google Open Buildings count density as a 100m raster
 * for all UrbanSARFloods events in one run.
 *
 * Each pixel = number of building footprints whose centroid falls
 * within that 100m cell. Uses reduceToImage which is the correct
 * GEE API for rasterising a FeatureCollection by count.
 *
 * Output: Drive → water_detection_building_density/
 *   {EventName}_building_density.tif
 *
 * Coverage note: Google Open Buildings v3 covers Africa, Asia,
 * Pacific islands, and parts of Europe/Oceania.
 * Canada and parts of Iran may have sparse coverage.
 */

var EVENTS = [
  // ── Test events ───────────────────────────────────────────────
  { name: 'Hagibis',       lon: 139.6503, lat:  35.6762, buffer_km: 60 },
  { name: 'Sydney',        lon: 150.9000, lat: -33.8688, buffer_km: 50 },
  { name: 'Coraki',        lon: 153.2750, lat: -28.8133, buffer_km: 25 },
  { name: 'Niger',         lon:   6.3667, lat:   7.8333, buffer_km: 40 },
  { name: 'Hebei',         lon: 115.4000, lat:  39.3054, buffer_km: 60 },
  { name: 'Beledweyne',    lon:  45.2000, lat:   4.7361, buffer_km: 30 },
  { name: 'PortMacquarie', lon: 152.9086, lat: -31.4298, buffer_km: 25 },
  // ── Train events ──────────────────────────────────────────────
  { name: 'Houston',       lon: -95.3698, lat:  29.7604, buffer_km: 60 },
  { name: 'Beira',         lon:  34.8389, lat: -19.8436, buffer_km: 40 },
  { name: 'Japan',         lon: 133.0000, lat:  34.2000, buffer_km: 60 },
  { name: 'Canada',        lon:-116.0000, lat:  55.0000, buffer_km: 60 },
  { name: 'Iran',          lon:  48.0000, lat:  31.5000, buffer_km: 60 },
  { name: 'Lumberton',     lon: -79.0742, lat:  34.6182, buffer_km: 30 },
];

// Google Open Buildings v3 — polygon footprints
var BUILDINGS = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons');

print('Queueing ' + EVENTS.length + ' building density exports...');

EVENTS.forEach(function(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);

  // Filter buildings to AOI
  var local = BUILDINGS.filterBounds(aoi);

  // Rasterise: count footprints per 100m pixel using reduceToImage.
  // 'confidence' is a numeric property on every feature — counting it
  // gives the number of buildings per pixel. unmask(0) fills empty pixels.
  var density = local
    .reduceToImage(['confidence'], ee.Reducer.count())
    .rename('building_count')
    .unmask(0)
    .toFloat();

  Export.image.toDrive({
    image:          density,
    description:    event.name + '_building_density',
    folder:         'water_detection_building_density',
    fileNamePrefix: event.name + '_building_density',
    region:         aoi.bounds(),
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  print('Queued: ' + event.name);
});

print('Done — ' + EVENTS.length + ' tasks queued.');
print('Go to Tasks tab and click Run on each task.');
print('Files will appear in Drive: water_detection_building_density/');
