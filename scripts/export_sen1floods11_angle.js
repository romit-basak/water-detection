/*
 * export_sen1floods11_angle.js
 *
 * Exports Sentinel-1 incidence angle rasters for all 11 Sen1Floods11
 * event regions. Normalised [20, 46] degrees -> [0, 1].
 *
 * Output: Drive -> water_detection_aux_features/
 *   {EventName}_s1_incidence_angle.tif
 *
 * Run once in GEE, then download to data/aux_features/.
 */

var EVENTS = [
  // name must match the prefix in Sen1Floods11 filenames exactly
  { name: 'Bolivia',   lon: -65.24, lat: -14.33, date_start: '2018-02-14', date_end: '2018-02-21', buffer_km: 80  },
  { name: 'Ghana',     lon:  -0.50, lat:   7.50, date_start: '2017-06-10', date_end: '2017-06-24', buffer_km: 100 },
  { name: 'India',     lon:  76.50, lat:  16.00, date_start: '2018-08-10', date_end: '2018-08-31', buffer_km: 150 },
  { name: 'Mekong',   lon: 103.00, lat:  14.00, date_start: '2019-09-01', date_end: '2019-09-20', buffer_km: 120 },
  { name: 'Nigeria',   lon:   7.00, lat:   9.00, date_start: '2017-07-01', date_end: '2017-07-20', buffer_km: 100 },
  { name: 'Pakistan',  lon:  70.00, lat:  30.00, date_start: '2019-09-10', date_end: '2019-09-30', buffer_km: 150 },
  { name: 'Paraguay',  lon: -58.00, lat: -20.00, date_start: '2018-04-01', date_end: '2018-04-20', buffer_km: 120 },
  { name: 'Somalia',   lon:  43.00, lat:   5.00, date_start: '2016-05-01', date_end: '2016-05-20', buffer_km: 100 },
  { name: 'Spain',     lon:  -0.50, lat:  39.00, date_start: '2019-09-10', date_end: '2019-09-30', buffer_km: 100 },
  { name: 'Sri-Lanka', lon:  81.00, lat:   7.50, date_start: '2017-05-25', date_end: '2017-06-10', buffer_km: 100 },
  { name: 'USA',       lon: -91.00, lat:  30.00, date_start: '2017-08-26', date_end: '2017-09-05', buffer_km: 150 },
];

print('Queueing ' + EVENTS.length + ' incidence angle exports for Sen1Floods11...');

EVENTS.forEach(function(event) {
  var aoi = ee.Geometry.Point([event.lon, event.lat])
              .buffer(event.buffer_km * 1000);

  // Get incidence angle from a Sentinel-1 acquisition near the flood event
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(event.date_start, event.date_end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select('angle')
    .first();

  // Fallback: widen the date window if no acquisition found
  var s1_wide = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(
      ee.Date(event.date_start).advance(-20, 'day').format('YYYY-MM-dd'),
      ee.Date(event.date_end).advance(20, 'day').format('YYYY-MM-dd')
    )
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select('angle')
    .first();

  var angle_image = ee.Image(ee.Algorithms.If(s1, s1, s1_wide));

  // Normalise [20, 46] -> [0, 1]
  var angle_norm = angle_image
    .subtract(20).divide(26)
    .clamp(0, 1)
    .unmask(0.5)   // 0.5 = 33 degrees midrange for any gaps
    .toFloat();

  Export.image.toDrive({
    image:          angle_norm,
    description:    event.name + '_s1_incidence_angle',
    folder:         'water_detection_aux_features',
    fileNamePrefix: event.name + '_s1_incidence_angle',
    region:         aoi.bounds(),
    scale:          100,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  print('Queued: ' + event.name);
});

print('Done. Run all tasks in the Tasks tab.');
print('Download to: data/aux_features/');
