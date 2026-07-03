/*
 * export_sar_pseudo.js
 *
 * Exports Sentinel-1 GRD SAR imagery (VV + VH) for new pseudo-label events.
 * These are the SAR GeoTIFFs that precompute_pseudo_chips.py chips into .npy.
 *
 * Output: Google Drive --> water_detection_pseudolabels_sar/
 *   {EventName}_VV.tif   (float32, linear power, unitless)
 *   {EventName}_VH.tif   (float32, linear power, unitless)
 *
 * IMPORTANT: exported as LINEAR POWER, not dB.
 * precompute_pseudo_chips.py applies:  10 * log10(x), clip [-30, 0], normalise to [0,1]
 * Do NOT apply .log() or toDb() here.
 *
 * SAR dates are pinned from discovery_summary.csv.
 * After download, move files to:
 *   data/pseudolabels/sar/{EventName}_VV.tif
 *   data/pseudolabels/sar/{EventName}_VH.tif
 */

var PSEUDO_SAR_EVENTS = [

  // Australia
  { name: 'Brisbane_2022',        lon:  153.0251, lat:  -27.4698, buffer_km: 60,
    sar_date: '2022-03-07', orbit_dir: 'DESCENDING' },
  { name: 'Sydney_2022',          lon:  150.7930, lat:  -33.8688, buffer_km: 60,
    sar_date: '2022-03-10', orbit_dir: 'ASCENDING' },

  // Spain
  { name: 'Valencia_Spain_2024',  lon:   -0.3763, lat:   39.4699, buffer_km: 50,
    sar_date: '2024-10-31', orbit_dir: 'ASCENDING' },

  // Belgium
  { name: 'Liege_Belgium_2021_A', lon:    5.5797, lat:   50.6326, buffer_km: 40,
    sar_date: '2021-07-18', orbit_dir: 'ASCENDING' },
  { name: 'Liege_Belgium_2021_B', lon:    5.5797, lat:   50.6326, buffer_km: 40,
    sar_date: '2021-07-22', orbit_dir: 'DESCENDING' },

  // Germany
  { name: 'Cologne_Germany_2021', lon:    6.9603, lat:   50.9333, buffer_km: 50,
    sar_date: '2021-07-18', orbit_dir: 'ASCENDING' },

  // China
  { name: 'Zhengzhou_China_2021', lon:  113.6254, lat:   34.7466, buffer_km: 50,
    sar_date: '2021-07-27', orbit_dir: 'ASCENDING' },

  // Libya
  { name: 'Derna_Libya_2023_A',   lon:   22.6436, lat:   32.7558, buffer_km: 30,
    sar_date: '2023-09-12', orbit_dir: 'ASCENDING' },
  { name: 'Derna_Libya_2023_B',   lon:   22.6436, lat:   32.7558, buffer_km: 30,
    sar_date: '2023-09-20', orbit_dir: 'DESCENDING' },

  // USA
  { name: 'Omaha_USA_2019',       lon:  -96.0082, lat:   41.2565, buffer_km: 50,
    sar_date: '2019-03-29', orbit_dir: 'ASCENDING' },

  // Italy
  { name: 'Emilia_Romagna_2023',  lon:   11.3426, lat:   44.4949, buffer_km: 60,
    sar_date: '2023-05-22', orbit_dir: 'DESCENDING' },

  // Brazil
  { name: 'Rio_Grande_Sul_2024',  lon:  -51.2177, lat:  -30.0346, buffer_km: 70,
    sar_date: '2024-05-08', orbit_dir: 'ASCENDING' },
];

// =============================================================================
// EXPORT FUNCTION
// =============================================================================

function exportSAR(event) {
  var aoi   = ee.Geometry.Point([event.lon, event.lat]).buffer(event.buffer_km * 1000);
  var start = ee.Date(event.sar_date).advance(-1, 'day');
  var end   = ee.Date(event.sar_date).advance( 1, 'day');
  var region = aoi.bounds();

  // Primary: filter by pinned orbit direction
  var col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', event.orbit_dir))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  // Fallback: any orbit direction (handles occasional GEE metadata quirks)
  var col_fallback = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

  var scene = ee.Image(ee.Algorithms.If(
    col.size().gt(0), col.first(), col_fallback.first()
  ));

  // GEE S1 GRD is in dB. Convert to linear power: 10^(dB/10).
  // precompute_pseudo_chips.py converts back: 10*log10(linear).
  // Unmask to 0 for pixels outside the SAR swath (treated as nodata in chipping).
  var vv = ee.Image(10).pow(scene.select('VV').divide(10)).rename('VV').unmask(0).toFloat();
  var vh = ee.Image(10).pow(scene.select('VH').divide(10)).rename('VH').unmask(0).toFloat();

  Export.image.toDrive({
    image:          vv,
    description:    event.name + '_VV',
    folder:         'water_detection_pseudolabels_sar',
    fileNamePrefix: event.name + '_VV',
    region:         region,
    scale:          10,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  Export.image.toDrive({
    image:          vh,
    description:    event.name + '_VH',
    folder:         'water_detection_pseudolabels_sar',
    fileNamePrefix: event.name + '_VH',
    region:         region,
    scale:          10,
    crs:            'EPSG:4326',
    maxPixels:      1e11,
    fileFormat:     'GeoTIFF',
  });

  print('Queued: ' + event.name + '  [' + event.sar_date + ', ' + event.orbit_dir + ']');
}

// =============================================================================
// RUN
// =============================================================================

print('=== Sentinel-1 SAR exports for pseudo-label events ===');
print(PSEUDO_SAR_EVENTS.length + ' events x 2 pol = ' +
      (PSEUDO_SAR_EVENTS.length * 2) + ' tasks');
print('Output: water_detection_pseudolabels_sar/');
print('');

PSEUDO_SAR_EVENTS.forEach(exportSAR);

print('');
print('After download, move to: data/pseudolabels/sar/');
print('Large exports (Brisbane, Sydney, Rio Grande Sul) may take 15-20 min each.');
print('Then run: uv run python scripts/precompute_pseudo_chips.py --dry_run');
