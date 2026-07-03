/*
 * ERA5 Wind & Rain vs SAR Backscatter Validation
 * Purpose: Test if wind speed correlates with water surface backscatter
 *          in open water vs urban canyon environments
 * 
 * Hypothesis: Wind → Roughness → Backscatter holds for open water
 *             but breaks down in urban canyons due to shielding
 * 
 * Author: Romit Basak
 * Date: 2025
 */

// =============================================================================
// 1. DEFINE REGIONS OF INTEREST
// =============================================================================

// Hooghly River - Open water section near Howrah Bridge
// Wide river (~500m), minimal shielding
var hooghly_river = ee.Geometry.Polygon([
  [88.335, 22.595],
  [88.345, 22.595],
  [88.345, 22.575],
  [88.335, 22.575],
  [88.335, 22.595]
]);

// Tolly's Nullah (Adi Ganga) - Narrow urban canal in South Kolkata
// ~30-50m wide, surrounded by dense buildings
var urban_canal = ee.Geometry.Polygon([
  [88.340, 22.505],
  [88.350, 22.505],
  [88.350, 22.495],
  [88.340, 22.495],
  [88.340, 22.505]
]);

// Kestopur Canal - Another urban drainage canal in Salt Lake area
// Narrow channel through residential area
var kestopur_canal = ee.Geometry.Polygon([
  [88.415, 22.595],
  [88.425, 22.595],
  [88.425, 22.585],
  [88.415, 22.585],
  [88.415, 22.595]
]);

// East Kolkata Wetlands - Semi-open water (for comparison)
// Large water bodies but with surrounding development
var ekw_wetlands = ee.Geometry.Polygon([
  [88.42, 22.54],
  [88.44, 22.54],
  [88.44, 22.52],
  [88.42, 22.52],
  [88.42, 22.54]
]);

// Combine all ROIs for visualization
var all_rois = {
  'Hooghly River (Open)': hooghly_river,
  'Tollys Nullah (Urban Canal)': urban_canal,
  'Kestopur Canal (Urban)': kestopur_canal,
  'EKW Wetlands (Semi-Open)': ekw_wetlands
};

// =============================================================================
// 2. VISUALIZATION: Show ROIs on map
// =============================================================================

Map.centerObject(hooghly_river, 12);
Map.addLayer(hooghly_river, {color: 'blue'}, 'Hooghly River (Open)');
Map.addLayer(urban_canal, {color: 'red'}, 'Tollys Nullah (Urban Canal)');
Map.addLayer(kestopur_canal, {color: 'orange'}, 'Kestopur Canal (Urban)');
Map.addLayer(ekw_wetlands, {color: 'green'}, 'EKW Wetlands (Semi-Open)');

// =============================================================================
// 3. DATA LOADING FUNCTIONS
// =============================================================================

/**
 * Load and filter Sentinel-1 GRD data for a region
 */
function loadS1(roi, startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) // Consistent geometry
    .select(['VV', 'VH']);
}

/**
 * Load ERA5 hourly data for a region
 */
function loadERA5(roi, startDate, endDate) {
  return ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .select(['u_component_of_wind_10m', 'v_component_of_wind_10m', 'total_precipitation']);
}

// =============================================================================
// 4. ANALYSIS FUNCTION
// =============================================================================

/**
 * Run wind-backscatter correlation analysis for a given ROI
 * Returns a FeatureCollection with paired SAR-ERA5 observations
 */
function analyzeROI(roi, roiName, startDate, endDate) {
  var s1 = loadS1(roi, startDate, endDate);
  var era5 = loadERA5(roi, startDate, endDate);
  
  // Temporal join: match each SAR image with closest ERA5 hour
  var filterTime = ee.Filter.maxDifference({
    difference: 2 * 60 * 60 * 1000, // 2 hour window
    leftField: 'system:time_start',
    rightField: 'system:time_start'
  });
  
  var saveBestJoin = ee.Join.saveBest({
    matchKey: 'era5_match',
    measureKey: 'time_diff'
  });
  
  var joined = saveBestJoin.apply(s1, era5, filterTime);
  
  // Extract statistics for each paired observation
  var data = joined.map(function(img) {
    var sarImg = ee.Image(img);
    var era5Img = ee.Image(img.get('era5_match'));
    
    // Calculate wind speed magnitude
    var u = era5Img.select('u_component_of_wind_10m');
    var v = era5Img.select('v_component_of_wind_10m');
    var windSpeed = u.pow(2).add(v.pow(2)).sqrt().rename('wind_speed');
    
    // Precipitation (convert m to mm)
    var precip = era5Img.select('total_precipitation').multiply(1000).rename('precip_mm');
    
    // Combine all bands
    var combined = sarImg.addBands(windSpeed).addBands(precip);
    
    // Reduce over ROI
    var stats = combined.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 10, // S1 native resolution
      bestEffort: true
    });
    
    // Create feature with all attributes
    return ee.Feature(null, {
      'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
      'VV': stats.get('VV'),
      'VH': stats.get('VH'),
      'wind_speed': stats.get('wind_speed'),
      'precip_mm': stats.get('precip_mm'),
      'roi_name': roiName
    });
  });
  
  // Filter out any null values
  return data.filter(ee.Filter.notNull(['VV', 'wind_speed']));
}

// =============================================================================
// 5. RUN ANALYSIS
// =============================================================================

// Analysis period: 2018-2024 (multiple monsoon seasons)
var startDate = '2018-01-01';
var endDate = '2024-12-31';

// Run analysis for each ROI
var hooghlyData = analyzeROI(hooghly_river, 'Hooghly River (Open)', startDate, endDate);
var tollyData = analyzeROI(urban_canal, 'Tollys Nullah (Urban)', startDate, endDate);
var kestopurData = analyzeROI(kestopur_canal, 'Kestopur Canal (Urban)', startDate, endDate);
var ekwData = analyzeROI(ekw_wetlands, 'EKW Wetlands (Semi-Open)', startDate, endDate);

// Merge all data
var allData = hooghlyData.merge(tollyData).merge(kestopurData).merge(ekwData);

// =============================================================================
// 6. GENERATE CHARTS
// =============================================================================

// Chart 1: Wind Speed vs VV Backscatter (By ROI)
var windVsVV = ui.Chart.feature.groups(allData, 'wind_speed', 'VV', 'roi_name')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'SAR Backscatter (VV) vs Wind Speed - By Location Type',
    hAxis: {title: 'Wind Speed at 10m (m/s)', minValue: 0, maxValue: 15},
    vAxis: {title: 'VV Backscatter (dB)', minValue: -25, maxValue: -5},
    pointSize: 4,
    colors: ['blue', 'red', 'orange', 'green'],
    legend: {position: 'right'},
    trendlines: {
      0: {color: 'blue', lineWidth: 2, opacity: 0.5, showR2: true, visibleInLegend: true},
      1: {color: 'red', lineWidth: 2, opacity: 0.5, showR2: true, visibleInLegend: true},
      2: {color: 'orange', lineWidth: 2, opacity: 0.5, showR2: true, visibleInLegend: true},
      3: {color: 'green', lineWidth: 2, opacity: 0.5, showR2: true, visibleInLegend: true}
    }
  });

// Chart 2: Precipitation vs VV Backscatter (By ROI)
var precipVsVV = ui.Chart.feature.groups(allData, 'precip_mm', 'VV', 'roi_name')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'SAR Backscatter (VV) vs Precipitation - Rain Splash Effect',
    hAxis: {title: 'Precipitation (mm)', minValue: 0},
    vAxis: {title: 'VV Backscatter (dB)', minValue: -25, maxValue: -5},
    pointSize: 4,
    colors: ['blue', 'red', 'orange', 'green'],
    legend: {position: 'right'}
  });

// Chart 3: Hooghly Only - Detailed view with trendline
var hooghlyWindChart = ui.Chart.feature.byFeature(hooghlyData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Hooghly River (Open Water): Wind Speed vs VV Backscatter',
    hAxis: {title: 'Wind Speed (m/s)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 5,
    pointColor: 'blue',
    trendlines: {0: {color: 'darkblue', lineWidth: 3, showR2: true, visibleInLegend: true}}
  });

// Chart 4: Urban Canal Only - Detailed view with trendline
var urbanWindChart = ui.Chart.feature.byFeature(tollyData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Tollys Nullah (Urban Canal): Wind Speed vs VV Backscatter',
    hAxis: {title: 'Wind Speed (m/s)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 5,
    pointColor: 'red',
    trendlines: {0: {color: 'darkred', lineWidth: 3, showR2: true, visibleInLegend: true}}
  });

// Chart 5: VH polarization comparison (cross-pol is more sensitive to roughness)
var windVsVH = ui.Chart.feature.groups(allData, 'wind_speed', 'VH', 'roi_name')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'SAR Backscatter (VH Cross-Pol) vs Wind Speed',
    hAxis: {title: 'Wind Speed at 10m (m/s)'},
    vAxis: {title: 'VH Backscatter (dB)'},
    pointSize: 4,
    colors: ['blue', 'red', 'orange', 'green'],
    legend: {position: 'right'}
  });

// =============================================================================
// 7. PRINT RESULTS
// =============================================================================

print('=== Wind-Backscatter Correlation Analysis ===');
print('Analysis Period:', startDate, 'to', endDate);
print('');

print('Sample counts per ROI:');
print('Hooghly River:', hooghlyData.size());
print('Tollys Nullah:', tollyData.size());
print('Kestopur Canal:', kestopurData.size());
print('EKW Wetlands:', ekwData.size());
print('');

print('--- MAIN RESULTS ---');
print(windVsVV);
print(precipVsVV);
print('');

print('--- DETAILED VIEWS ---');
print(hooghlyWindChart);
print(urbanWindChart);
print('');

print('--- CROSS-POLARIZATION (VH) ---');
print(windVsVH);

// =============================================================================
// 8. INTERPRETATION GUIDE
// =============================================================================

print('');
print('=== HOW TO INTERPRET ===');
print('');
print('HYPOTHESIS: Wind → Roughness → Higher Backscatter');
print('');
print('IF Hooghly shows POSITIVE slope + high R²:');
print('   → Wind proxy WORKS for open water');
print('');
print('IF Urban canals show FLAT/WEAK slope:');
print('   → Urban canyon shielding BREAKS the proxy');
print('   → Need coherence or other features for urban');
print('');
print('IF Precipitation shows effect independent of wind:');
print('   → Rain splash is a separate roughness mechanism');
print('   → Should include ERA5 precip as model input');

// =============================================================================
// 9. EXPORT DATA (Optional - for further analysis in Python)
// =============================================================================

// Uncomment to export results as CSV
/*
Export.table.toDrive({
  collection: allData,
  description: 'wind_backscatter_correlation_data',
  fileFormat: 'CSV'
});
*/
