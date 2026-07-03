/*
 * ERA5 Wind & Rain vs SAR Backscatter Validation
 * MEMORY-OPTIMIZED VERSION
 * 
 * Changes from v1:
 * - Reduced time range (1 year instead of 7)
 * - Smaller ROIs
 * - Limited collection size
 * - Sequential processing instead of merge
 * 
 * Author: Romit Basak
 * Date: 2025
 */

// =============================================================================
// 1. DEFINE SMALLER, PRECISE ROIs
// =============================================================================

// Hooghly River - Small section of open water
var hooghly_river = ee.Geometry.Rectangle([88.338, 22.580, 88.343, 22.590]);

// Tolly's Nullah - Narrow urban canal section
var urban_canal = ee.Geometry.Rectangle([88.342, 22.498, 88.348, 22.503]);

// =============================================================================
// 2. ANALYSIS PARAMETERS
// =============================================================================

// Use just 2 years of monsoon seasons (June-October)
// This captures high variability in wind/rain while limiting data volume
var dateRanges = [
  {start: '2022-06-01', end: '2022-10-31', label: '2022 Monsoon'},
  {start: '2023-06-01', end: '2023-10-31', label: '2023 Monsoon'}
];

// For initial test, use just one year
var startDate = '2023-01-01';
var endDate = '2023-12-31';

// =============================================================================
// 3. LOAD DATA WITH LIMITS
// =============================================================================

function loadS1Limited(roi, startDate, endDate, maxImages) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV'])
    .limit(maxImages);
}

function loadERA5Limited(roi, startDate, endDate) {
  return ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .select(['u_component_of_wind_10m', 'v_component_of_wind_10m', 'total_precipitation']);
}

// =============================================================================
// 4. SIMPLIFIED ANALYSIS FUNCTION
// =============================================================================

function analyzeROI(roi, roiName, startDate, endDate) {
  // Limit to 50 images max
  var s1 = loadS1Limited(roi, startDate, endDate, 50);
  var era5 = loadERA5Limited(roi, startDate, endDate);
  
  // Temporal join
  var filterTime = ee.Filter.maxDifference({
    difference: 3 * 60 * 60 * 1000, // 3 hour window
    leftField: 'system:time_start',
    rightField: 'system:time_start'
  });
  
  var saveBestJoin = ee.Join.saveBest({
    matchKey: 'era5_match',
    measureKey: 'time_diff'
  });
  
  var joined = saveBestJoin.apply(s1, era5, filterTime);
  
  // Extract statistics
  var data = joined.map(function(img) {
    var sarImg = ee.Image(img);
    var era5Img = ee.Image(img.get('era5_match'));
    
    // Wind speed
    var u = era5Img.select('u_component_of_wind_10m');
    var v = era5Img.select('v_component_of_wind_10m');
    var windSpeed = u.pow(2).add(v.pow(2)).sqrt();
    
    // Precipitation
    var precip = era5Img.select('total_precipitation').multiply(1000);
    
    // Get mean values - use larger scale to reduce computation
    var sarMean = sarImg.select('VV').reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 30,
      bestEffort: true
    }).get('VV');
    
    var windMean = windSpeed.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 10000, // ERA5 is coarse anyway
      bestEffort: true
    }).get('u_component_of_wind_10m');
    
    var precipMean = precip.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 10000,
      bestEffort: true
    }).get('total_precipitation');
    
    return ee.Feature(null, {
      'VV': sarMean,
      'wind_speed': windMean,
      'precip_mm': precipMean,
      'roi': roiName
    });
  });
  
  return data.filter(ee.Filter.notNull(['VV', 'wind_speed']));
}

// =============================================================================
// 5. RUN ANALYSIS SEPARATELY FOR EACH ROI
// =============================================================================

print('=== Wind-Backscatter Validation (Memory Optimized) ===');
print('Period:', startDate, 'to', endDate);
print('');

// Analyze Hooghly River
var hooghlyData = analyzeROI(hooghly_river, 'Hooghly', startDate, endDate);

print('Hooghly River sample count:');
print(hooghlyData.size());

var hooghlyChart = ui.Chart.feature.byFeature(hooghlyData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Hooghly River (Open Water): Wind vs Backscatter',
    hAxis: {title: 'Wind Speed (m/s)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 5,
    pointColor: 'blue',
    trendlines: {0: {color: 'darkblue', showR2: true}}
  });

print(hooghlyChart);

// Analyze Urban Canal
var urbanData = analyzeROI(urban_canal, 'Urban Canal', startDate, endDate);

print('');
print('Urban Canal sample count:');
print(urbanData.size());

var urbanChart = ui.Chart.feature.byFeature(urbanData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Tollys Nullah (Urban Canal): Wind vs Backscatter',
    hAxis: {title: 'Wind Speed (m/s)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 5,
    pointColor: 'red',
    trendlines: {0: {color: 'darkred', showR2: true}}
  });

print(urbanChart);

// =============================================================================
// 6. PRECIPITATION CHARTS
// =============================================================================

print('');
print('--- Precipitation Effect ---');

var hooghlyPrecipChart = ui.Chart.feature.byFeature(hooghlyData, 'precip_mm', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Hooghly: Precipitation vs Backscatter',
    hAxis: {title: 'Precipitation (mm)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 5,
    pointColor: 'blue'
  });

print(hooghlyPrecipChart);

var urbanPrecipChart = ui.Chart.feature.byFeature(urbanData, 'precip_mm', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'Urban Canal: Precipitation vs Backscatter',
    hAxis: {title: 'Precipitation (mm)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 5,
    pointColor: 'red'
  });

print(urbanPrecipChart);

// =============================================================================
// 7. SHOW ROIS ON MAP
// =============================================================================

Map.centerObject(hooghly_river, 13);
Map.addLayer(hooghly_river, {color: 'blue'}, 'Hooghly River');
Map.addLayer(urban_canal, {color: 'red'}, 'Urban Canal');

// Add a S1 image for context
var s1_sample = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(hooghly_river)
  .filterDate('2023-06-01', '2023-06-30')
  .first();

Map.addLayer(s1_sample.select('VV'), {min: -25, max: 0}, 'S1 VV Sample', false);

// =============================================================================
// 8. INTERPRETATION
// =============================================================================

print('');
print('=== INTERPRETATION GUIDE ===');
print('Look at the R² values in the trendlines:');
print('');
print('Hooghly R² > 0.2 = Wind proxy works for open water');
print('Urban R² < 0.1 = Urban shielding breaks the proxy');
print('');
print('If both are low: Wind proxy may not work at C-band');
