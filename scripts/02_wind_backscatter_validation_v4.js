/*
 * ERA5 Wind & Rain vs SAR Backscatter Validation
 * VERSION 4 - USER-DRAWN ROIs
 * 
 * INSTRUCTIONS:
 * 1. Run this script first to load the basemap
 * 2. Use the drawing tools (top-left) to draw TWO rectangles:
 *    - Name the first one: hooghly  (draw on the river ONLY)
 *    - Name the second one: canal   (draw on Tolly's Nullah ONLY)
 * 3. Run the script again
 * 
 * Author: Romit Basak
 * Date: 2025
 */

// =============================================================================
// 1. CHECK IF USER HAS DRAWN GEOMETRIES
// =============================================================================

// These will be defined when you draw rectangles and name them
var hooghly, canal;

try {
  // If geometries exist, use them
  print('✓ Found user-drawn geometries');
  print('Hooghly ROI:', hooghly);
  print('Canal ROI:', canal);
} catch(e) {
  print('=== STEP 1: DRAW YOUR ROIs ===');
  print('');
  print('1. Click the rectangle tool (top-left of map)');
  print('2. Draw a thin rectangle ON the Hooghly River water');
  print('3. In the popup, name it: hooghly');
  print('4. Draw another rectangle ON Tollys Nullah canal');
  print('5. Name it: canal');
  print('6. Click RUN again');
  print('');
  print('TIP: Make ROIs small and ONLY on water, not land');
  
  // Center map on Kolkata for drawing
  Map.setCenter(88.35, 22.55, 13);
  Map.setOptions('SATELLITE');
  
  // Add S1 layer to help identify water (dark = water)
  var s1_helper = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(ee.Geometry.Point(88.35, 22.55))
    .filterDate('2023-06-01', '2023-06-30')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .first();
  
  Map.addLayer(s1_helper.select('VV'), {min: -25, max: 0}, 'S1 VV (dark=water)', true);
  
  // Stop execution until geometries are drawn
  throw('Draw geometries first, then run again');
}

// =============================================================================
// 2. VISUALIZE THE DRAWN ROIs
// =============================================================================

Map.centerObject(hooghly, 14);
Map.setOptions('SATELLITE');
Map.addLayer(hooghly, {color: 'blue'}, 'Hooghly River (DRAWN)');
Map.addLayer(canal, {color: 'red'}, 'Urban Canal (DRAWN)');

// Show S1 to verify water coverage
var s1_sample = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(hooghly)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .first();

Map.addLayer(s1_sample.select('VV'), {min: -25, max: 0}, 'S1 VV (dark=water)', false);

// =============================================================================
// 3. ANALYSIS FUNCTION
// =============================================================================

var startDate = '2023-01-01';
var endDate = '2023-12-31';

function analyzeROI(roi, roiName) {
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select(['VV'])
    .limit(50);
    
  var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .select(['u_component_of_wind_10m', 'v_component_of_wind_10m', 'total_precipitation']);
  
  var filterTime = ee.Filter.maxDifference({
    difference: 3 * 60 * 60 * 1000,
    leftField: 'system:time_start',
    rightField: 'system:time_start'
  });
  
  var joined = ee.Join.saveBest({
    matchKey: 'era5_match',
    measureKey: 'time_diff'
  }).apply(s1, era5, filterTime);
  
  var data = joined.map(function(img) {
    var sarImg = ee.Image(img);
    var era5Img = ee.Image(img.get('era5_match'));
    
    var u = era5Img.select('u_component_of_wind_10m');
    var v = era5Img.select('v_component_of_wind_10m');
    var windSpeed = u.pow(2).add(v.pow(2)).sqrt();
    var precip = era5Img.select('total_precipitation').multiply(1000);
    
    var sarMean = sarImg.select('VV').reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 30,
      bestEffort: true
    }).get('VV');
    
    var windMean = windSpeed.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 10000,
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
// 4. RUN ANALYSIS
// =============================================================================

print('');
print('=== ANALYSIS RESULTS ===');

var hooghlyData = analyzeROI(hooghly, 'Hooghly');
var canalData = analyzeROI(canal, 'Urban Canal');

print('Hooghly samples:', hooghlyData.size());
print('Canal samples:', canalData.size());

// Hooghly Wind Chart
var hooghlyWindChart = ui.Chart.feature.byFeature(hooghlyData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'HOOGHLY RIVER (Open Water): Wind Speed vs VV Backscatter',
    hAxis: {title: 'Wind Speed (m/s)', minValue: 0},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'blue',
    trendlines: {0: {color: 'darkblue', lineWidth: 2, showR2: true}}
  });
print(hooghlyWindChart);

// Canal Wind Chart
var canalWindChart = ui.Chart.feature.byFeature(canalData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'URBAN CANAL: Wind Speed vs VV Backscatter',
    hAxis: {title: 'Wind Speed (m/s)', minValue: 0},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'red',
    trendlines: {0: {color: 'darkred', lineWidth: 2, showR2: true}}
  });
print(canalWindChart);

// Precipitation Charts
print('');
print('--- Precipitation Effect ---');

var hooghlyPrecipChart = ui.Chart.feature.byFeature(hooghlyData, 'precip_mm', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'HOOGHLY: Precipitation vs Backscatter',
    hAxis: {title: 'Precipitation (mm)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'blue'
  });
print(hooghlyPrecipChart);

var canalPrecipChart = ui.Chart.feature.byFeature(canalData, 'precip_mm', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'URBAN CANAL: Precipitation vs Backscatter',
    hAxis: {title: 'Precipitation (mm)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'red'
  });
print(canalPrecipChart);

// =============================================================================
// 5. INTERPRETATION GUIDE
// =============================================================================

print('');
print('=== INTERPRETATION ===');
print('');
print('Compare R² values between the two charts:');
print('');
print('IF Hooghly R² > 0.2 AND Canal R² < 0.1:');
print('   → Urban shielding confirmed');
print('   → Wind proxy works for open water, fails for urban');
print('');
print('IF both R² > 0.2:');
print('   → Wind proxy works everywhere (good news!)');
print('');
print('IF both R² < 0.1:');
print('   → Wind proxy doesnt work at C-band in this region');
