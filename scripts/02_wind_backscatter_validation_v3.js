/*
 * ERA5 Wind & Rain vs SAR Backscatter Validation
 * VERSION 3 - CORRECTED COORDINATES
 * 
 * Author: Romit Basak
 * Date: 2025
 */

// =============================================================================
// 1. CORRECTED ROIs - Actually on the water bodies!
// =============================================================================

// Hooghly River - Section near Howrah Bridge
// The river runs roughly 88.34 to 88.35 longitude here
// Shifted EAST to actually capture the river
var hooghly_river = ee.Geometry.Rectangle([88.345, 22.575, 88.355, 22.590]);

// Tolly's Nullah (Adi Ganga) - The canal runs along "Tolly Canal" label
// Looking at map: canal is around 88.335 longitude
// Create a thin rectangle along the canal
var urban_canal = ee.Geometry.Rectangle([88.333, 22.505, 88.340, 22.520]);

// Alternative: Let user draw ROIs interactively
// Uncomment this and draw polygons named 'hooghly_drawn' and 'canal_drawn'
// var hooghly_river = hooghly_drawn;
// var urban_canal = canal_drawn;

// =============================================================================
// 2. VISUALIZATION - VERIFY BEFORE RUNNING ANALYSIS
// =============================================================================

Map.centerObject(hooghly_river, 14);
Map.addLayer(hooghly_river, {color: 'blue'}, 'Hooghly River ROI');
Map.addLayer(urban_canal, {color: 'red'}, 'Urban Canal ROI');

// Add satellite basemap to verify water coverage
Map.setOptions('SATELLITE');

// =============================================================================
// 3. QUICK VISUAL CHECK - Show a sample S1 image
// =============================================================================

var s1_sample = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(hooghly_river)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .first();

// Water appears DARK in VV
Map.addLayer(s1_sample.select('VV'), {min: -25, max: 0}, 'S1 VV (dark=water)', true);

print('=== STEP 1: VERIFY ROI PLACEMENT ===');
print('Check the map - blue box should be ON the Hooghly River');
print('Red box should be ON Tollys Nullah canal');
print('');
print('If placement is wrong, adjust coordinates above and re-run');
print('');

// =============================================================================
// 4. ANALYSIS (same as before)
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
// 5. RUN ANALYSIS
// =============================================================================

print('=== STEP 2: ANALYSIS RESULTS ===');

var hooghlyData = analyzeROI(hooghly_river, 'Hooghly');
var urbanData = analyzeROI(urban_canal, 'Urban Canal');

print('Hooghly River samples:', hooghlyData.size());
print('Urban Canal samples:', urbanData.size());

// Hooghly chart
var hooghlyChart = ui.Chart.feature.byFeature(hooghlyData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'HOOGHLY RIVER (Open Water): Wind Speed vs VV Backscatter',
    hAxis: {title: 'Wind Speed (m/s)', minValue: 0},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'blue',
    trendlines: {0: {color: 'darkblue', lineWidth: 2, showR2: true}}
  });
print(hooghlyChart);

// Urban chart
var urbanChart = ui.Chart.feature.byFeature(urbanData, 'wind_speed', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'TOLLYS NULLAH (Urban Canal): Wind Speed vs VV Backscatter',
    hAxis: {title: 'Wind Speed (m/s)', minValue: 0},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'red',
    trendlines: {0: {color: 'darkred', lineWidth: 2, showR2: true}}
  });
print(urbanChart);

// Precip charts
print('');
print('--- Precipitation Effect ---');

var hooghlyPrecip = ui.Chart.feature.byFeature(hooghlyData, 'precip_mm', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'HOOGHLY: Precipitation vs Backscatter',
    hAxis: {title: 'Precipitation (mm)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'blue'
  });
print(hooghlyPrecip);

var urbanPrecip = ui.Chart.feature.byFeature(urbanData, 'precip_mm', 'VV')
  .setChartType('ScatterChart')
  .setOptions({
    title: 'URBAN CANAL: Precipitation vs Backscatter',
    hAxis: {title: 'Precipitation (mm)'},
    vAxis: {title: 'VV Backscatter (dB)'},
    pointSize: 6,
    pointColor: 'red'
  });
print(urbanPrecip);

// =============================================================================
// 6. INTERPRETATION
// =============================================================================

print('');
print('=== INTERPRETATION ===');
print('R² > 0.2 with positive slope = Wind proxy works');
print('R² < 0.1 or flat = Wind proxy fails');
print('');
print('Expected: Hooghly shows correlation, Urban does not (shielding)');
