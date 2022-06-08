//The ROI and SAMPLES variables, referenced in the code, were imported using the Assets property of the code editor
//ROI is the shapefile with the geometry of the region of interest
//SAMPLES is the shapefile with the training sample to be used by the model

//MUTABLE VARIABLES 
var date = ee.DateRange('2021-01-01', '2021-12-31');
//Index available: NDVI or NDBI ou NDWI.
var map_index = 'NDVI';
//Bands available: 'B1','B2','B3','B4','B5','B6','B7','B8'.
var bands = ['B1','B2','B3','B4','B5','B6','B8',map_index.toLowerCase()];
//Tree Numbers: From 30 to 1000.
var tree_numbers = 300;


//Constant variables
var SPLIT = 0.7;
var VIZPARAMS = {
   bands: ['B4', 'B3', 'B2'],
     min: 0,
     max: 0.3,
     gamma: [0.95, 1.1, 1]
 };

//-------------------------------- Start -------------------------------------

function colorPalette(index) {
  if (index == 'index_color') {
    return {min: 0, max: 1, palette: ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
                                '74A901', '66A000', '529400', '3E8601', '207401', '056201',
                                '004C00', '023B01', '012E01', '011D01', '011301']}
  }
  else if (index == 'classification_color') {
    return {min: 0, max: 6, palette: ['FFFFFF', '98FB98', 'FFAA88', '227B22', '007FFF', '000000', '808080']}
  }
}

var ls8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA");
var image_colletion = ee.ImageCollection(ls8.filterBounds(ROI).filterDate(date));
var composite = image_colletion.median();

var maskL8 = function(composite) {
  var qa = composite.select('BQA');
  var mask = qa.bitwiseAnd(1 << 4).eq(0);
  return composite.updateMask(mask);
};

var composite_mask = ee.ImageCollection(ls8.filterBounds(ROI)
    .filterDate(date))
    .map(maskL8)
    .median();

Map.addLayer(composite, VIZPARAMS, 'Mosaic with clouds');
Map.centerObject(ROI,12);
Map.addLayer(composite_mask, VIZPARAMS, 'Mosaic without clouds');
Map.centerObject(ROI,12);
Map.addLayer(SAMPLES, colorPalette('sample_color'), 'Training samples');
Map.centerObject(ROI,12);

switch(map_index.toUpperCase()) {
  case "NDVI":
    print('Creating layer NDVI - Normalized Difference Vegetation Index');
    var ndvi = composite_mask.normalizedDifference(['B5', 'B4']).rename('ndvi');
    composite_mask = composite_mask.addBands(ndvi,['ndvi']);
    Map.centerObject (ROI,12); 
    Map.addLayer(composite_mask.select('ndvi'), colorPalette('index_color'), 'NDVI layer');
    break;
  case "NDBI":
    print('Creating layer NDBI - Normalized Difference Built-up Index');
    var ndbi = composite_mask.normalizedDifference(['B6', 'B5']).rename('ndbi');
    composite_mask = composite_mask.addBands(ndbi,['ndbi']);
    Map.centerObject (ROI,12); 
    Map.addLayer(composite_mask.select('ndbi'), colorPalette('index_color'), 'NDBI layer');
    break;
  case "NDWI":
    print('Creating layer NDWI - Normalized Difference Water Index');
    var ndwi = composite_mask.normalizedDifference(['B5', 'B6']).rename('ndwi');
    composite_mask = composite_mask.addBands(ndwi,['ndwi']);
    Map.centerObject (ROI,12);
    Map.addLayer(composite_mask.select('ndwi'), colorPalette('index_color'), 'NDWI layer');
    break;
}

var random_samples = composite_mask.sampleRegions({
  collection: SAMPLES,
  properties: ['Classe_ID'],
  //region: ls8.filterBounds(ROI),
  scale: 30,
  tileScale: 16
  }).randomColumn('random');


var training_samples = random_samples.filter(ee.Filter.lt('random', SPLIT));
var test_samples = random_samples.filter(ee.Filter.gte('random', SPLIT));

var classification = ee.Classifier.smileRandomForest ({numberOfTrees: tree_numbers}).train({
   features: training_samples,
   classProperty: 'Classe_ID',
   inputProperties:bands
  });

var test_classification = test_samples.classify(classification);

//confmat - Construct confusion matrix 
var confmat = test_classification.errorMatrix('Classe_ID','classification');
print('Error Matrix - First model training (RF1)', confmat);
print('Accuracy - First model training (RF1)', confmat.accuracy());

var random_classifier = ee.Classifier.smileRandomForest ({numberOfTrees: tree_numbers}).train({
   features: random_samples,
   classProperty: 'Classe_ID',
   inputProperties:bands
  });

var random_class = composite_mask.classify(random_classifier);
var accuracy_test = random_samples.classify(random_classifier);
var random_confmat = accuracy_test.errorMatrix('Classe_ID','classification');

print('Error Matrix - Second model training (RF1)', random_confmat);
print('Accuracy - Second model training (RF1)', random_confmat.accuracy());

Map.addLayer(random_class, colorPalette('classification_color'), 'Image classification');

var geet = require('users/elacerda/geet:geet');
var majority = geet.majority(random_class, 1);
Map.addLayer(majority, colorPalette('classification_color'), 'Image classification filtered');

Export.image.toDrive({
  image: majority,
  description: 'ImagemRibeirao3_NDVI_2015',
  scale: 30,
  region: ROI,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});