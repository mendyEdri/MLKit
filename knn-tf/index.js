
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv');

log = console.log;

function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
    
    const result = features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
	    .pow(2)
	    .sum(1)
	    .pow(0.5)
        .expandDims(1)
	    .concat(labels, 1)
        .unstack()
	    .sort((a, b) => {
            const aValue = a.dataSync()[0];
            const bValue = b.dataSync()[0];
            return aValue > bValue ? 1 : -1
        })
        .slice(0, k)
	    .reduce((acc, pair) => acc + pair.dataSync()[1], 0) / k;

    return result;
}

let { features, labels, testFeatures, testLabels, trainSize } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'], // 'condition'
    labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const error = (testLabels[i][0] - result) / testLabels[i][0]; 
    log('Error', error * 100);
});