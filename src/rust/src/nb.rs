// Module for Naive Bayes classifier
use linfa_bayes::{MultinomialNbParams, MultinomialNb, NaiveBayesError};
use linfa::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};
use ndarray::Axis;
use ndarray::Dim;

pub fn convert2dvec_array(x: Vec<Vec<f64>>) -> Array<f64, Dim<[usize; 2]>> {
    let train_sampleno = x.len();
    let train_featureno = x[0].len();

    let mut x_arr = Array::zeros((train_sampleno, train_featureno));
    for (i, mut row) in x_arr.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col =x[i][j];
        }
    }
    x_arr
}


fn compare_predictions(pred: &ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>, actual: &ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>>, data_type: &str) -> f64 {
    // Compare predictions to actual test_y values
    let correct: usize = pred.iter()
        .zip(actual.iter()) // Zip predictions with actual labels
        .filter(|(pred, actual)| *pred == *actual) // Compare prediction with actual
        .count();
        
    // Calculate accuracy
    let accuracy = correct as f64 / actual.len() as f64 * 100.0;
    
    // Print the results
    println!("Accuracy on {} data: {:.3}%", data_type, accuracy);
    
    accuracy
}


  pub type MyResult<T> = std::result::Result<(Vec<i32>, T), NaiveBayesError>;


  pub fn multinomial_nb(x: Vec<Vec<f64>>, y: Vec<usize>, test_x: Vec<Vec<f64>>, test_y: Vec<usize>, query: Vec<Vec<f64>>) -> MyResult<MultinomialNb<f64, usize>> {
    // Convert input data to ndarray Arrays
    let train_array = convert2dvec_array(x);
    let train_y: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = Array::from_vec(y);
    let test_array = convert2dvec_array(test_x);
    let test_y: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = Array::from_vec(test_y);
    let query_array = convert2dvec_array(query);
    let query_y: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = Array::zeros(query_array.shape()[0] as usize);
    // Create DatasetView for training and testing
    let ds = DatasetView::new(train_array.view(), train_y.view());
    let test_ds = DatasetView::new(test_array.view(), test_y.view());
    let query_ds = DatasetView::new(query_array.view(), query_y.view());
    // Create a new MultinomialNB model with smoothing parameter `alpha = 1.0`
    let params = MultinomialNbParams::new().alpha(1.0);
    
    // Fit the model using training data
    let model = params.fit(&ds)?;
    
    // Perform predictions on training and test datasets
    let train_preds = model.predict(&ds);
    let test_preds: ArrayBase<OwnedRepr<usize>, Dim<[usize; 1]>> = model.predict(&test_ds);
    let query_preds = model.predict(&query_ds);
    let query_vec: Vec<i32> = query_preds.iter().map(|&x| x as i32).collect();
    // Compare predictions to actual labels
    compare_predictions(&train_preds, &train_y, "training");
    compare_predictions(&test_preds, &test_y, "validation");
    
    // Return the fitted model
    Ok((query_vec, model))
}
