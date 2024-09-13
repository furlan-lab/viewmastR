// #![allow(dead_code)]
use linfa_bayes::{MultinomialNbParams, MultinomialNb, NaiveBayesError};
use linfa::prelude::*;
// use anyhow::Result as OtherResult;
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

// use burn::{backend::Wgpu, tensor::Tensor};
// use std::collections::HashMap;

// #[derive(Debug)]
// pub struct MultinomialNB {
//     class_log_prior_: Vec<f64>,
//     feature_log_prob_: Vec<Vec<f64>>,
//     pub num_classes_: usize,
//     pub num_features_: usize,
//     pub num_samples_: usize,
// }

// impl MultinomialNB {
//     pub fn new() -> Self {
//         MultinomialNB {
//             class_log_prior_: vec![],
//             feature_log_prob_: vec![],
//             num_classes_: 0,
//             num_features_: 0,
//             num_samples_: 0,
//         }
//     }

//     pub fn fit(&mut self, x_train: &Tensor<Wgpu, 2>, y_train: &Vec<u64>) {
//         let shape = x_train.shape();
//         self.num_samples_ = shape.dims[0];
//         self.num_features_ = shape.dims[1];
//         let classes: Vec<u64> = y_train.iter().cloned().collect();
//         self.num_classes_ = classes.iter().cloned().collect::<Vec<_>>().len();

//         // Get the data from the tensor as a Vec<f64>
//         let x_train_data = x_train.to_data().convert::<f64>().value;

//         // Count class frequencies
//         let class_count = self.class_count(&y_train);

//         // Calculate class prior (log probabilities of each class)
//         self.class_log_prior_ = classes
//             .iter()
//             .map(|&class| (class_count[&class] as f64).ln() - (self.num_samples_ as f64).ln())
//             .collect();

//         // Calculate feature counts by class
//         let mut feature_count = vec![vec![0u64; self.num_features_]; classes.len()];

//         for (i, class) in y_train.iter().enumerate() {
//             for j in 0..self.num_features_ {
//                 feature_count[*class as usize][j] += x_train_data[i * self.num_features_ + j] as u64;
//             }
//         }

//         // Compute smoothed feature log probabilities
//         self.feature_log_prob_ = feature_count
//             .iter()
//             .map(|counts| {
//                 let total_count: u64 = counts.iter().sum();
//                 counts
//                     .iter()
//                     .map(|&count| ((count + 1) as f64).ln() - ((total_count + self.num_features_ as u64) as f64).ln())
//                     .collect()
//             })
//             .collect();
//     }

//     pub fn predict(&self, x_test: Tensor<Wgpu, 2>) -> Vec<u64> {
//         let mut predictions = Vec::new();
//         let shape = x_test.shape();
//         let n_features = shape.dims[1];
    
//         // Get the data from the tensor as a Vec<f64>
//         let x_test_data = x_test.to_data().convert::<f64>().value;
    
//         for i in 0..shape.dims[0] {
//             let mut log_probs: Vec<f64> = Vec::new();
    
//             for (class_index, class_log_prior) in self.class_log_prior_.iter().enumerate() {
//                 let mut log_prob = *class_log_prior;
//                 for j in 0..n_features {
//                     // Add the log-probability for the feature and its count
//                     log_prob += x_test_data[i * n_features + j] * self.feature_log_prob_[class_index][j];
//                 }
//                 log_probs.push(log_prob);
//             }
    
//             // Select the class with the highest log-probability
//             let predicted_class = log_probs
//                 .iter()
//                 .enumerate()
//                 .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
//                 .map(|(index, _)| index as u64)
//                 .unwrap();
    
//             predictions.push(predicted_class);
//         }
    
//         predictions
//     }
    
//     fn class_count(&self, y_train: &Vec<u64>) -> HashMap<u64, usize> {
//         let mut class_count = HashMap::new();
//         for &label in y_train.iter() {
//             *class_count.entry(label).or_insert(0) += 1;
//         }
//         class_count
//     }
// }

// // Testing
// pub mod tests {
//     use super::*;
//     use burn::tensor::{Shape, Data};
    
//     pub fn test() {
//         let x_train = Tensor::<Wgpu, 2>::from_data(Data::new(vec![
//             2.0, 1.0, 0.0,  // Class 0
//             3.0, 0.0, 1.0,  // Class 1
//             0.0, 2.0, 3.0,  // Class 0
//             4.0, 0.0, 1.0,  // Class 1
//         ], Shape::from([4, 3])));
    
//         let y_train = vec![0, 1, 0, 1];
    
//         let x_test = Tensor::<Wgpu, 2>::from_data(Data::new(vec![
//             1.0, 0.0, 1.0,  // Predicted as Class 0
//             3.0, 2.0, 1.0,  // Predicted as Class 1
//         ], Shape::from([2, 3])));
    
//         let mut nb = MultinomialNB::new();
//         nb.fit(&x_train, &y_train);
    
//         let predictions = nb.predict(x_test);
//         assert_eq!(predictions, vec![0, 1]);
//     }
// }
