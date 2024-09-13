#![allow(non_snake_case)]
// #![allow(dead_code)]
// #![allow(unused_imports)]
// #![allow(unused_variables)]

use extendr_api::prelude::*;
mod mnist_conv;
mod scrna_ann;
mod scrna_ann2l;
mod scrna_mlr;
mod scrna_conv;
mod utils;
mod mnist_ann;
mod pb;
mod common;
mod inference;
mod nb;

// use core::num;
use std::path::Path;
use std::time::Instant;
// use crate::common::{ModelRExport, extract_vectors, extract_scalars, create_tensor, extract_scitemraw};
use crate::common::{ModelRExport, extract_vectors, extract_scalars, extract_scitemraw};
use crate::inference::infer_helper;
// use linfa::prelude::Predict;
// use nb::MultinomialNB;
// use std::io;
// use std::io::prelude::*;


// fn pause() {
//     let mut stdin = io::stdin();
//     let mut stdout = io::stdout();

//     // We want the cursor to stay at the end of the line, so we print without a newline and flush manually.
//     write!(stdout, "Press any key to continue...").unwrap();
//     stdout.flush().unwrap();

//     // Read a single byte and discard
//     let _ = stdin.read(&mut [0u8]).unwrap();
// }


/// Run test nb training
/// @export
/// @keywords internal
// #[extendr]
// fn run_nb_test(){
//   let result = nb::tests::test();
//   eprint!("{:?}", result);
// }

/// Process Robj learning objects for MLR
/// @export
/// @keywords internal
#[extendr]
fn process_learning_obj_nb(train: Robj, test: Robj, query: Robj) -> List {
  let start = Instant::now();
  let verbose =true;
  // Extracting data
  if verbose {eprint!("Extracting data\n");}
  let test_data = extract_vectors(&test, 0);
  let test_y = extract_scalars(&test, 1); // Actual test labels
  let train_data = extract_vectors(&train, 0);
  let train_y = extract_scalars(&train, 1);
  let query = extract_vectors(&query, 0);

  // let mut train_arr = Array2::<bool>::default((train_sampleno, train_featureno));
  // for (i, mut row) in train_arr.axis_iter_mut(Axis(0)).enumerate() {
  //     for (j, col) in row.iter_mut().enumerate() {
  //         *col = train_data[i][j];
  //     }
  // }

  // Convert data to tensors
  // if verbose {eprint!("Converting to tensors\n");}
  // let test_data = create_tensor(test_data);
  // let train_data = create_tensor(train_data);
  // let query = create_tensor(query);


  // Initialize and train the Naive Bayes model
  // let mut nb = MultinomialNB::new();
  if verbose {eprint!("Training model\n");}
  let (query_pred, _model) = nb::multinomial_nb(train_data, train_y, test_data, test_y, query).unwrap();
  // let query_predictions = model.predict(query);
  // nb.fit(&train_data, &train_y);


  if verbose {eprint!("Evaluating model\n");}
  // fn compare_predictions(pred: Vec<u64>, actual: Vec<u64>, data_type: String) -> f64 {
  //   // Compare predictions to actual test_y values
  //   let correct: usize = pred.iter()
  //     .zip(actual.iter()) // Zip predictions with actual labels
  //     .filter(|(pred, actual)| **pred == **actual as u64) // Compare prediction with actual
  //     .count();
  //   // Print the results
  //   let accuracy = correct as f64 / actual.len() as f64 * 100.0;
  //   println!("Accuracy on {} data: {:.3}%", data_type, accuracy);
  //   accuracy
  // }

  // let train_acc = compare_predictions(nb.predict(train_data), train_y, String::from("training"));
  // let test_acc = compare_predictions(nb.predict(test_data), test_y, String::from("validation"));

  // Measure and return the elapsed time
  let duration = start.elapsed();
  let duration_r: List = list!(total_duration = duration.as_secs_f64());
  // let query_predictions_r: Vec<Robj> = nb.predict(query).iter().map(|x| r!(x)).collect();
  
  // let params = list!(num_classes = nb.num_classes_, num_features = nb.num_features_);

  let history: List = list!(train_acc = "ND", test_acc = "ND");
  // let history: List = list!(train_acc = train_acc, test_acc = test_acc);
  // Return the list of predictions, duration, and accuracy
  // pause();
  let params = list!();
  return list!(params = params, predictions = list!(query_pred), history = history, duration = duration_r)
  // list!(duration = duration_r, acc_r = acc_r, query_predictions = query_predictions_r)
}

  
/// Run full mnist training in R
/// @export
/// @keywords internal

#[extendr]
fn run_mnist_terminal(){
  let _result = mnist_conv::run_burn();
}


/// Run full mnist training in R; for Rstudio
/// @export
/// @keywords internal
#[extendr]
fn run_mnist(){
  let _result = mnist_conv::run_custom();
}

/// Run full mnist training in R; for Rstudio
/// @export
/// @keywords internal
#[extendr]
fn run_mnist_ann(){
  let _result = mnist_ann::run_mnist_mlr_custom();
}

/// test data
/// @export
/// @keywords internal
#[extendr]
fn test_dataset(){
  let _result = mnist_ann::test_dataset();
}

/// Read an R object
/// @export
/// @keywords internal
#[extendr]
fn readR(obj: Robj){
  eprint!("{:?}", obj);
}

/// Compute sparse row variance
/// @export
/// @keywords internal
#[extendr]
fn computeSparseRowVariances(j: Robj, val: Robj, rm: Robj, n: Robj)-> Vec<f64>{
  let j_usize = j.as_real_vector().unwrap().iter().map(|&val| val as usize).collect();
  utils::sparse_row_variances(j_usize, val.as_real_vector().unwrap(), rm.as_real_vector().unwrap(), n.as_integer().unwrap() as usize)
}


/// Process Robj learning objects for MLR
/// @export
/// @keywords internal
#[extendr]
fn process_learning_obj_mlr(train: Robj, test: Robj, query: Robj, labels: Robj, learning_rate: Robj, num_epochs: Robj, directory: Robj, verbose: Robj, backend: Robj)-> List {
  let backend = match backend.as_str_vector(){
    Some(string_vec) => string_vec.first().unwrap().to_string(),
    _ => panic!("Cound not find backend: '{:?}'", backend)
  };
  if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
    panic!("Cound not find backend: '{:?}'", backend)
  }
  let start = Instant::now();
  let verbose: bool = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
  let learning_rate = *learning_rate.as_real_vector().unwrap().first().unwrap_or(&0.2) as f64;
  let num_epochs = *num_epochs.as_real_vector().unwrap().first().unwrap_or(&10.0) as usize;
  let artifact_dir = match directory.as_str_vector() {
    Some(string_vec) => string_vec.first().unwrap().to_string(),
    _ => panic!("Cound not find folder: '{:?}'", directory)
  };
  if !Path::new(&artifact_dir).exists(){
    panic!("Could not find folder: '{:?}'", artifact_dir)
  }
  let labelvec = labels.as_str_vector().unwrap();

  // Refactored code
  if verbose {eprint!("Loading train\n");}
  let test_raw = extract_scitemraw(&test, None);   // No default target, extract from list
  if verbose {eprint!("Loading test\n");}
  let train_raw = extract_scitemraw(&train, None); // No default target, extract from list
  if verbose {eprint!("Loading query\n");}
  let query_raw = extract_scitemraw(&query, Some(0)); // Default target is 0 for query

  let model_export: ModelRExport;
  if backend == "candle"{
    model_export = scrna_mlr::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  } 
  else if backend == "wpgu"{
    model_export = scrna_mlr::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  } 
  else {
    model_export = scrna_mlr::run_custom_nd(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  }

  // model_export = scrna_mlr::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  let params = list!(lr = model_export.lr, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
  let predictions = list!(model_export.predictions);
  let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
  let duration = start.elapsed();
  let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
  return list!(params = params, predictions = predictions, history = history, duration = duration)
}


/// Process Robj learning objects for ANN
/// @export
/// @keywords internal
#[extendr]
fn process_learning_obj_ann(train: Robj, test: Robj, query: Robj, labels: Robj, hidden_size: Robj, learning_rate: Robj, num_epochs: Robj, directory: Robj, verbose: Robj, backend: Robj)-> List {
  let backend = match backend.as_str_vector(){
    Some(string_vec) => string_vec.first().unwrap().to_string(),
    _ => panic!("Cound not find backend: '{:?}'", backend)
  };
  if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
    panic!("Cound not find backend: '{:?}'", backend)
  }
  let start = Instant::now();
  let verbose: bool = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
  let learning_rate = *learning_rate.as_real_vector().unwrap().first().unwrap_or(&0.2);
  let num_epochs = *num_epochs.as_real_vector().unwrap().first().unwrap_or(&10.0) as usize;
  let hidden_size = hidden_size.as_real_vector().unwrap();
  let hidden_size1 = *hidden_size.first().unwrap() as usize;
  let mut hidden_size2 = 0 as usize; 
  if hidden_size.len() == 2{
    hidden_size2 = hidden_size[1] as usize;
  }
  let artifact_dir = match directory.as_str_vector() {
    Some(string_vec) => string_vec.first().unwrap().to_string(),
    _ => panic!("Cound not find folder: '{:?}'", directory)
  };
  if !Path::new(&artifact_dir).exists(){
    panic!("Could not find folder: '{:?}'", artifact_dir)
  }
  let labelvec = labels.as_str_vector().unwrap();

  // Refactored code
  let test_raw = extract_scitemraw(&test, None);   // No default target, extract from list
  let train_raw = extract_scitemraw(&train, None); // No default target, extract from list
  let query_raw = extract_scitemraw(&query, Some(0)); // Default target is 0 for query

    
  let model_export: ModelRExport;
  if hidden_size.len() == 1 {
    if backend == "candle"{
      model_export = scrna_ann::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else if backend == "wpgu"{
      model_export = scrna_ann::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else {
      model_export = scrna_ann::run_custom_nd(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    }
  } else {
    if backend == "candle"{
      model_export = scrna_ann2l::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else if backend == "wpgu"{
      model_export = scrna_ann2l::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else {
      model_export = scrna_ann2l::run_custom_nd(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    }
  }
  
  let params = list!(lr = model_export.lr, hidden_size = model_export.hidden_size, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
  let predictions = list!(model_export.predictions);
  let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
  let duration = start.elapsed();
  let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
  return list!(params = params, predictions = predictions, history = history, duration = duration)
}



/// Process Robj learning objects for ANN
/// @export
/// @keywords internal
#[extendr]
fn test_backend(){
  // crate::scrna_mlr::tch_gpu::run();
}

/// infer from saved model
/// @export
/// @keywords internal
#[extendr]
fn infer_from_model(model_path: Robj, query: Robj, num_classes: Robj, num_features: Robj, verbose: Robj) -> List{
  let verbose =  verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
  if verbose {eprintln!("Loading model")};
  let model_path_tested = match model_path.as_str_vector() {
    Some(string_vec) => string_vec.first().unwrap().to_string(),
    _ => panic!("Cound not parse folder: '{:?}'", model_path)
  };
  if !Path::new(&model_path_tested).exists(){
    panic!("Could not find folder: '{:?}'", model_path)
  }
  if verbose {eprintln!("Loading data")};
  let query_raw = extract_scitemraw(&query, Some(0)); // Default target is 0 for query
  let num_classes = num_classes.as_integer().unwrap() as usize;
  let num_features = num_features.as_integer().unwrap() as usize;
  if verbose {eprintln!("Running inference")};
  let (predictions, probs) = infer_helper(model_path_tested, num_classes, num_features, query_raw);
  if verbose {eprintln!("Returning results")};
  return list!(predictions = predictions, probs = probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>())

  
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
  mod viewmastR;
  fn run_mnist_terminal;
  fn run_mnist;
  fn run_mnist_ann;
  fn readR;
  fn computeSparseRowVariances;
  fn process_learning_obj_ann;
  fn process_learning_obj_mlr;
  fn test_backend;
  fn infer_from_model;
  // fn run_nb_test;
  fn process_learning_obj_nb;
}
