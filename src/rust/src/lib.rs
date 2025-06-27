#![allow(non_snake_case)]
// #![allow(dead_code)]
// #![allow(unused_imports)]
// #![allow(unused_variables)]


use extendr_api::prelude::*;
mod scrna_ann;
mod scrna_ann2l;
mod scrna_mlr;
mod scrna_conv;
mod utils;
mod pb;
mod common;
mod inference;
mod nb;

use std::path::Path;
use std::time::Instant;
use crate::common::*;
use crate::inference::*;

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
  if verbose {eprint!("Training model\n");}
  let (query_pred, _model) = nb::multinomial_nb(train_data, train_y, test_data, test_y, query).unwrap();

  if verbose {eprint!("Evaluating model\n");}
  // Measure and return the elapsed time
  let duration = start.elapsed();
  let duration_r: List = list!(total_duration = duration.as_secs_f64());
  let history: List = list!(train_acc = "ND", test_acc = "ND");
  let model = list!();
  return list!(model = model, predictions = list!(query_pred), history = history, duration = duration_r)
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

  let test_raw = extract_scitemraw(&test, None);   // No default target, extract from list
  let train_raw = extract_scitemraw(&train, None); // No default target, extract from list
  let query_raw = extract_scitemraw(&query, Some(0)); // Default target is 0 for query

  let model_export: RExport;
  if backend == "candle"{
    model_export = scrna_mlr::run_custom_candle(train_raw, test_raw, Some(query_raw), labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  } 
  else if backend == "wpgu"{
    model_export = scrna_mlr::run_custom_wgpu(train_raw, test_raw, Some(query_raw), labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  } 
  else {
    model_export = scrna_mlr::run_custom_nd(train_raw, test_raw, Some(query_raw), labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  }

  let params = list!(lr = model_export.lr, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
  let probs = list!(model_export.probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>());
  let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
  let duration = start.elapsed();
  let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
  return list!(params = params, probs = probs, history = history, duration = duration)
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

    
  let model_export: RExport;
  if hidden_size.len() == 1 {
    if backend == "candle"{
      model_export = scrna_ann::run_custom_candle(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else if backend == "wpgu"{
      model_export = scrna_ann::run_custom_wgpu(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else {
      model_export = scrna_ann::run_custom_nd(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    }
  } else {
    if backend == "candle"{
      model_export = scrna_ann2l::run_custom_candle(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else if backend == "wpgu"{
      model_export = scrna_ann2l::run_custom_wgpu(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    } 
    else {
      model_export = scrna_ann2l::run_custom_nd(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    }
  }
  
  let params = list!(lr = model_export.lr, hidden_size = model_export.hidden_size, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
  let probs = list!(model_export.probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>());
  let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
  let duration = start.elapsed();
  let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
  return list!(params = params, probs = probs, history = history, duration = duration)
}

/// Infer from a saved model (MLR, 1-hidden ANN, or 2-hidden ANN)
///
/// @param model_path  Character scalar – path to the `.mpk` checkpoint
/// @param query       A data-frame or matrix you can pass to `extract_scitemraw()`
/// @param num_classes Integer scalar – number of output classes
/// @param num_features Integer scalar – number of input features
/// @param model_type  Character scalar: `"mlr"`, `"ann1"`, or `"ann2"`
/// @param hidden1     (optional) Integer – size of the first hidden layer
/// @param hidden2     (optional) Integer – size of the second hidden layer (only for `"ann2"`)
/// @param verbose     Logical scalar – print progress to stderr?
///
/// @return A list with a single element `probs`, the flat numeric vector
///         of logits returned by the Rust model.
///
/// @export
#[extendr]          // @export
fn infer_from_model(
    model_path  : Robj,
    query       : Robj,
    num_classes : Robj,
    num_features: Robj,
    model_type  : Robj,
    hidden1     : Nullable<Robj>,
    hidden2     : Nullable<Robj>,
    verbose     : Robj,
    batch_size  : Robj,
    backend: Robj
) -> List {
    // ── verbosity ---------------------------------------------------------
    let verbose = verbose
        .as_logical_vector()
        .unwrap()
        .first()
        .unwrap()
        .to_bool();
    let backend = match backend.as_str_vector(){
      Some(string_vec) => string_vec.first().unwrap().to_string(),
      _ => panic!("Cound not find backend: '{:?}'", backend)
    };
    if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
      panic!("Cound not find backend: '{:?}'", backend)
    }
    // ── scalars -----------------------------------------------------------
    let model_path = model_path
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("`model_path` must be a string");
    if !Path::new(&model_path).exists() {
        panic!("Checkpoint not found: {model_path}");
    }

    let model_kind_str = model_type
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .unwrap_or_else(|| "mlr")
        .to_lowercase();

    let num_classes = num_classes
        .as_integer_vector()
        .and_then(|v| v.first().copied())
        .expect("`num_classes` must be an integer") as usize;

    let num_features = num_features
        .as_integer_vector()
        .and_then(|v| v.first().copied())
        .expect("`num_features` must be an integer") as usize;

    // ── optional hidden sizes --------------------------------------------
    let h1 = usize_from_nullable(hidden1);
    let h2 = usize_from_nullable(hidden2);

    let batch_size = batch_size
        .as_real()
        .map(|x| x as usize)
        .or_else(|| batch_size.as_integer().map(|x| x as usize))
        .unwrap_or_else(|| panic!("`batch_size` must be numeric"));

    // ── query to Vec<SCItemRaw> ------------------------------------------
    if verbose { eprintln!("Preparing query data"); }
    let query_raw = extract_scitemraw(&query, Some(0));

    // ── decide which NetKind to run --------------------------------------
    let net = match model_kind_str.as_str() {
        "mlr" => NetKind::Mlr,

        "ann1" | "ann" => {
            let size1 = h1.expect("`hidden1` must be supplied for ANN1 models");
            NetKind::Ann { hidden: size1 }
        }

        "ann2" => {
            let size1 = h1.expect("`hidden1` must be supplied for ANN2 models");
            let size2 = h2.expect("`hidden2` must be supplied for ANN2 models");
            NetKind::Ann2 { hidden1: size1, hidden2: size2 }
        }

        other => panic!("Unknown `model_type`: {other}  (use \"mlr\", \"ann1\", or \"ann2\")"),
    };

    // ── run the generic inference ----------------------------------------
    if verbose { eprintln!("Running inference as {:?}", net); }
    
    // Use the appropriate backend based on the `backend` parameter
    let probs = match backend.as_str() {
        "wgpu" => infer_wgpu(
            &model_path,
            net,
            num_classes,
            num_features,
            query_raw,
            Some(batch_size)),
        "candle" => infer_candle(
            &model_path,
            net,
            num_classes,
            num_features,
            query_raw,
            Some(batch_size)),
        "nd" => infer_nd(
            &model_path,
            net,
            num_classes,
            num_features,
            query_raw,
            Some(batch_size)),
        _ => panic!("Unknown backend: {}", backend),
    };

    // ── return to R -------------------------------------------------------
    if verbose { eprintln!("Returning results"); }
    list!(probs = probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>())
}


// ---------- util -------------------------------------------------------------
fn usize_from_nullable(n: Nullable<Robj>) -> Option<usize> {
    match n {
        Nullable::Null => None,

        Nullable::NotNull(robj) => {
            // Parse again as Nullable<Option<i32>>
            let parsed: Nullable<Option<i32>> = robj.try_into().ok()?;

            match parsed {
                Nullable::Null          => None,          // shouldn’t occur
                Nullable::NotNull(None) => None,          // NA
                Nullable::NotNull(Some(x)) => Some(x as usize),
            }
        }
    }
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
  mod viewmastR;
  fn readR;
  fn computeSparseRowVariances;
  fn process_learning_obj_ann;
  fn process_learning_obj_mlr;
  fn infer_from_model;
  fn process_learning_obj_nb;
}
