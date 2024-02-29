#![allow(non_snake_case)]

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

// use core::num;
use std::path::Path;
use std::time::Instant;
use crate::common::{SCItemRaw, ModelRExport};
use crate::inference::infer_helper;


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

// /// Process Robj learning objects
// /// @export
// #[extendr]
// fn process_learning_obj(train: Robj, test: Robj, query: Robj, labels: Robj, hidden_layers: Robj)-> Robj {
//   let hidden_layers = hidden_layers.as_integer().unwrap() as usize;
//   let labelvec = labels.as_str_vector().unwrap();
//   let mut test_raw: Vec<SCItemRaw> =  vec![];
//   let mut train_raw: Vec<SCItemRaw> =  vec![];
//   let mut query_raw: Vec<SCItemRaw> =  vec![];

//   let sc_from_list = test.as_list().unwrap();
//   for (_item_str, item_robj) in sc_from_list{
//     let list_items = item_robj.as_list().unwrap();
//     let data = list_items[0].as_real_vector().unwrap();
//     // let datain: Vec<f64> = data.iter().map(|n| *n as f32).collect();
//     test_raw.push(SCItemRaw{
//       data: data,
//       target: (list_items[1].as_real().unwrap() as i32)
//     });
//   }
//   let sc_from_list = train.as_list().unwrap();
//   for (_item_str, item_robj) in sc_from_list{
//     let list_items = item_robj.as_list().unwrap();
//     let data = list_items[0].as_real_vector().unwrap();
//     // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
//     train_raw.push(SCItemRaw{
//       data: data,
//       target: (list_items[1].as_real().unwrap() as i32)
//     });
//   }
//   let sc_from_list = query.as_list().unwrap();
//   for (_item_str, item_robj) in sc_from_list{
//     let list_items = item_robj.as_list().unwrap();
//     let data = list_items[0].as_real_vector().unwrap();
//     // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
//     query_raw.push(SCItemRaw{
//       data: data,
//       target: 0
//     });
//   }
//   let predictions = scrna_conv::run_custom(train_raw, test_raw, query_raw, labelvec.len(), hidden_layers);
//   return r!(predictions)
// }


/// Process Robj learning objects for MLR
/// @export
/// @keywords internal
#[extendr]
fn process_learning_obj_mlr(train: Robj, test: Robj, query: Robj, labels: Robj, learning_rate: Robj, num_epochs: Robj, directory: Robj, verbose: Robj, backend: Robj, return_probs: Robj)-> List {
  let backend = match backend.as_str_vector(){
    Some(string_vec) => string_vec.first().unwrap().to_string(),
    _ => panic!("Cound not find backend: '{:?}'", backend)
  };
  // if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
  //   panic!("Cound not find backend: '{:?}'", backend)
  // }
  if ! ["wgpu"].contains(&backend.as_str()){
    panic!("Cound not find backend: '{:?}'", backend)
  }
  let return_probs: bool = return_probs.as_logical_vector().unwrap().first().unwrap().to_bool();
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
  let mut test_raw: Vec<SCItemRaw> =  vec![];
  let mut train_raw: Vec<SCItemRaw> =  vec![];
  let mut query_raw: Vec<SCItemRaw> =  vec![];

  let sc_from_list = test.as_list().unwrap();
  for (_item_str, item_robj) in sc_from_list{
    let list_items = item_robj.as_list().unwrap();
    let data = list_items[0].as_real_vector().unwrap();
    // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
    test_raw.push(SCItemRaw{
      data: data,
      target: (list_items[1].as_real().unwrap() as i32)
    });
  }
  let sc_from_list = train.as_list().unwrap();
  for (_item_str, item_robj) in sc_from_list{
    let list_items = item_robj.as_list().unwrap();
    let data = list_items[0].as_real_vector().unwrap();
    // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
    train_raw.push(SCItemRaw{
      data: data,
      target: (list_items[1].as_real().unwrap() as i32)
    });
  }
  let sc_from_list = query.as_list().unwrap();
  for (_item_str, item_robj) in sc_from_list{
    let list_items = item_robj.as_list().unwrap();
    let data = list_items[0].as_real_vector().unwrap();
    // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
    query_raw.push(SCItemRaw{
      data: data,
      target: 0
    });
  }
  let model_export= scrna_mlr::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose, return_probs);
  // if backend == "candle"{
  //   // model_export = scrna_mlr::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose, return_probs);

  // } 
  // else if backend == "wpgu"{
  //   model_export = scrna_mlr::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose, return_probs);
  // } 
  // else {
  //   // model_export = scrna_mlr::run_custom_nd(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose, return_probs);
  // }
  
  // model_export = scrna_mlr::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
  let params = list!(lr = model_export.lr, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
  let predictions = list!(model_export.predictions);
  let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
  let duration = start.elapsed();
  let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
  let probs: List = list!(model_export.probs.unwrap().iter().map(|x| r!(x)).collect::<Vec<Robj>>());
  return list!(params = params, predictions = predictions, history = history, duration = duration, probs = probs)
}


/// Process Robj learning objects for ANN
/// @export
/// @keywords internal
#[extendr]
fn process_learning_obj_ann(train: Robj, test: Robj, query: Robj, labels: Robj, hidden_size: Robj, learning_rate: Robj, num_epochs: Robj, directory: Robj, verbose: Robj, backend: Robj, return_probs: Robj)-> List {
  let backend = match backend.as_str_vector(){
    Some(string_vec) => string_vec.first().unwrap().to_string(),
    _ => panic!("Cound not find backend: '{:?}'", backend)
  };
  // if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
  //   panic!("Cound not find backend: '{:?}'", backend)
  // }
  if ! ["wgpu"].contains(&backend.as_str()){
    panic!("Cound not find backend: '{:?}'", backend)
  }
  let return_probs: bool = return_probs.as_logical_vector().unwrap().first().unwrap().to_bool();
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
  let mut test_raw: Vec<SCItemRaw> =  vec![];
  let mut train_raw: Vec<SCItemRaw> =  vec![];
  let mut query_raw: Vec<SCItemRaw> =  vec![];

  let sc_from_list = test.as_list().unwrap();
  for (_item_str, item_robj) in sc_from_list{
    let list_items = item_robj.as_list().unwrap();
    let data = list_items[0].as_real_vector().unwrap();
    // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
    test_raw.push(SCItemRaw{
      data: data,
      target: (list_items[1].as_real().unwrap() as i32)
    });
  }
  let sc_from_list = train.as_list().unwrap();
  for (_item_str, item_robj) in sc_from_list{
    let list_items = item_robj.as_list().unwrap();
    let data = list_items[0].as_real_vector().unwrap();
    // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
    train_raw.push(SCItemRaw{
      data: data,
      target: (list_items[1].as_real().unwrap() as i32)
    });
  }
  let sc_from_list = query.as_list().unwrap();
  for (_item_str, item_robj) in sc_from_list{
    let list_items = item_robj.as_list().unwrap();
    let data = list_items[0].as_real_vector().unwrap();
    // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
    query_raw.push(SCItemRaw{
      data: data,
      target: 0
    });
  }
  let model_export: ModelRExport;
  if hidden_size.len() == 1 {
    model_export = scrna_ann::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose, return_probs);
    // if backend == "candle"{
    //   model_export = scrna_ann::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    // } 
    // else if backend == "wpgu"{
    //   model_export = scrna_ann::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    // } 
    // else {
    //   model_export = scrna_ann::run_custom_nd(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
    // }
  } else {
    model_export = scrna_ann2l::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose, return_probs);
    // if backend == "candle"{
    //   model_export = scrna_ann2l::run_custom_candle(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    // } 
    // else if backend == "wpgu"{
    //   model_export = scrna_ann2l::run_custom_wgpu(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    // } 
    // else {
    //   model_export = scrna_ann2l::run_custom_nd(train_raw, test_raw, query_raw, labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
    // }
  }
  
  let params = list!(lr = model_export.lr, hidden_size = model_export.hidden_size, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
  let predictions = list!(model_export.predictions);
  let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
  let duration = start.elapsed();
  let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
  let probs: List;
  if return_probs {
    probs = list!(model_export.probs.unwrap().iter().map(|x| r!(x)).collect::<Vec<Robj>>());
  } else {
    probs = list!();
  }
  return list!(params = params, predictions = predictions, history = history, duration = duration, probs = probs)
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
  let mut query_raw: Vec<SCItemRaw> =  vec![];
  let sc_from_list = query.as_list().unwrap();
  if verbose {eprintln!("Loading data")};
  for (_item_str, item_robj) in sc_from_list{
    let list_items = item_robj.as_list().unwrap();
    let data = list_items[0].as_real_vector().unwrap();
    // let datain: Vec<f32> = data.iter().map(|n| *n as f32).collect();
    // eprint!("Pushing data {:?}\n", item_str);
    query_raw.push(SCItemRaw{
      data: data,
      target: 0
    });
  }
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
}
