#![allow(non_snake_case)]
// #![allow(dead_code)]
// #![allow(unused_imports)]
// #![allow(unused_variables)]


use extendr_api::prelude::*;
// use num_traits::cast;
mod scrna_ann;
mod scrna_ann2l;
mod scrna_mlr;
mod scrna_conv;
mod utils;
mod pb;
mod common;
mod inference;
mod nb;
// mod train;

use std::path::Path;
use std::time::Instant;
use crate::common::*;
use crate::inference::{infer_helper_mlr, infer_helper_ann, infer_helper_ann2l};

// use std::convert::TryFrom;
// use std::convert::TryInto;
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
  // let my_model = MyMultinomialNb(model);
  // let model_export: Vec<f64> = my_model.try_into().unwrap();
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


// /// Process Robj learning objects for MLR
// /// @export
// /// @keywords internal
// #[extendr]
// fn process_learning_obj_mlr(train: Robj, test: Robj, query: Robj, labels: Robj, learning_rate: Robj, num_epochs: Robj, directory: Robj, verbose: Robj, backend: Robj)-> List {
//   let backend = match backend.as_str_vector(){
//     Some(string_vec) => string_vec.first().unwrap().to_string(),
//     _ => panic!("Cound not find backend: '{:?}'", backend)
//   };
//   if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
//     panic!("Cound not find backend: '{:?}'", backend)
//   }
//   let start = Instant::now();
//   let verbose: bool = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
//   let learning_rate = *learning_rate.as_real_vector().unwrap().first().unwrap_or(&0.2) as f64;
//   let num_epochs = *num_epochs.as_real_vector().unwrap().first().unwrap_or(&10.0) as usize;
//   let artifact_dir = match directory.as_str_vector() {
//     Some(string_vec) => string_vec.first().unwrap().to_string(),
//     _ => panic!("Cound not find folder: '{:?}'", directory)
//   };
//   if !Path::new(&artifact_dir).exists(){
//     panic!("Could not find folder: '{:?}'", artifact_dir)
//   }
//   let labelvec = labels.as_str_vector().unwrap();

//   let test_raw = extract_scitemraw(&test, None);   // No default target, extract from list
//   let train_raw = extract_scitemraw(&train, None); // No default target, extract from list
//   let query_raw = extract_scitemraw(&query, Some(0)); // Default target is 0 for query

//   let model_export: RExport;
//   if backend == "candle"{
//     model_export = scrna_mlr::run_custom_candle(train_raw, test_raw, Some(query_raw), labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
//   } 
//   else if backend == "wpgu"{
//     model_export = scrna_mlr::run_custom_wgpu(train_raw, test_raw, Some(query_raw), labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
//   } 
//   else {
//     model_export = scrna_mlr::run_custom_nd(train_raw, test_raw, Some(query_raw), labelvec.len(), learning_rate, num_epochs, Some(artifact_dir), verbose);
//   }

//   let params = list!(lr = model_export.lr, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
//   // let predictions = list!(model_export.predictions);
//   let probs = list!(model_export.probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>());
//   let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
//   let duration = start.elapsed();
//   let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
//   return list!(params = params, probs = probs, history = history, duration = duration)
// }


// /// Process Robj learning objects for ANN
// /// @export
// /// @keywords internal
// #[extendr]
// fn process_learning_obj_ann(train: Robj, test: Robj, query: Robj, labels: Robj, hidden_size: Robj, learning_rate: Robj, num_epochs: Robj, directory: Robj, verbose: Robj, backend: Robj)-> List {
//   let backend = match backend.as_str_vector(){
//     Some(string_vec) => string_vec.first().unwrap().to_string(),
//     _ => panic!("Cound not find backend: '{:?}'", backend)
//   };
//   if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
//     panic!("Cound not find backend: '{:?}'", backend)
//   }
//   let start = Instant::now();
//   let verbose: bool = verbose.as_logical_vector().unwrap().first().unwrap().to_bool();
//   let learning_rate = *learning_rate.as_real_vector().unwrap().first().unwrap_or(&0.2);
//   let num_epochs = *num_epochs.as_real_vector().unwrap().first().unwrap_or(&10.0) as usize;
//   let hidden_size = hidden_size.as_real_vector().unwrap();
//   let hidden_size1 = *hidden_size.first().unwrap() as usize;
//   let mut hidden_size2 = 0 as usize; 
//   if hidden_size.len() == 2{
//     hidden_size2 = hidden_size[1] as usize;
//   }
//   let artifact_dir = match directory.as_str_vector() {
//     Some(string_vec) => string_vec.first().unwrap().to_string(),
//     _ => panic!("Cound not find folder: '{:?}'", directory)
//   };
//   if !Path::new(&artifact_dir).exists(){
//     panic!("Could not find folder: '{:?}'", artifact_dir)
//   }
//   let labelvec = labels.as_str_vector().unwrap();

//   // Refactored code
//   let test_raw = extract_scitemraw(&test, None);   // No default target, extract from list
//   let train_raw = extract_scitemraw(&train, None); // No default target, extract from list
//   let query_raw = extract_scitemraw(&query, Some(0)); // Default target is 0 for query

    
//   let model_export: RExport;
//   if hidden_size.len() == 1 {
//     if backend == "candle"{
//       model_export = scrna_ann::run_custom_candle(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
//     } 
//     else if backend == "wpgu"{
//       model_export = scrna_ann::run_custom_wgpu(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
//     } 
//     else {
//       model_export = scrna_ann::run_custom_nd(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, learning_rate, num_epochs, Some(artifact_dir), verbose);
//     }
//   } else {
//     if backend == "candle"{
//       model_export = scrna_ann2l::run_custom_candle(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
//     } 
//     else if backend == "wpgu"{
//       model_export = scrna_ann2l::run_custom_wgpu(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
//     } 
//     else {
//       model_export = scrna_ann2l::run_custom_nd(train_raw, test_raw, Some(query_raw), labelvec.len(), hidden_size1, hidden_size2, learning_rate, num_epochs, Some(artifact_dir), verbose);
//     }
//   }
  
//   let params = list!(lr = model_export.lr, hidden_size = model_export.hidden_size, epochs = model_export.num_epochs, batch_size = model_export.batch_size, workers = model_export.num_workers, seed = model_export.seed);
//   let probs = list!(model_export.probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>());
//   let history: List = list!(train_acc = model_export.train_history.acc, test_acc = model_export.test_history.acc, train_loss = model_export.train_history.loss, test_loss = model_export.test_history.loss);
//   let duration = start.elapsed();
//   let duration: List = list!(total_duration = duration.as_secs_f64(), training_duration = model_export.training_duration);
//   return list!(params = params, probs = probs, history = history, duration = duration)
// }


/// Extract backend string ("wgpu" | "candle" | "nd").
fn parse_backend(backend: &Robj) -> String {
    let be = backend
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("Could not find backend");
    if !["wgpu", "candle", "nd"].contains(&be.as_ref()) {
        panic!("Unknown backend: {be}");
    }
    be.to_string()
}

/// Pull out the directory and make sure it exists.
fn parse_directory(directory: &Robj) -> String {
    let dir = directory
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("Could not find folder");
    if !Path::new(&dir).exists() {
        panic!("Could not find folder: {dir}");
    }
    dir.to_string()
}

/// A *single* entry-point that covers MLR and ANN/ANN-2L.
///
/// * `model_type` – `"mlr"`, `"ann"`, or `"ann2"` (you can choose any tokens you like)
/// * `hidden_size` – `NULL` for MLR; numeric (len 1 or 2) for ANN.
/// @export
/// @keywords internal
#[extendr]
fn process_learning_obj(
    model_type: Robj,
    train: Robj,
    test: Robj,
    query: Robj,
    labels: Robj,
    feature_names: Robj,
    hidden_size: Nullable<Robj>,      // <- optional
    learning_rate: Robj,
    num_epochs: Robj,
    directory: Robj,
    verbose: Robj,
    backend: Robj,
) -> List {
    // -------------------------------------------------------------------
    // Common parsing ----------------------------------------------------
    // -------------------------------------------------------------------
    let model = model_type
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("Must supply model_type (\"mlr\" | \"ann\")");
    let be = parse_backend(&backend);
    let artifact_dir = parse_directory(&directory);

    let verbose: bool = verbose
        .as_logical_vector()
        .unwrap()
        .first()
        .unwrap()
        .to_bool();
    let lr = *learning_rate
        .as_real_vector()
        .unwrap()
        .first()
        .unwrap_or(&0.2) as f64;
    let epochs = *num_epochs
        .as_real_vector()
        .unwrap()
        .first()
        .unwrap_or(&10.0) as usize;
    let hidden_size: Option<Robj> = Some(hidden_size.into_robj());
    let start = Instant::now();

    let labelvec = labels.as_string_vector().unwrap();
    let test_raw  = extract_scitemraw(&test,  None);      // <- helper from your code base
    let train_raw = extract_scitemraw(&train, None);
    let query_raw = extract_scitemraw(&query, Some(0));   // default target 0
    let feature_names_vec = feature_names.as_string_vector().unwrap();
    // -------------------------------------------------------------------
    // Dispatch per-model -------------------------------------------------
    // -------------------------------------------------------------------
    use crate::{scrna_mlr, scrna_ann, scrna_ann2l};       // adjust paths if needed
    let export = match model {
        "mlr" => match be.as_str() {
            "candle" => scrna_mlr::run_custom_candle(
                train_raw, test_raw, Some(query_raw),
                labelvec.len(), lr, epochs, Some(artifact_dir.clone()), verbose
            ),
            "wgpu"   => scrna_mlr::run_custom_wgpu(
                train_raw, test_raw, Some(query_raw),
                labelvec.len(), lr, epochs, Some(artifact_dir.clone()), verbose
            ),
            _        => scrna_mlr::run_custom_nd(
                train_raw, test_raw, Some(query_raw),
                labelvec.len(), lr, epochs, Some(artifact_dir.clone()), verbose
            ),
        },
        "ann" if hidden_size.is_some() => {
            let hs      = hidden_size.unwrap().as_integer_vector().unwrap();
            let hidden1 = *hs.first().unwrap() as usize;

            match be.as_str() {
                "candle" => scrna_ann::run_custom_candle(
                    train_raw, test_raw, Some(query_raw),
                    labelvec.len(), hidden1, lr, epochs,
                    Some(artifact_dir.clone()), verbose),
                "wgpu"   => scrna_ann::run_custom_wgpu   (
                    train_raw, test_raw, Some(query_raw),
                    labelvec.len(), hidden1, lr, epochs,
                    Some(artifact_dir.clone()), verbose),
                _        => scrna_ann::run_custom_nd     (
                    train_raw, test_raw, Some(query_raw),
                    labelvec.len(), hidden1, lr, epochs,
                    Some(artifact_dir.clone()), verbose),
            }
        }

        "ann2" if hidden_size.is_some() => {
            // eprint!("Running ANN2 with hidden_size = {:?}\n", hidden_size);
            let hs      = hidden_size.unwrap().as_integer_vector().unwrap();
            assert!(hs.len() == 2, "ann2 expects two hidden sizes");
            let (h1, h2) = (hs[0] as usize, hs[1] as usize);

            match be.as_str() {
                "candle" => scrna_ann2l::run_custom_candle(
                    train_raw, test_raw, Some(query_raw),
                    labelvec.len(), h1, h2, lr, epochs,
                    Some(artifact_dir.clone()), verbose),
                "wgpu"   => scrna_ann2l::run_custom_wgpu(
                    train_raw, test_raw, Some(query_raw),
                    labelvec.len(), h1, h2, lr, epochs,
                    Some(artifact_dir.clone()), verbose),
                _        => scrna_ann2l::run_custom_nd(
                    train_raw, test_raw, Some(query_raw),
                    labelvec.len(), h1, h2, lr, epochs,
                    Some(artifact_dir.clone()), verbose),
            }
        }

        _ => panic!("Unsupported model_type \"{model}\" or missing hidden_size"),
    };

    // -------------------------------------------------------------------
    // Assemble the return value (same shape for every model) ------------
    // -------------------------------------------------------------------
    let params = if model == "mlr" {
        list!(lr = export.lr,
              epochs = export.num_epochs,
              batch_size = export.batch_size,
              workers = export.num_workers,
              seed = export.seed)
    } else {
        list!(lr = export.lr,
              hidden_size = export.hidden_size,    // always present for ANN
              epochs = export.num_epochs,
              batch_size = export.batch_size,
              workers = export.num_workers,
              seed = export.seed)
    };

    let probs   = list!(export.probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>());
    let history = list!(train_acc = export.train_history.acc,
                        test_acc  = export.test_history.acc,
                        train_loss = export.train_history.loss,
                        test_loss  = export.test_history.loss);

    let duration = list!(
        total_duration   = start.elapsed().as_secs_f64(),
        training_duration = export.training_duration
    );
    // write metadata
    // artifacts/
    // │
    // ├─ model.mpk         ← Burn weights (already written)
    // └─ meta.mpk          ← MessagePack blob with:
    //      • feature_names : Vec<String>
    //      • class_labels  : Vec<String>  (or Vec<i32>)
    save_artifacts(artifact_dir.clone().as_str(), feature_names_vec, labelvec)
        .expect("Could not save artifacts");
    list!(params = params, probs = probs, history = history, duration = duration)
}

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
    batch_size: Robj
) -> List {
    // ── verbosity -----------------------------------------------------------
    let verbose = verbose
        .as_logical_vector()
        .unwrap()
        .first()
        .unwrap()
        .to_bool();

    // ── scalars -------------------------------------------------------------
    let model_path = model_path
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("`model_path` must be a string");
    if !Path::new(&model_path).exists() {
        panic!("Checkpoint not found: {model_path}");
    }

    let model_kind = model_type
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

    // ── optional hidden sizes ----------------------------------------------
    let h1 = usize_from_nullable(hidden1);
    let h2 = usize_from_nullable(hidden2);

    // if verbose {
    //     eprintln!("h1 = {:?}, h2 = {:?}", h1, h2);
    // }

    let batch_size = batch_size
        .as_real()
        .map(|x| x as usize)
        .or_else(|| batch_size.as_integer().map(|x| x as usize))
        .unwrap_or_else(|| {
            panic!("Batch size must be a real number or integer");
        });

    // ── query to Vec<SCItemRaw> --------------------------------------------
    if verbose {
        eprintln!("Preparing query data");
    }
    let query_raw = extract_scitemraw(&query, Some(0));

    // ── dispatch ------------------------------------------------------------
    if verbose {
        eprintln!("Running inference with model type \"{}\"", model_kind);
    }
    let probs: Vec<f32> = match model_kind.as_str() {
        "mlr" => infer_helper_mlr(
            model_path.to_string(),
            num_classes,
            num_features,
            query_raw,
            Some(batch_size)
        ),

        "ann1" | "ann" => {
            let size1 = h1.expect("`hidden1` must be supplied for ANN1 models");
            infer_helper_ann(
                model_path.to_string(),
                num_classes,
                num_features,
                query_raw,
                size1,
                Some(batch_size)
            )
        }

        "ann2" => {
            let size1 = h1.expect("`hidden1` must be supplied for ANN2 models");
            let size2 = h2.expect("`hidden2` must be supplied for ANN2 models");
            infer_helper_ann2l(
                model_path.to_string(),
                num_classes,
                num_features,
                query_raw,
                size1,
                size2,
                Some(batch_size)
            )
        }

        other => panic!("Unknown `model_type`: {other}  (use \"mlr\", \"ann1\", or \"ann2\")"),
    };

    // ── return to R ---------------------------------------------------------
    if verbose {
        eprintln!("Returning results");
    }
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
  // fn process_learning_obj_ann;
  // fn process_learning_obj_mlr;
  fn infer_from_model;
  fn process_learning_obj;
  // fn run_nb_test;
  fn process_learning_obj_nb;
}
