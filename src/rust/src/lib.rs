#![recursion_limit = "256"]
#![allow(non_snake_case)]
// #![allow(dead_code)]
// #![allow(unused_imports)]
// #![allow(unused_variables)]


// use burn::backend;
use extendr_api::prelude::*;
// use serde_json::de;
mod scrna_ann;
mod scrna_ann2l;
mod scrna_mlr;
mod scrna_conv;
mod utils;
mod pb;
mod common;
mod inference;
mod nb;
// mod signal;
mod signal;
mod em;
mod splat;
mod sparse;
mod train_sparse;


use std::path::Path;
use std::time::Instant;
use crate::common::*;
use crate::sparse::infer_sparse_parallel;
use crate::sparse::calculate_size_factors;
use crate::sparse::CscMatrix;
// use crate::inference::*;
use burn::prelude::Backend;
// use burn::tensor::Tensor;
// use burn::backend::{ndarray::{NdArray}, candle::Candle, Autodiff};
use burn::backend::{ndarray::{NdArray}, Autodiff};
// use crate::utils::{rmat_to_tensor, lgamma_plus_one};
// use crate::signal::{Consts, Params, train};

/// Global backend you’ll use everywhere
pub type B = Autodiff<NdArray<f32>>;
pub type Device = <B as Backend>::Device;

// pub type B = Autodiff<Candle<f32>>;
// pub type Device = <B as Backend>::Device;

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


// /// Process Robj learning objects for MLR #depracated
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
    hidden_size: Nullable<Integers>,      // <- optional
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

/// Sparse matrix training entry-point for MLR and ANN/ANN-2L.
///
/// Instead of receiving lists of cells from R, receives sparse matrices directly.
/// Size factors should be pre-computed in R (or set to 1.0 if data is already normalized).
///
/// @export
/// @keywords internal
#[extendr]
fn process_learning_obj_sparse(
    model_type: Robj,
    // Training data (sparse)
    train_x: Robj,
    train_i: Robj,
    train_p: Robj,
    train_dims: Robj,
    train_labels: Robj,
    train_size_factors: Robj,
    // Test data (sparse)
    test_x: Robj,
    test_i: Robj,
    test_p: Robj,
    test_dims: Robj,
    test_labels: Robj,
    test_size_factors: Robj,
    // Query data (sparse) - can be NULL
    query_x: Robj,
    query_i: Robj,
    query_p: Robj,
    query_dims: Robj,
    query_size_factors: Robj,
    // Metadata
    labels: Robj,
    feature_names: Robj,
    // Model params
    hidden_size: Nullable<Integers>,
    learning_rate: Robj,
    num_epochs: Robj,
    directory: Robj,
    verbose: Robj,
    backend: Robj,
) -> List {
    use crate::sparse::CscMatrix;
    use crate::train_sparse;
    use std::time::Instant;
    
    // -------------------------------------------------------------------
    // Parse common parameters
    // -------------------------------------------------------------------
    let model = model_type
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("Must supply model_type (\"mlr\" | \"ann\" | \"ann2\")");
    
    let be = parse_backend(&backend);
    let artifact_dir = parse_directory(&directory);
    
    let verbose: bool = verbose
        .as_logical_vector()
        .and_then(|v| v.first().map(|b| b.to_bool()))
        .unwrap_or(true);
    
    let lr = learning_rate
        .as_real_vector()
        .and_then(|v| v.first().copied())
        .unwrap_or(0.2);
    
    let epochs = num_epochs
        .as_real_vector()
        .and_then(|v| v.first().copied())
        .map(|x| x as usize)
        .unwrap_or(10);
    
    let start = Instant::now();
    
    let labelvec = labels.as_string_vector().expect("labels must be character vector");
    let feature_names_vec = feature_names.as_string_vector().expect("feature_names must be character vector");
    let num_classes = labelvec.len();
    
    // -------------------------------------------------------------------
    // Helper to parse dimensions
    // -------------------------------------------------------------------
    let parse_dims = |dims: &Robj| -> (usize, usize) {
        let d = dims.as_integer_vector().expect("dims must be integer vector");
        (d[0] as usize, d[1] as usize)
    };
    
    // -------------------------------------------------------------------
    // Parse training sparse matrix
    // -------------------------------------------------------------------
    let (train_nrow, train_ncol) = parse_dims(&train_dims);
    let train_mat = CscMatrix::from_r_parts(
        train_nrow,
        train_ncol,
        train_x.as_real_vector().expect("train_x must be numeric"),
        train_i.as_integer_vector().expect("train_i must be integer"),
        train_p.as_integer_vector().expect("train_p must be integer"),
    );
    let train_labels_vec: Vec<usize> = train_labels
        .as_integer_vector()
        .expect("train_labels must be integer")
        .iter()
        .map(|&x| x as usize)
        .collect();
    let train_sf: Vec<f64> = train_size_factors
        .as_real_vector()
        .expect("train_size_factors must be numeric");
    
    // -------------------------------------------------------------------
    // Parse test sparse matrix
    // -------------------------------------------------------------------
    let (test_nrow, test_ncol) = parse_dims(&test_dims);
    let test_mat = CscMatrix::from_r_parts(
        test_nrow,
        test_ncol,
        test_x.as_real_vector().expect("test_x must be numeric"),
        test_i.as_integer_vector().expect("test_i must be integer"),
        test_p.as_integer_vector().expect("test_p must be integer"),
    );
    let test_labels_vec: Vec<usize> = test_labels
        .as_integer_vector()
        .expect("test_labels must be integer")
        .iter()
        .map(|&x| x as usize)
        .collect();
    let test_sf: Vec<f64> = test_size_factors
        .as_real_vector()
        .expect("test_size_factors must be numeric");
    
    // -------------------------------------------------------------------
    // Parse query sparse matrix (optional - check if NULL)
    // -------------------------------------------------------------------
    let query_data: Option<(CscMatrix, Vec<f64>)> = if query_x.is_null() {
        None
    } else {
        let (qnrow, qncol) = parse_dims(&query_dims);
        let qmat = CscMatrix::from_r_parts(
            qnrow,
            qncol,
            query_x.as_real_vector().expect("query_x must be numeric"),
            query_i.as_integer_vector().expect("query_i must be integer"),
            query_p.as_integer_vector().expect("query_p must be integer"),
        );
        let qsf: Vec<f64> = query_size_factors
            .as_real_vector()
            .expect("query_size_factors must be numeric");
        Some((qmat, qsf))
    };
    
    if verbose {
        eprintln!(
            "Sparse training: train={}x{}, test={}x{}, query={}, classes={}",
            train_nrow, train_ncol,
            test_nrow, test_ncol,
            query_data.as_ref().map(|(m, _)| format!("{}x{}", m.nrow, m.ncol)).unwrap_or_else(|| "None".to_string()),
            num_classes
        );
    }
    
    // -------------------------------------------------------------------
    // Parse hidden_size for ANN models
    // -------------------------------------------------------------------
    let hidden_size_vec: Option<Vec<i32>> = match hidden_size {
        Nullable::NotNull(hs) => Some(hs.iter().map(|x| x.inner()).collect()),
        Nullable::Null => None,
    };
    
    // -------------------------------------------------------------------
    // Dispatch per-model and backend
    // -------------------------------------------------------------------
    let export = match model {
        "mlr" => match be.as_str() {
            "candle" => train_sparse::run_mlr_sparse_candle(
                &train_mat, &train_labels_vec, &train_sf,
                &test_mat, &test_labels_vec, &test_sf,
                query_data.as_ref(),
                num_classes, lr, epochs,
                Some(artifact_dir.clone()), verbose
            ),
            "wgpu" => train_sparse::run_mlr_sparse_wgpu(
                &train_mat, &train_labels_vec, &train_sf,
                &test_mat, &test_labels_vec, &test_sf,
                query_data.as_ref(),
                num_classes, lr, epochs,
                Some(artifact_dir.clone()), verbose
            ),
            _ => train_sparse::run_mlr_sparse_nd(
                &train_mat, &train_labels_vec, &train_sf,
                &test_mat, &test_labels_vec, &test_sf,
                query_data.as_ref(),
                num_classes, lr, epochs,
                Some(artifact_dir.clone()), verbose
            ),
        },
        
        "ann" => {
            let hs = hidden_size_vec.as_ref().expect("ann requires hidden_size");
            let hidden1 = hs[0] as usize;
            
            match be.as_str() {
                "candle" => train_sparse::run_ann_sparse_candle(
                    &train_mat, &train_labels_vec, &train_sf,
                    &test_mat, &test_labels_vec, &test_sf,
                    query_data.as_ref(),
                    num_classes, hidden1, lr, epochs,
                    Some(artifact_dir.clone()), verbose
                ),
                "wgpu" => train_sparse::run_ann_sparse_wgpu(
                    &train_mat, &train_labels_vec, &train_sf,
                    &test_mat, &test_labels_vec, &test_sf,
                    query_data.as_ref(),
                    num_classes, hidden1, lr, epochs,
                    Some(artifact_dir.clone()), verbose
                ),
                _ => train_sparse::run_ann_sparse_nd(
                    &train_mat, &train_labels_vec, &train_sf,
                    &test_mat, &test_labels_vec, &test_sf,
                    query_data.as_ref(),
                    num_classes, hidden1, lr, epochs,
                    Some(artifact_dir.clone()), verbose
                ),
            }
        }
        
        "ann2" => {
            let hs = hidden_size_vec.as_ref().expect("ann2 requires hidden_size");
            assert!(hs.len() >= 2, "ann2 expects two hidden sizes");
            let (h1, h2) = (hs[0] as usize, hs[1] as usize);
            
            match be.as_str() {
                "candle" => train_sparse::run_ann2_sparse_candle(
                    &train_mat, &train_labels_vec, &train_sf,
                    &test_mat, &test_labels_vec, &test_sf,
                    query_data.as_ref(),
                    num_classes, h1, h2, lr, epochs,
                    Some(artifact_dir.clone()), verbose
                ),
                "wgpu" => train_sparse::run_ann2_sparse_wgpu(
                    &train_mat, &train_labels_vec, &train_sf,
                    &test_mat, &test_labels_vec, &test_sf,
                    query_data.as_ref(),
                    num_classes, h1, h2, lr, epochs,
                    Some(artifact_dir.clone()), verbose
                ),
                _ => train_sparse::run_ann2_sparse_nd(
                    &train_mat, &train_labels_vec, &train_sf,
                    &test_mat, &test_labels_vec, &test_sf,
                    query_data.as_ref(),
                    num_classes, h1, h2, lr, epochs,
                    Some(artifact_dir.clone()), verbose
                ),
            }
        }
        
        _ => panic!("Unsupported model_type \"{model}\""),
    };
    
    // -------------------------------------------------------------------
    // Assemble return value
    // -------------------------------------------------------------------
    let params = if model == "mlr" {
        list!(
            lr = export.lr,
            epochs = export.num_epochs,
            batch_size = export.batch_size,
            workers = export.num_workers,
            seed = export.seed
        )
    } else {
        list!(
            lr = export.lr,
            hidden_size = export.hidden_size,
            epochs = export.num_epochs,
            batch_size = export.batch_size,
            workers = export.num_workers,
            seed = export.seed
        )
    };
    
    let probs: Vec<Robj> = export.probs.iter().map(|x| r!(x.clone())).collect();
    let probs_list = List::from_values(probs);
    
    let history = list!(
        train_acc = export.train_history.acc.clone(),
        test_acc = export.test_history.acc.clone(),
        train_loss = export.train_history.loss.clone(),
        test_loss = export.test_history.loss.clone()
    );
    
    let duration = list!(
        total_duration = start.elapsed().as_secs_f64(),
        training_duration = export.training_duration
    );
    
    // Save metadata
    save_artifacts(artifact_dir.as_str(), feature_names_vec, labelvec)
        .expect("Could not save artifacts");
    
    list!(
        params = params, 
        probs = probs_list, 
        history = history, 
        duration = duration
    )
}

// OLD FUNCTION NOT SPARSE
// #[extendr]          // @export
// fn infer_from_model(
//     model_path  : Robj,
//     query       : Robj,
//     num_classes : Robj,
//     num_features: Robj,
//     model_type  : Robj,
//     hidden1     : Nullable<Integers>,
//     hidden2     : Nullable<Integers>,
//     verbose     : Robj,
//     batch_size  : Robj,
//     backend: Robj,
//     num_threads: Robj
// ) -> List {
//     // ── verbosity ---------------------------------------------------------
//     let verbose = verbose
//         .as_logical_vector()
//         .unwrap()
//         .first()
//         .unwrap()
//         .to_bool();

//     let num_threads = num_threads
//         .as_integer()
//         .map(|x| x as usize)
//         .unwrap_or(1);
//     eprintln!("num_threads parsed as: {}", num_threads);
    
//     let backend = match backend.as_str_vector(){
//       Some(string_vec) => string_vec.first().unwrap().to_string(),
//       _ => panic!("Cound not find backend: '{:?}'", backend)
//     };
//     if ! ["wgpu", "candle", "nd"].contains(&backend.as_str()){
//       panic!("Cound not find backend: '{:?}'", backend)
//     }
//     // ── scalars -----------------------------------------------------------
//     let model_path = model_path
//         .as_str_vector()
//         .and_then(|v| v.first().cloned())
//         .expect("`model_path` must be a string");
//     if !Path::new(&model_path).exists() {
//         panic!("Checkpoint not found: {model_path}");
//     }

//     let model_kind_str = model_type
//         .as_str_vector()
//         .and_then(|v| v.first().cloned())
//         .unwrap_or_else(|| "mlr")
//         .to_lowercase();

//     let num_classes = num_classes
//         .as_integer_vector()
//         .and_then(|v| v.first().copied())
//         .expect("`num_classes` must be an integer") as usize;

//     let num_features = num_features
//         .as_integer_vector()
//         .and_then(|v| v.first().copied())
//         .expect("`num_features` must be an integer") as usize;

//     // ── optional hidden sizes --------------------------------------------
//     let h1 = usize_from_nullable(hidden1);
//     let h2 = usize_from_nullable(hidden2);

//     let batch_size = batch_size
//         .as_real()
//         .map(|x| x as usize)
//         .or_else(|| batch_size.as_integer().map(|x| x as usize))
//         .unwrap_or_else(|| panic!("`batch_size` must be numeric"));

//     // ── query to Vec<SCItemRaw> ------------------------------------------
//     if verbose { eprintln!("Preparing query data"); }
//     let query_raw = extract_scitemraw(&query, Some(0));

//     // ── decide which NetKind to run --------------------------------------
//     let net = match model_kind_str.as_str() {
//         "mlr" => NetKind::Mlr,

//         "ann1" | "ann" => {
//             let size1 = h1.expect("`hidden1` must be supplied for ANN1 models");
//             NetKind::Ann { hidden: size1 }
//         }

//         "ann2" => {
//             let size1 = h1.expect("`hidden1` must be supplied for ANN2 models");
//             let size2 = h2.expect("`hidden2` must be supplied for ANN2 models");
//             NetKind::Ann2 { hidden1: size1, hidden2: size2 }
//         }

//         other => panic!("Unknown `model_type`: {other}  (use \"mlr\", \"ann1\", or \"ann2\")"),
//     };

//     // ── run the generic inference ----------------------------------------
//     if verbose { eprintln!("Running inference using a {:?} model", net); }
//     if verbose { eprintln!("Using backend: {:?}", backend); }

//     // Use the appropriate backend based on the `backend` parameter
//     let probs = if num_threads > 1 && backend == "nd" {
//         if verbose {
//             eprintln!("Using parallel nd backend with {} threads", num_threads);
//         }
//         infer_nd_parallel(
//             &model_path,
//             net,
//             num_classes,
//             num_features,
//             query_raw,
//             Some(batch_size),
//             num_threads,
//             verbose,
//         )
//     } else {
//         match backend.as_str() {
//             "wgpu" => infer_wgpu(&model_path, net, num_classes, num_features, query_raw, Some(batch_size)),
//             "candle" => infer_candle(&model_path, net, num_classes, num_features, query_raw, Some(batch_size)),
//             "nd" => infer_nd(&model_path, net, num_classes, num_features, query_raw, Some(batch_size)),
//             _ => panic!("Unknown backend: {}", backend),
//         }
//     };
//     // let probs = match backend.as_str() {
//     //     "wgpu" => infer_wgpu(
//     //         &model_path,
//     //         net,
//     //         num_classes,
//     //         num_features,
//     //         query_raw,
//     //         Some(batch_size)),
//     //     "candle" => infer_candle(
//     //         &model_path,
//     //         net,
//     //         num_classes,
//     //         num_features,
//     //         query_raw,
//     //         Some(batch_size)),
//     //     "nd" => infer_nd(
//     //         &model_path,
//     //         net,
//     //         num_classes,
//     //         num_features,
//     //         query_raw,
//     //         Some(batch_size)),
//     //     _ => panic!("Unknown backend: {}", backend),
//     // };

//     // ── return to R -------------------------------------------------------
//     if verbose { eprintln!("Returning results"); }
//     list!(probs = probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>())
// }


// // ---------- util -------------------------------------------------------------
// // fn usize_from_nullable(n: Nullable<Integers>) -> Option<usize> {
// //     match n {
// //         Nullable::Null => None,

// //         Nullable::NotNull(robj) => {
// //             // Parse again as Nullable<Option<i32>>
// //             let parsed: Nullable<Option<i32>> = robj.try_into().ok()?;

// //             match parsed {
// //                 Nullable::Null          => None,          // shouldn’t occur
// //                 Nullable::NotNull(None) => None,          // NA
// //                 Nullable::NotNull(Some(x)) => Some(x as usize),
// //             }
// //         }
// //     }
// // }

// // fn usize_from_nullable(n: Nullable<Integers>) -> Option<usize> {
// //     match n {
// //         Nullable::Null => None,
// //         Nullable::NotNull(x) => {
// //             if x.is_number() {
// //                 let vec = x.into_robj().as_integer_vector().unwrap();
// //                 let int = vec.first().unwrap();
// //                 Some(*int as usize)
// //             } else {
// //                 None
// //             }
// //         }
// //     }
// // }


/// @export
/// @keywords internal
#[extendr]
pub fn infer_sparse(
    x: Robj,
    i: Robj,
    p: Robj,
    dims: Robj,
    size_factors: Robj,
    model_path: Robj,
    model_type: Robj,
    num_classes: Robj,
    hidden1: Nullable<Integers>,
    hidden2: Nullable<Integers>,
    batch_size: Robj,
    num_threads: Robj,
    verbose: Robj,
) -> List {
    // Parse verbose
    let verbose = verbose
        .as_logical_vector()
        .and_then(|v| v.first().map(|b| b.to_bool()))
        .unwrap_or(false);
    
    if verbose {
        eprintln!("Parsing sparse matrix from R...");
    }
    
    // Parse dimensions
    let dims_vec = dims.as_integer_vector().expect("dims must be integer vector");
    let nrow = dims_vec[0] as usize;
    let ncol = dims_vec[1] as usize;
    
    // Parse sparse matrix components
    let x_vec: Vec<f64> = x.as_real_vector().expect("x must be numeric vector");
    let i_vec: Vec<i32> = i.as_integer_vector().expect("i must be integer vector");
    let p_vec: Vec<i32> = p.as_integer_vector().expect("p must be integer vector");
    
    // Create CSC matrix
    let mat = CscMatrix::from_r_parts(nrow, ncol, x_vec, i_vec, p_vec);
    
    if verbose {
        eprintln!("Sparse matrix: {} rows × {} cols, {} non-zeros", 
                  mat.nrow, mat.ncol, mat.x.len());
    }
    
    // Parse or compute size factors
    let sf: Vec<f64> = if size_factors.is_null() {
        if verbose {
            eprintln!("Computing size factors...");
        }
        calculate_size_factors(&mat)
    } else {
        size_factors.as_real_vector().expect("size_factors must be numeric vector")
    };
    
    // Parse model parameters
    let model_path_str = model_path
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("model_path must be a string");
    
    let model_type_str = model_type
        .as_str_vector()
        .and_then(|v| v.first().cloned())
        .expect("model_type must be a string");
    
    let num_classes_val = num_classes
        .as_integer_vector()
        .and_then(|v| v.first().copied())
        .expect("num_classes must be integer") as usize;
    
    let hidden1_val = parse_nullable_int(hidden1);
    let hidden2_val = parse_nullable_int(hidden2);
    
    let batch_size_val = batch_size
        .as_integer_vector()
        .and_then(|v| v.first().copied())
        .unwrap_or(1024) as usize;
    
    let num_threads_val = num_threads
        .as_integer_vector()
        .and_then(|v| v.first().copied())
        .unwrap_or(1) as usize;
    
    if verbose {
        eprintln!("Model: {}, classes: {}, batch_size: {}, threads: {}",
                  model_type_str, num_classes_val, batch_size_val, num_threads_val);
    }
    
    // Run inference
    let probs = infer_sparse_parallel(
        &mat,
        &sf,
        &model_path_str,
        &model_type_str,
        num_classes_val,
        hidden1_val,
        hidden2_val,
        batch_size_val,
        num_threads_val,
        verbose,
    );
    
    if verbose {
        eprintln!("Inference complete: {} probability values", probs.len());
    }
    
    // Return as R list
    list!(probs = probs.iter().map(|x| r!(x)).collect::<Vec<Robj>>())
}

/// Helper to parse Nullable<Integers> to Option<usize>
fn parse_nullable_int(n: Nullable<Integers>) -> Option<usize> {
    match n {
        Nullable::Null => None,
        Nullable::NotNull(x) => {
            if x.len() > 0 {
                let robj = x.into_robj();
                robj.as_integer_vector()
                    .and_then(|v| v.first().copied())
                    .filter(|&i| i >= 0)
                    .map(|i| i as usize)
            } else {
                None
            }
        }
    }
}



// fn usize_from_nullable(n: Nullable<Integers>) -> Option<usize> {
//     match n {
//         Nullable::Null => None,
//         Nullable::NotNull(x) => {
//             if x.is_number() {
//                 let robj = x.into_robj();
                
//                 if let Some(vec) = robj.as_integer_vector() {
//                     if let Some(int) = vec.first() {
//                         if *int >= 0 {
//                             return Some(*int as usize);
//                         }
//                     }
//                 }
//                 None
//             } else {
//                 None
//             }
//         }
//     }
// }

///@export
///@keywords internal
#[extendr]

fn fit_deconv(
    sigs: Robj,
    bulk: Robj,
    gene_lengths: Robj,
    w_vec: Robj,
    backend: Robj,
    insert_size: Robj,
    init_log_exp: Robj,
    lr: Robj,
    l1_lambda: Robj,
    l2_lambda: Robj,
    max_iter: Robj,
    poll_interval: Robj,
    ll_tol: Robj,
    sparsity_tol: Robj,
    verbose: Robj,
) -> Result<List>{
    signal::fit_deconv(
        sigs,
        bulk,
        gene_lengths,
        w_vec,
        backend,
        insert_size,
        init_log_exp,
        lr,
        l1_lambda,
        l2_lambda,
        max_iter,
        poll_interval,
        ll_tol,
        sparsity_tol,
        verbose
    )
}


// Use EM for deconvolution prediction.
///@export
///@keywords internal
#[extendr]
fn fit_deconvolution_em(
    sigs: Robj,
    bulk: Robj,
    gene_lengths: Robj,
    gene_weights: Robj,
    max_iter: Robj,
    tolerance: Robj,
    l1_lambda: Robj,
    verbose: Robj,
) -> Result<List> {
    em::fit_deconv_em(sigs, bulk, gene_lengths, gene_weights, max_iter, tolerance, l1_lambda, verbose)
}

///@export
///@keywords internal
#[extendr]
fn splat_bulk_reference_rust(
    counts_matrix: Robj,
    universe: Robj,
    sizes: Robj,
    bandwidth: Robj,
    n_cells_per_bulk: Robj,
    replace_counts: Robj,
    seed: Robj,
    verbose: Robj,
) -> Result<List>{
    splat::splat_bulk_reference_rust_core(
        counts_matrix,
        universe,
        sizes,
        bandwidth,
        n_cells_per_bulk,
        replace_counts,
        seed,
        verbose
    )
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
  mod viewmastR;
  fn readR;
  fn computeSparseRowVariances;
  fn fit_deconv;
  fn fit_deconvolution_em;
  // fn process_learning_obj_ann;
  // fn process_learning_obj_mlr;
//   fn infer_from_model;
  fn process_learning_obj;
  fn process_learning_obj_nb;
  fn process_learning_obj_sparse;
  fn splat_bulk_reference_rust;
  // fn run_nb_test;
  fn infer_sparse;

}
