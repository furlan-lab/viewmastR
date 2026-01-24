//! High-performance inference module
//! 
//! This module provides optimized inference that:
//! 1. Accepts sparse matrices directly from R (no dense conversion in R)
//! 2. Parallelizes across cells using Rayon
//! 3. Normalizes on-the-fly during inference

use crate::scrna_mlr::ModelConfig as MlrCfg;
use crate::scrna_ann::ModelConfig as AnnCfg;
use crate::scrna_ann2l::ModelConfig as Ann2Cfg;

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{DType, Tensor},
};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use rayon::prelude::*;

type B = NdArray<f32, i32>;

/// Compressed Sparse Column (CSC) matrix - matches R's dgCMatrix format
#[derive(Debug, Clone)]
pub struct CscMatrix {
    /// Number of rows (genes)
    pub nrow: usize,
    /// Number of columns (cells)  
    pub ncol: usize,
    /// Non-zero values (length = nnz)
    pub x: Vec<f64>,
    /// Row indices for each non-zero (length = nnz)
    pub i: Vec<usize>,
    /// Column pointers (length = ncol + 1)
    pub p: Vec<usize>,
}

impl CscMatrix {
    /// Create from R dgCMatrix components
    pub fn from_r_parts(
        nrow: usize,
        ncol: usize,
        x: Vec<f64>,
        i: Vec<i32>,
        p: Vec<i32>,
    ) -> Self {
        Self {
            nrow,
            ncol,
            x,
            i: i.into_iter().map(|v| v as usize).collect(),
            p: p.into_iter().map(|v| v as usize).collect(),
        }
    }

    /// Get column sum (for size factor calculation)
    #[inline]
    pub fn col_sum(&self, col: usize) -> f64 {
        let start = self.p[col];
        let end = self.p[col + 1];
        self.x[start..end].iter().sum()
    }

    /// Calculate all column sums in parallel
    pub fn col_sums_parallel(&self) -> Vec<f64> {
        (0..self.ncol)
            .into_par_iter()
            .map(|col| self.col_sum(col))
            .collect()
    }

    /// Extract and normalize a single column, returning as f32 for inference
    #[inline]
    pub fn col_normalized_f32(&self, col: usize, size_factor: f64) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.nrow];
        let start = self.p[col];
        let end = self.p[col + 1];
        
        for idx in start..end {
            let row = self.i[idx];
            dense[row] = (self.x[idx] / size_factor) as f32;
        }
        dense
    }
}

/// Run inference on a batch of pre-normalized cell vectors
fn infer_batch_mlr(
    model: &crate::scrna_mlr::Model<B>,
    cells: &[Vec<f32>],
    device: &NdArrayDevice,
) -> Vec<f32> {
    if cells.is_empty() {
        return vec![];
    }
    
    let n_cells = cells.len();
    let n_features = cells[0].len();
    
    let flat: Vec<f32> = cells.iter().flat_map(|c| c.iter().copied()).collect();
    let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device)
        .reshape([n_cells, n_features]);
    
    let output = model.forward(input);
    
    let mut data = output.into_data();
    if data.dtype != DType::F32 {
        data = data.convert::<f32>();
    }
    data.into_vec::<f32>().unwrap()
}

fn infer_batch_ann(
    model: &crate::scrna_ann::Model<B>,
    cells: &[Vec<f32>],
    device: &NdArrayDevice,
) -> Vec<f32> {
    if cells.is_empty() {
        return vec![];
    }
    
    let n_cells = cells.len();
    let n_features = cells[0].len();
    
    let flat: Vec<f32> = cells.iter().flat_map(|c| c.iter().copied()).collect();
    let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device)
        .reshape([n_cells, n_features]);
    
    let output = model.forward(input);
    
    let mut data = output.into_data();
    if data.dtype != DType::F32 {
        data = data.convert::<f32>();
    }
    data.into_vec::<f32>().unwrap()
}

fn infer_batch_ann2(
    model: &crate::scrna_ann2l::Model<B>,
    cells: &[Vec<f32>],
    device: &NdArrayDevice,
) -> Vec<f32> {
    if cells.is_empty() {
        return vec![];
    }
    
    let n_cells = cells.len();
    let n_features = cells[0].len();
    
    let flat: Vec<f32> = cells.iter().flat_map(|c| c.iter().copied()).collect();
    let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device)
        .reshape([n_cells, n_features]);
    
    let output = model.forward(input);
    
    let mut data = output.into_data();
    if data.dtype != DType::F32 {
        data = data.convert::<f32>();
    }
    data.into_vec::<f32>().unwrap()
}

/// Main entry point: parallel inference directly from sparse matrix
pub fn infer_sparse_parallel(
    mat: &CscMatrix,
    size_factors: &[f64],
    model_path: &str,
    model_type: &str,
    num_classes: usize,
    hidden1: Option<usize>,
    hidden2: Option<usize>,
    batch_size: usize,
    num_threads: usize,
    verbose: bool,
) -> Vec<f32> {
    let n_cells = mat.ncol;
    let n_features = mat.nrow;
    
    if verbose {
        eprintln!(
            "Sparse parallel inference: {} cells × {} features, {} threads, batch_size {}",
            n_cells, n_features, num_threads, batch_size
        );
    }
    
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build thread pool");
    
    let chunk_size = ((n_cells / num_threads) + 1).max(batch_size);
    let chunks: Vec<Vec<usize>> = (0..n_cells)
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|c| c.to_vec())
        .collect();
    
    if verbose {
        eprintln!("Split into {} chunks of ~{} cells", chunks.len(), chunk_size);
    }
    
    let model_path = model_path.to_string();
    let model_type = model_type.to_string();
    
    let results: Vec<Vec<f32>> = pool.install(|| {
        chunks
            .into_par_iter()
            .enumerate()
            .map(|(chunk_idx, cell_indices)| {
                if verbose {
                    eprintln!("Processing chunk {}", chunk_idx + 1);
                }
                
                let device = NdArrayDevice::default();
                let mut chunk_results = Vec::with_capacity(cell_indices.len() * num_classes);
                
                match model_type.as_str() {
                    "mlr" => {
                        let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                            .load(model_path.clone().into(), &device)
                            .expect("Failed to load MLR weights");
                        let model = MlrCfg::new(num_classes)
                            .init(n_features, device.clone())
                            .load_record(rec);
                        
                        for batch_start in (0..cell_indices.len()).step_by(batch_size) {
                            let batch_end = (batch_start + batch_size).min(cell_indices.len());
                            let batch_indices = &cell_indices[batch_start..batch_end];
                            
                            let batch_cells: Vec<Vec<f32>> = batch_indices
                                .iter()
                                .map(|&col| mat.col_normalized_f32(col, size_factors[col]))
                                .collect();
                            
                            let batch_output = infer_batch_mlr(&model, &batch_cells, &device);
                            chunk_results.extend(batch_output);
                        }
                    }
                    "ann" | "ann1" => {
                        let hidden = hidden1.expect("hidden1 required for ANN model");
                        let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                            .load(model_path.clone().into(), &device)
                            .expect("Failed to load ANN weights");
                        let model = AnnCfg::new(num_classes, 0, hidden, 0.0)
                            .init(n_features, device.clone())
                            .load_record(rec);
                        
                        for batch_start in (0..cell_indices.len()).step_by(batch_size) {
                            let batch_end = (batch_start + batch_size).min(cell_indices.len());
                            let batch_indices = &cell_indices[batch_start..batch_end];
                            
                            let batch_cells: Vec<Vec<f32>> = batch_indices
                                .iter()
                                .map(|&col| mat.col_normalized_f32(col, size_factors[col]))
                                .collect();
                            
                            let batch_output = infer_batch_ann(&model, &batch_cells, &device);
                            chunk_results.extend(batch_output);
                        }
                    }
                    "ann2" => {
                        let h1 = hidden1.expect("hidden1 required for ANN2 model");
                        let h2 = hidden2.expect("hidden2 required for ANN2 model");
                        let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                            .load(model_path.clone().into(), &device)
                            .expect("Failed to load ANN2 weights");
                        let model = Ann2Cfg::new(num_classes, 0, h1, h2, 0.0)
                            .init(n_features, device.clone())
                            .load_record(rec);
                        
                        for batch_start in (0..cell_indices.len()).step_by(batch_size) {
                            let batch_end = (batch_start + batch_size).min(cell_indices.len());
                            let batch_indices = &cell_indices[batch_start..batch_end];
                            
                            let batch_cells: Vec<Vec<f32>> = batch_indices
                                .iter()
                                .map(|&col| mat.col_normalized_f32(col, size_factors[col]))
                                .collect();
                            
                            let batch_output = infer_batch_ann2(&model, &batch_cells, &device);
                            chunk_results.extend(batch_output);
                        }
                    }
                    other => panic!("Unknown model type: {}", other),
                }
                
                chunk_results
            })
            .collect()
    });
    
    results.into_iter().flatten().collect()
}

/// Calculate size factors from sparse matrix column sums
pub fn calculate_size_factors(mat: &CscMatrix) -> Vec<f64> {
    let col_sums = mat.col_sums_parallel();
    let mean_sum: f64 = col_sums.iter().sum::<f64>() / col_sums.len() as f64;
    col_sums.into_iter().map(|s| s / mean_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_factors() {
        let mat = CscMatrix {
            nrow: 3,
            ncol: 3,
            x: vec![1.0, 4.0, 3.0, 2.0, 5.0],
            i: vec![0, 2, 1, 0, 2],
            p: vec![0, 2, 3, 5],
        };
        
        let sf = calculate_size_factors(&mat);
        assert!((sf[0] - 1.0).abs() < 0.001);
        assert!((sf[1] - 0.6).abs() < 0.001);
        assert!((sf[2] - 1.4).abs() < 0.001);
    }

    #[test]
    fn test_col_sums() {
        let mat = CscMatrix {
            nrow: 3,
            ncol: 3,
            x: vec![1.0, 4.0, 3.0, 2.0, 5.0],
            i: vec![0, 2, 1, 0, 2],
            p: vec![0, 2, 3, 5],
        };
        
        assert_eq!(mat.col_sum(0), 5.0);
        assert_eq!(mat.col_sum(1), 3.0);
        assert_eq!(mat.col_sum(2), 7.0);
    }
}


// //! High-performance inference module
// //! 
// //! This module provides optimized inference that:
// //! 1. Accepts sparse matrices directly from R (no dense conversion in R)
// //! 2. Parallelizes across cells using Rayon
// //! 3. Normalizes on-the-fly during inference

// // use crate::sparse::CscMatrix;
// use crate::scrna_mlr::ModelConfig as MlrCfg;
// use crate::scrna_ann::ModelConfig as AnnCfg;
// use crate::scrna_ann2l::ModelConfig as Ann2Cfg;

// use burn::{
//     module::Module,
//     record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
//     tensor::{DType, Tensor},
// };
// use burn::backend::ndarray::{NdArray, NdArrayDevice};
// // use num_traits::ToPrimitive;
// use rayon::prelude::*;

// type B = NdArray<f32, i32>;



// /// Compressed Sparse Column (CSC) matrix - matches R's dgCMatrix format
// #[derive(Debug, Clone)]
// pub struct CscMatrix {
//     /// Number of rows (genes)
//     pub nrow: usize,
//     /// Number of columns (cells)  
//     pub ncol: usize,
//     /// Non-zero values (length = nnz)
//     pub x: Vec<f64>,
//     /// Row indices for each non-zero (length = nnz)
//     pub i: Vec<usize>,
//     /// Column pointers (length = ncol + 1)
//     pub p: Vec<usize>,
// }

// impl CscMatrix {
//     /// Create from R dgCMatrix components
//     pub fn from_r_parts(
//         nrow: usize,
//         ncol: usize,
//         x: Vec<f64>,
//         i: Vec<i32>,
//         p: Vec<i32>,
//     ) -> Self {
//         Self {
//             nrow,
//             ncol,
//             x,
//             i: i.into_iter().map(|v| v as usize).collect(),
//             p: p.into_iter().map(|v| v as usize).collect(),
//         }
//     }

//     // /// Get a single column as a dense vector
//     // #[inline]
//     // pub fn col_to_dense(&self, col: usize) -> Vec<f64> {
//     //     let mut dense = vec![0.0; self.nrow];
//     //     let start = self.p[col];
//     //     let end = self.p[col + 1];
        
//     //     for idx in start..end {
//     //         let row = self.i[idx];
//     //         dense[row] = self.x[idx];
//     //     }
//     //     dense
//     // }

//     /// Get column sum (for size factor calculation)
//     #[inline]
//     pub fn col_sum(&self, col: usize) -> f64 {
//         let start = self.p[col];
//         let end = self.p[col + 1];
//         self.x[start..end].iter().sum()
//     }

//     /// Calculate all column sums in parallel
//     pub fn col_sums_parallel(&self) -> Vec<f64> {
//         (0..self.ncol)
//             .into_par_iter()
//             .map(|col| self.col_sum(col))
//             .collect()
//     }

//     /// Extract and normalize a single column, returning as f32 for inference
//     #[inline]
//     pub fn col_normalized_f32(&self, col: usize, size_factor: f64) -> Vec<f32> {
//         let mut dense = vec![0.0f32; self.nrow];
//         let start = self.p[col];
//         let end = self.p[col + 1];
        
//         for idx in start..end {
//             let row = self.i[idx];
//             dense[row] = (self.x[idx] / size_factor) as f32;
//         }
//         dense
//     }
// }

// /// Process cells in parallel: normalize and prepare for inference
// /// Returns Vec of (cell_index, normalized_dense_vector)
// // pub fn prepare_cells_parallel(
// //     mat: &CscMatrix,
// //     size_factors: &[f64],
// //     cell_indices: &[usize],
// // ) -> Vec<Vec<f32>> {
// //     cell_indices
// //         .par_iter()
// //         .map(|&col| mat.col_normalized_f32(col, size_factors[col]))
// //         .collect()
// // }

// /// Subset the matrix to specific rows (genes/features)
// /// Returns a new CscMatrix with only the selected rows
// // pub fn subset_rows(mat: &CscMatrix, row_indices: &[usize]) -> CscMatrix {
// //     let new_nrow = row_indices.len();
    
// //     // Create a mapping from old row index to new row index
// //     let mut row_map = vec![None; mat.nrow];
// //     for (new_idx, &old_idx) in row_indices.iter().enumerate() {
// //         row_map[old_idx] = Some(new_idx);
// //     }
    
// //     let mut new_x = Vec::new();
// //     let mut new_i = Vec::new();
// //     let mut new_p = vec![0usize];
    
// //     for col in 0..mat.ncol {
// //         let start = mat.p[col];
// //         let end = mat.p[col + 1];
        
// //         for idx in start..end {
// //             let old_row = mat.i[idx];
// //             if let Some(new_row) = row_map[old_row] {
// //                 new_x.push(mat.x[idx]);
// //                 new_i.push(new_row);
// //             }
// //         }
// //         new_p.push(new_x.len());
// //     }
    
// //     CscMatrix {
// //         nrow: new_nrow,
// //         ncol: mat.ncol,
// //         x: new_x,
// //         i: new_i,
// //         p: new_p,
// //     }
// // }


// // /// Model type enum for runtime dispatch
// // #[derive(Clone, Debug)]
// // pub enum ModelType {
// //     Mlr,
// //     Ann { hidden: usize },
// //     Ann2 { hidden1: usize, hidden2: usize },
// // }

// /// Run inference on a batch of pre-normalized cell vectors
// fn infer_batch_mlr(
//     model: &crate::scrna_mlr::Model<B>,
//     cells: &[Vec<f32>],
//     device: &NdArrayDevice,
// ) -> Vec<f32> {
//     if cells.is_empty() {
//         return vec![];
//     }
    
//     let n_cells = cells.len();
//     let n_features = cells[0].len();
    
//     // Build input tensor: [n_cells, n_features]
//     let flat: Vec<f32> = cells.iter().flat_map(|c| c.iter().copied()).collect();
//     let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device)
//         .reshape([n_cells, n_features]);
    
//     // Forward pass
//     let output = model.forward(input);
    
//     // Extract results
//     let mut data = output.into_data();
//     if data.dtype != DType::F32 {
//         data = data.convert::<f32>();
//     }
//     data.into_vec::<f32>().unwrap()
// }

// fn infer_batch_ann(
//     model: &crate::scrna_ann::Model<B>,
//     cells: &[Vec<f32>],
//     device: &NdArrayDevice,
// ) -> Vec<f32> {
//     if cells.is_empty() {
//         return vec![];
//     }
    
//     let n_cells = cells.len();
//     let n_features = cells[0].len();
    
//     let flat: Vec<f32> = cells.iter().flat_map(|c| c.iter().copied()).collect();
//     let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device)
//         .reshape([n_cells, n_features]);
    
//     let output = model.forward(input);
    
//     let mut data = output.into_data();
//     if data.dtype != DType::F32 {
//         data = data.convert::<f32>();
//     }
//     data.into_vec::<f32>().unwrap()
// }

// fn infer_batch_ann2(
//     model: &crate::scrna_ann2l::Model<B>,
//     cells: &[Vec<f32>],
//     device: &NdArrayDevice,
// ) -> Vec<f32> {
//     if cells.is_empty() {
//         return vec![];
//     }
    
//     let n_cells = cells.len();
//     let n_features = cells[0].len();
    
//     let flat: Vec<f32> = cells.iter().flat_map(|c| c.iter().copied()).collect();
//     let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device)
//         .reshape([n_cells, n_features]);
    
//     let output = model.forward(input);
    
//     let mut data = output.into_data();
//     if data.dtype != DType::F32 {
//         data = data.convert::<f32>();
//     }
//     data.into_vec::<f32>().unwrap()
// }

// /// Main entry point: parallel inference directly from sparse matrix
// /// 
// /// # Arguments
// /// * `mat` - Sparse CSC matrix (genes x cells), already subset to selected features
// /// * `size_factors` - Pre-computed size factors for each cell
// /// * `model_path` - Path to the .mpk model file
// /// * `model_type` - "mlr", "ann", or "ann2"
// /// * `num_classes` - Number of output classes
// /// * `hidden1` - Hidden layer 1 size (for ANN models)
// /// * `hidden2` - Hidden layer 2 size (for ANN2 models)
// /// * `batch_size` - Cells per inference batch
// /// * `num_threads` - Number of parallel threads
// /// * `verbose` - Print progress messages
// /// 
// /// # Returns
// /// Flat vector of log-odds: [cell0_class0, cell0_class1, ..., cell1_class0, ...]
// pub fn infer_sparse_parallel(
//     mat: &CscMatrix,
//     size_factors: &[f64],
//     model_path: &str,
//     model_type: &str,
//     num_classes: usize,
//     hidden1: Option<usize>,
//     hidden2: Option<usize>,
//     batch_size: usize,
//     num_threads: usize,
//     verbose: bool,
// ) -> Vec<f32> {
//     let n_cells = mat.ncol;
//     let n_features = mat.nrow;
    
//     if verbose {
//         eprintln!(
//             "Sparse parallel inference: {} cells × {} features, {} threads, batch_size {}",
//             n_cells, n_features, num_threads, batch_size
//         );
//     }
    
//     // Build thread pool
//     let pool = rayon::ThreadPoolBuilder::new()
//         .num_threads(num_threads)
//         .build()
//         .expect("Failed to build thread pool");
    
//     // Create cell index chunks for parallel processing
//     let chunk_size = ((n_cells / num_threads) + 1).max(batch_size);
//     let chunks: Vec<Vec<usize>> = (0..n_cells)
//         .collect::<Vec<_>>()
//         .chunks(chunk_size)
//         .map(|c| c.to_vec())
//         .collect();
    
//     if verbose {
//         eprintln!("Split into {} chunks of ~{} cells", chunks.len(), chunk_size);
//     }
    
//     // Clone model_path for thread safety
//     let model_path = model_path.to_string();
//     let model_type = model_type.to_string();
    
//     // Process chunks in parallel
//     let results: Vec<Vec<f32>> = pool.install(|| {
//         chunks
//             .into_par_iter()
//             .enumerate()
//             .map(|(chunk_idx, cell_indices)| {
//                 if verbose {
//                     eprintln!("Processing chunk {}", chunk_idx + 1);
//                 }
                
//                 // Each thread gets its own device and loads the model
//                 let device = NdArrayDevice::default();
                
//                 // Process cells in batches within this chunk
//                 let mut chunk_results = Vec::with_capacity(cell_indices.len() * num_classes);
                
//                 // Load model based on type (once per chunk/thread)
//                 match model_type.as_str() {
//                     "mlr" => {
//                         let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//                             .load(model_path.clone().into(), &device)
//                             .expect("Failed to load MLR weights");
//                         let model = MlrCfg::new(num_classes)
//                             .init(n_features, device.clone())
//                             .load_record(rec);
                        
//                         for batch_start in (0..cell_indices.len()).step_by(batch_size) {
//                             let batch_end = (batch_start + batch_size).min(cell_indices.len());
//                             let batch_indices = &cell_indices[batch_start..batch_end];
                            
//                             let batch_cells: Vec<Vec<f32>> = batch_indices
//                                 .iter()
//                                 .map(|&col| mat.col_normalized_f32(col, size_factors[col]))
//                                 .collect();
                            
//                             let batch_output = infer_batch_mlr(&model, &batch_cells, &device);
//                             chunk_results.extend(batch_output);
//                         }
//                     }
//                     "ann" | "ann1" => {
//                         let hidden = hidden1.expect("hidden1 required for ANN model");
//                         let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//                             .load(model_path.clone().into(), &device)
//                             .expect("Failed to load ANN weights");
//                         let model = AnnCfg::new(num_classes, 0, hidden, 0.0)
//                             .init(n_features, device.clone())
//                             .load_record(rec);
                        
//                         for batch_start in (0..cell_indices.len()).step_by(batch_size) {
//                             let batch_end = (batch_start + batch_size).min(cell_indices.len());
//                             let batch_indices = &cell_indices[batch_start..batch_end];
                            
//                             let batch_cells: Vec<Vec<f32>> = batch_indices
//                                 .iter()
//                                 .map(|&col| mat.col_normalized_f32(col, size_factors[col]))
//                                 .collect();
                            
//                             let batch_output = infer_batch_ann(&model, &batch_cells, &device);
//                             chunk_results.extend(batch_output);
//                         }
//                     }
//                     "ann2" => {
//                         let h1 = hidden1.expect("hidden1 required for ANN2 model");
//                         let h2 = hidden2.expect("hidden2 required for ANN2 model");
//                         let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//                             .load(model_path.clone().into(), &device)
//                             .expect("Failed to load ANN2 weights");
//                         let model = Ann2Cfg::new(num_classes, 0, h1, h2, 0.0)
//                             .init(n_features, device.clone())
//                             .load_record(rec);
                        
//                         for batch_start in (0..cell_indices.len()).step_by(batch_size) {
//                             let batch_end = (batch_start + batch_size).min(cell_indices.len());
//                             let batch_indices = &cell_indices[batch_start..batch_end];
                            
//                             let batch_cells: Vec<Vec<f32>> = batch_indices
//                                 .iter()
//                                 .map(|&col| mat.col_normalized_f32(col, size_factors[col]))
//                                 .collect();
                            
//                             let batch_output = infer_batch_ann2(&model, &batch_cells, &device);
//                             chunk_results.extend(batch_output);
//                         }
//                     }
//                     other => panic!("Unknown model type: {}", other),
//                 }
                
//                 chunk_results
//             })
//             .collect()
//     });
    
//     // Flatten results (already in correct order since chunks are sequential)
//     results.into_iter().flatten().collect()
// }

// /// Calculate size factors from sparse matrix column sums
// pub fn calculate_size_factors(mat: &CscMatrix) -> Vec<f64> {
//     let col_sums = mat.col_sums_parallel();
//     let mean_sum: f64 = col_sums.iter().sum::<f64>() / col_sums.len() as f64;
//     col_sums.into_iter().map(|s| s / mean_sum).collect()
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_size_factors() {
//         let mat = CscMatrix {
//             nrow: 3,
//             ncol: 3,
//             x: vec![1.0, 4.0, 3.0, 2.0, 5.0],
//             i: vec![0, 2, 1, 0, 2],
//             p: vec![0, 2, 3, 5],
//         };
        
//         let sf = calculate_size_factors(&mat);
//         // col sums: [5, 3, 7], mean = 5
//         // sf: [5/5, 3/5, 7/5] = [1.0, 0.6, 1.4]
//         assert!((sf[0] - 1.0).abs() < 0.001);
//         assert!((sf[1] - 0.6).abs() < 0.001);
//         assert!((sf[2] - 1.4).abs() < 0.001);
//     }


//     // #[test]
//     // fn test_col_to_dense() {
//     //     // 3x3 matrix:
//     //     // [1, 0, 2]
//     //     // [0, 3, 0]
//     //     // [4, 0, 5]
//     //     let mat = CscMatrix {
//     //         nrow: 3,
//     //         ncol: 3,
//     //         x: vec![1.0, 4.0, 3.0, 2.0, 5.0],
//     //         i: vec![0, 2, 1, 0, 2],
//     //         p: vec![0, 2, 3, 5],
//     //     };
        
//     //     assert_eq!(mat.col_to_dense(0), vec![1.0, 0.0, 4.0]);
//     //     assert_eq!(mat.col_to_dense(1), vec![0.0, 3.0, 0.0]);
//     //     assert_eq!(mat.col_to_dense(2), vec![2.0, 0.0, 5.0]);
//     // }

//     #[test]
//     fn test_col_sums() {
//         let mat = CscMatrix {
//             nrow: 3,
//             ncol: 3,
//             x: vec![1.0, 4.0, 3.0, 2.0, 5.0],
//             i: vec![0, 2, 1, 0, 2],
//             p: vec![0, 2, 3, 5],
//         };
        
//         assert_eq!(mat.col_sum(0), 5.0);
//         assert_eq!(mat.col_sum(1), 3.0);
//         assert_eq!(mat.col_sum(2), 7.0);
//     }
// }


