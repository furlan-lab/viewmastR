//! Sparse matrix training module
//!
//! Converts sparse matrices to SCItemRaw vectors in Rust (not R),
//! then uses existing training infrastructure.
//!
//! This approach:
//! - Eliminates R-side dense conversion overhead
//! - Eliminates R-side list creation (100k+ small objects)
//! - Reuses all existing training code (no Burn API changes needed)

use crate::common::{RExport, SCItemRaw};
use crate::sparse::CscMatrix;

/// Convert sparse matrix + labels + size factors to Vec<SCItemRaw>
/// This happens once at the start of training, in Rust instead of R
pub fn sparse_to_items(
    mat: &CscMatrix,
    labels: &[usize],
    size_factors: &[f64],
) -> Vec<SCItemRaw> {
    assert_eq!(mat.ncol, labels.len(), 
        "Number of columns ({}) must match number of labels ({})", mat.ncol, labels.len());
    assert_eq!(mat.ncol, size_factors.len(),
        "Number of columns ({}) must match number of size factors ({})", mat.ncol, size_factors.len());
    
    (0..mat.ncol)
        .map(|col| {
            let data: Vec<f64> = mat.col_normalized_f32(col, size_factors[col])
                .into_iter()
                .map(|x| x as f64)
                .collect();
            SCItemRaw {
                data,
                target: labels[col] as i32,
            }
        })
        .collect()
}

/// Convert sparse matrix to Vec<SCItemRaw> for query (no labels)
pub fn sparse_to_items_query(
    mat: &CscMatrix,
    size_factors: &[f64],
) -> Vec<SCItemRaw> {
    assert_eq!(mat.ncol, size_factors.len(),
        "Number of columns ({}) must match number of size factors ({})", mat.ncol, size_factors.len());
    
    (0..mat.ncol)
        .map(|col| {
            let data: Vec<f64> = mat.col_normalized_f32(col, size_factors[col])
                .into_iter()
                .map(|x| x as f64)
                .collect();
            SCItemRaw {
                data,
                target: 0, // dummy value for query
            }
        })
        .collect()
}

// ============================================================================
// MLR Sparse Training - delegates to existing run_custom_* functions
// ============================================================================

pub fn run_mlr_sparse_nd(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_mlr::run_custom_nd;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (NdArray backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting MLR training...");
    }
    
    run_custom_nd(train, test, query, num_classes, learning_rate, num_epochs, directory, verbose)
}

pub fn run_mlr_sparse_wgpu(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_mlr::run_custom_wgpu;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (WGPU backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting MLR training...");
    }
    
    run_custom_wgpu(train, test, query, num_classes, learning_rate, num_epochs, directory, verbose)
}

pub fn run_mlr_sparse_candle(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_mlr::run_custom_candle;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (Candle backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting MLR training...");
    }
    
    run_custom_candle(train, test, query, num_classes, learning_rate, num_epochs, directory, verbose)
}

// ============================================================================
// ANN Sparse Training (1 hidden layer)
// ============================================================================

pub fn run_ann_sparse_nd(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    hidden_size: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_ann::run_custom_nd;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (NdArray backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
        eprintln!("  Hidden size: {}", hidden_size);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting ANN training...");
    }
    
    run_custom_nd(train, test, query, num_classes, hidden_size, learning_rate, num_epochs, directory, verbose)
}

pub fn run_ann_sparse_wgpu(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    hidden_size: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_ann::run_custom_wgpu;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (WGPU backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
        eprintln!("  Hidden size: {}", hidden_size);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting ANN training...");
    }
    
    run_custom_wgpu(train, test, query, num_classes, hidden_size, learning_rate, num_epochs, directory, verbose)
}

pub fn run_ann_sparse_candle(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    hidden_size: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_ann::run_custom_candle;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (Candle backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
        eprintln!("  Hidden size: {}", hidden_size);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting ANN training...");
    }
    
    run_custom_candle(train, test, query, num_classes, hidden_size, learning_rate, num_epochs, directory, verbose)
}

// ============================================================================
// ANN2 Sparse Training (2 hidden layers)
// ============================================================================

pub fn run_ann2_sparse_nd(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    hidden1: usize,
    hidden2: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_ann2l::run_custom_nd;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (NdArray backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
        eprintln!("  Hidden layers: {} -> {}", hidden1, hidden2);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting ANN2 training...");
    }
    
    run_custom_nd(train, test, query, num_classes, hidden1, hidden2, learning_rate, num_epochs, directory, verbose)
}

pub fn run_ann2_sparse_wgpu(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    hidden1: usize,
    hidden2: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_ann2l::run_custom_wgpu;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (WGPU backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
        eprintln!("  Hidden layers: {} -> {}", hidden1, hidden2);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting ANN2 training...");
    }
    
    run_custom_wgpu(train, test, query, num_classes, hidden1, hidden2, learning_rate, num_epochs, directory, verbose)
}

pub fn run_ann2_sparse_candle(
    train_mat: &CscMatrix,
    train_labels: &[usize],
    train_sf: &[f64],
    test_mat: &CscMatrix,
    test_labels: &[usize],
    test_sf: &[f64],
    query_data: Option<&(CscMatrix, Vec<f64>)>,
    num_classes: usize,
    hidden1: usize,
    hidden2: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> RExport {
    use crate::scrna_ann2l::run_custom_candle;
    
    if verbose {
        eprintln!("Converting sparse matrices to training items (Candle backend)...");
        eprintln!("  Train: {} cells x {} features", train_mat.ncol, train_mat.nrow);
        eprintln!("  Test: {} cells x {} features", test_mat.ncol, test_mat.nrow);
        eprintln!("  Hidden layers: {} -> {}", hidden1, hidden2);
    }
    
    let train = sparse_to_items(train_mat, train_labels, train_sf);
    let test = sparse_to_items(test_mat, test_labels, test_sf);
    let query = query_data.map(|(qmat, qsf)| {
        if verbose {
            eprintln!("  Query: {} cells x {} features", qmat.ncol, qmat.nrow);
        }
        sparse_to_items_query(qmat, qsf)
    });
    
    if verbose {
        eprintln!("Starting ANN2 training...");
    }
    
    run_custom_candle(train, test, query, num_classes, hidden1, hidden2, learning_rate, num_epochs, directory, verbose)
}
