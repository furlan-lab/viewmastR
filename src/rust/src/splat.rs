use extendr_api::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;
use rand::SeedableRng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};


use crate::pb::ProgressBar;

/// Convert Robj to Vec<f64>
fn robj_to_vec_f64(robj: Robj, _name: &str) -> Result<Vec<f64>> {
    robj.as_real_vector()
        .ok_or_else(|| Error::Other(format!("{} must be a numeric vector", _name)))
}

/// Convert Robj to Vec<String>
fn robj_to_vec_string(robj: Robj, _name: &str) -> Result<Vec<String>> {
    robj.as_str_vector()
        .ok_or_else(|| Error::Other(format!("{} must be a character vector", _name)))
        .map(|v| v.into_iter().map(|s| s.to_string()).collect())
}

/// Convert Robj to usize with default
fn robj_to_usize(robj: Robj, default: usize, _name: &str) -> usize {
    robj.as_integer()
        .map(|i| i.max(0) as usize)
        .unwrap_or(default)
}

/// Convert Robj to f64 with default
fn robj_to_f64(robj: Robj, default: f64, _name: &str) -> f64 {
    robj.as_real().unwrap_or(default)
}

/// Convert Robj to bool with default
fn robj_to_bool(robj: Robj, default: bool, _name: &str) -> bool {
    robj.as_bool().unwrap_or(default)
}

/// Convert R matrix to Vec<Vec<usize>> (column-major to row-major for genes x bulk)
fn rmat_to_counts(robj: Robj) -> Result<Vec<Vec<usize>>> {
    // Get matrix dimensions from dim attribute
    let dims = robj.dim()
        .ok_or_else(|| Error::Other("counts_matrix must be a matrix".to_string()))?;
    
    if dims.len() != 2 {
        return Err(Error::Other("counts_matrix must be 2-dimensional".to_string()));
    }
    
    let nrow = dims[0].inner() as usize;
    let ncol = dims[1].inner() as usize;
    
    let mut result = vec![vec![0usize; ncol]; nrow];
    
    // Try to get as integer vector first, then as real vector
    if let Some(data) = robj.as_integer_vector() {
        // Integer matrix
        for i in 0..nrow {
            for j in 0..ncol {
                let idx = j * nrow + i; // column-major indexing
                result[i][j] = data[idx].max(0) as usize;
            }
        }
    } else if let Some(data) = robj.as_real_vector() {
        // Real (double) matrix - convert to integers
        for i in 0..nrow {
            for j in 0..ncol {
                let idx = j * nrow + i; // column-major indexing
                result[i][j] = data[idx].max(0.0).round() as usize;
            }
        }
    } else {
        return Err(Error::Other("counts_matrix must be numeric (integer or real)".to_string()));
    }
    
    Ok(result)
}

/// Sample sizes with kernel density estimation
fn sample_sizes_with_kde(
    sizes: &[f64],
    bandwidth: f64,
    n: usize,
    min_size: f64,
    max_size: f64,
    rng: &mut StdRng,
) -> Vec<usize> {
    let mut result = Vec::with_capacity(n);
    
    while result.len() < n {
        let base_size: f64 = sizes[rng.random_range(0..sizes.len())];
        // Manual gaussian noise generation using Box-Muller transform
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let noisy_size: f64 = base_size + z * bandwidth;
        
        if noisy_size > min_size && noisy_size < max_size {
            result.push(noisy_size.round() as usize);
        }
    }
    
    result
}

/// Create a single simulated cell
fn create_simulated_cell(
    gene_pool: &[usize], // indices into universe
    target_size: usize,
    total_counts: usize,
    replace: bool,
    rng: &mut StdRng,
) -> HashMap<usize, usize> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    
    if target_size > total_counts {
        // Must sample with replacement
        for _ in 0..target_size {
            let gene_idx = gene_pool[rng.random_range(0..gene_pool.len())];
            *counts.entry(gene_idx).or_insert(0) += 1;
        }
    } else {
        if replace {
            // Sample with replacement
            for _ in 0..target_size {
                let gene_idx = gene_pool[rng.random_range(0..gene_pool.len())];
                *counts.entry(gene_idx).or_insert(0) += 1;
            }
        } else {
            // Sample without replacement using Fisher-Yates
            let mut pool = gene_pool.to_vec();
            for i in 0..target_size.min(pool.len()) {
                let j = rng.random_range(i..pool.len());
                pool.swap(i, j);
                *counts.entry(pool[i]).or_insert(0) += 1;
            }
        }
    }
    
    counts
}

/// Process a single bulk sample to create N simulated cells
/// Returns triplet format (i, j, x) for sparse matrix
fn process_bulk_sample(
    bulk_counts: &[usize],
    _n_genes: usize,
    sizes: &[f64],
    bandwidth: f64,
    min_size: f64,
    max_size: f64,
    n_cells: usize,
    replace_counts: bool,
    seed: u64,
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Sample target sizes for N cells
    let target_sizes = sample_sizes_with_kde(
        sizes,
        bandwidth,
        n_cells,
        min_size,
        max_size,
        &mut rng,
    );
    
    // Create gene pool (gene indices repeated by their counts)
    let total_counts: usize = bulk_counts.iter().sum();
    let mut gene_pool = Vec::with_capacity(total_counts);
    
    for (gene_idx, &count) in bulk_counts.iter().enumerate() {
        for _ in 0..count {
            gene_pool.push(gene_idx);
        }
    }
    
    // Generate triplets for sparse matrix
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    
    // Generate each simulated cell
    for (cell_idx, &target_size) in target_sizes.iter().enumerate() {
        let cell_counts = create_simulated_cell(
            &gene_pool,
            target_size,
            total_counts,
            replace_counts,
            &mut rng,
        );
        
        // Add to triplet vectors
        for (gene_idx, count) in cell_counts {
            rows.push(gene_idx as i32);
            cols.push(cell_idx as i32);
            vals.push(count as f64);
        }
    }
    
    (rows, cols, vals)
}

/// Main entry point: process all bulk samples in parallel
pub fn splat_bulk_reference_rust_core(
    counts_matrix: Robj,
    universe: Robj,
    sizes: Robj,
    bandwidth: Robj,
    n_cells_per_bulk: Robj,
    replace_counts: Robj,
    seed: Robj,
    verbose: Robj,
) -> Result<List> {
    // Parse inputs
    let counts = rmat_to_counts(counts_matrix)?;
    let universe_vec = robj_to_vec_string(universe, "universe")?;
    let sizes_vec = robj_to_vec_f64(sizes, "sizes")?;
    let bw = robj_to_f64(bandwidth, 1.0, "bandwidth");
    let n_cells = robj_to_usize(n_cells_per_bulk, 2, "n_cells_per_bulk");
    let replace = robj_to_bool(replace_counts, false, "replace_counts");
    let seed_val = robj_to_usize(seed, 42, "seed") as u64;
    let show_progress = robj_to_bool(verbose, true, "verbose");
    
    let n_genes = counts.len();
    let n_bulk = if n_genes > 0 { counts[0].len() } else { 0 };
    
    // Precompute min/max sizes
    let min_size = sizes_vec.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_size = sizes_vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Transpose counts: from genes x bulk to bulk x genes for easier iteration
    let mut bulk_samples: Vec<Vec<usize>> = vec![vec![0; n_genes]; n_bulk];
    for gene_idx in 0..n_genes {
        for bulk_idx in 0..n_bulk {
            bulk_samples[bulk_idx][gene_idx] = counts[gene_idx][bulk_idx];
        }
    }
    
    // Setup progress bar if verbose
    let progress = if show_progress {
        Some(Arc::new(Mutex::new(ProgressBar::cargo_style(n_bulk as u32, 50, true))))
    } else {
        None
    };
    
    // Process all bulk samples in parallel
    let results: Vec<(Vec<i32>, Vec<i32>, Vec<f64>)> = bulk_samples
        .par_iter()
        .enumerate()
        .map(|(bulk_idx, bulk_counts)| {
            let sample_seed = seed_val.wrapping_add(bulk_idx as u64);
            let result = process_bulk_sample(
                bulk_counts,
                n_genes,
                &sizes_vec,
                bw,
                min_size,
                max_size,
                n_cells,
                replace,
                sample_seed,
            );
            
            // Update progress bar
            if let Some(ref pb) = progress {
                if let Ok(mut bar) = pb.lock() {
                    bar.update();
                }
            }
            
            result
        })
        .collect();
    
    // Combine all triplets into a single sparse matrix
    // Each bulk sample produces n_cells columns, so total columns = n_bulk * n_cells
    let total_cols = n_bulk * n_cells;
    
    let mut combined_rows = Vec::new();
    let mut combined_cols = Vec::new();
    let mut combined_vals = Vec::new();
    
    // Offset columns for each bulk sample
    for (bulk_idx, (rows, cols, vals)) in results.into_iter().enumerate() {
        let col_offset = (bulk_idx * n_cells) as i32;
        
        for ((row, col), val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            combined_rows.push(*row);
            combined_cols.push(*col + col_offset);
            combined_vals.push(*val);
        }
    }
    
    // Return single combined triplet matrix with metadata for building dgCMatrix in R
    Ok(list!(
        i = combined_rows,
        j = combined_cols,
        x = combined_vals,
        nrow = n_genes as i32,
        ncol = total_cols as i32,
        universe = universe_vec,
        n_bulk = n_bulk as i32,
        n_cells_per_bulk = n_cells as i32
    ))
}

