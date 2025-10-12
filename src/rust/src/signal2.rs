use extendr_api::prelude::*;
use burn::prelude::*;
use burn::tensor::{Tensor, backend::{Backend, AutodiffBackend}, activation};
use burn::module::Param;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

// Don't import - they're defined in this same file
// use crate::signal2::{DeconvModel, DeconvConfig, train_deconvolution};

// Choose your backend - adjust based on what's available
// #[cfg(feature = "wgpu")]
// type MyBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

// #[cfg(not(feature = "wgpu"))]
// type MyBackend = burn::backend::Autodiff<burn::backend::NdArray>;

// Define all available backends as type aliases
type NdArrayBackend = burn::backend::Autodiff<burn::backend::NdArray>;
type WgpuBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
type CandleBackend = burn::backend::Autodiff<burn::backend::Candle>;

// Backend enum to choose at runtime
#[derive(Debug, Clone)]
enum BackendType {
    NdArray,
    Wgpu,
    Candle
}


// ============================================================================
// DECONVOLUTION MODEL DEFINITIONS
// ============================================================================

/// Configuration for deconvolution
#[derive(Debug, Clone)]
pub struct DeconvConfig {
    pub insert_size: f32,
    pub init_log_exposure: f32,
    pub learn_rate: f64,
    pub l1_lambda: f32,
    pub l2_lambda: f32,
    pub max_iterations: usize,
    pub poll_interval: usize,
    pub ll_tolerance: f32,
    pub sparsity_tolerance: f32,
}

impl Default for DeconvConfig {
    fn default() -> Self {
        Self {
            insert_size: 500.0,
            init_log_exposure: -5.0,      // Changed from -10 to -5 (starts at ~0.0067 instead of ~0.00005)
            learn_rate: 0.01,
            l1_lambda: 0.0,
            l2_lambda: 0.0,
            max_iterations: 10_000_000,
            poll_interval: 100,
            ll_tolerance: 1e-6,
            sparsity_tolerance: 1e-4,
        }
    }
}

/// Deconvolution model structure
#[derive(Module, Debug)]
pub struct DeconvModel<B: Backend> {
    // Learnable parameters
    log_exposures: Param<Tensor<B, 2>>,  // [n_sigs, n_samples]
    intercept: Param<Tensor<B, 2>>,      // [1, n_samples]
    
    // Fixed data (not learnable) - these are NOT in the Module derive
    #[module(skip)]
    signatures: Tensor<B, 2>,             // [n_genes, n_sigs]
    #[module(skip)]
    gene_lengths: Tensor<B, 2>,           // [n_genes, n_samples]
    #[module(skip)]
    observed_counts: Tensor<B, 2>,        // [n_genes, n_samples]
    #[module(skip)]
    gene_weights: Tensor<B, 1>,           // [n_genes]
}

impl<B: Backend> DeconvModel<B> {
    /// Create a new deconvolution model
    pub fn new(
        signatures: Tensor<B, 2>,
        gene_lengths: Tensor<B, 2>,
        observed_counts: Tensor<B, 2>,
        gene_weights: Tensor<B, 1>,
        config: &DeconvConfig,
        device: &B::Device,
    ) -> Self {
        let [_n_genes, n_sigs] = signatures.dims();
        let [_, n_samples] = observed_counts.dims();
        
        // Initialize log exposures with small random values instead of uniform
        // This breaks symmetry and allows different cell types to learn different values
        let log_exposures = Tensor::random(
            [n_sigs, n_samples],
            burn::tensor::Distribution::Normal(config.init_log_exposure as f64, 0.1),
            device
        );
        
        // Initialize intercept to zero
        let intercept = Tensor::zeros([1, n_samples], device);
        
        Self {
            log_exposures: Param::from_tensor(log_exposures),
            intercept: Param::from_tensor(intercept),
            signatures,
            gene_lengths,
            observed_counts,
            gene_weights,
        }
    }
    
    /// Forward pass: compute predicted counts
    pub fn forward(&self, insert_size: f32) -> Tensor<B, 2> {
        // Get positive exposures: E = exp(log_exposures)
        let exposures = self.log_exposures.val().exp();
        let intercept_exp = self.intercept.val().exp();
        
        // Concatenate exposures with intercept
        let all_exposures = Tensor::cat(vec![exposures, intercept_exp], 0);
        
        // Add intercept column to signatures (uniform 1/n_genes)
        let [n_genes, _] = self.signatures.dims();
        let intercept_sig = Tensor::ones([n_genes, 1], &self.signatures.device())
            .div_scalar(n_genes as f32);
        let signatures_with_int = Tensor::cat(
            vec![self.signatures.clone(), intercept_sig],
            1
        );
        
        // Predicted molecules: q = S * E
        let pred_mols = signatures_with_int.matmul(all_exposures);
        
        // Convert to reads: y = q * (gene_length / insert_size)
        let conversion = self.gene_lengths.clone()
            .div_scalar(insert_size);
        let pred_counts = pred_mols.mul(conversion);
        
        pred_counts
    }
    
    /// Compute Poisson negative log-likelihood
    pub fn poisson_nll(&self, pred_counts: Tensor<B, 2>) -> Tensor<B, 2> {
        let k = self.observed_counts.clone();
        
        // Variable part: k*log(y) - y
        let log_pred = pred_counts.clone().log();
        let var_part = k.clone().mul(log_pred).sub(pred_counts);
        
        // Constant part: log(k!) using Stirling's approximation
        let k_safe = k.clone().add_scalar(1e-10); // avoid log(0)
        let const_part = k.clone().mul(k_safe.log()).sub(k.clone());
        
        // NLL per gene per sample
        let nll_matrix = const_part.sub(var_part);
        
        // Weight by gene weights
        let [n_genes, n_samples] = nll_matrix.dims();
        let weights_expanded = self.gene_weights.clone()
            .reshape([n_genes, 1])
            .repeat(&[1, n_samples]);
        
        nll_matrix.mul(weights_expanded)
    }
    
    /// Compute L1 regularization penalty
    pub fn l1_penalty(&self, sample_nulls: &Tensor<B, 1>, l1_lambda: f32) -> Tensor<B, 1> {
        if l1_lambda == 0.0 {
            return Tensor::zeros([1], &self.log_exposures.device());
        }
        
        // Get all exposures (including intercept)
        let exposures = self.log_exposures.val().exp();
        let intercept_exp = self.intercept.val().exp();
        let all_exp = Tensor::cat(vec![exposures, intercept_exp], 0);
        
        // Scale penalty by null likelihood per sample
        let penalty_scale = sample_nulls.clone();
        let [_, n_samples] = all_exp.dims();
        let scale_expanded = penalty_scale.reshape([1, n_samples])
            .repeat(&[all_exp.dims()[0], 1]);
        
        // L1 penalty: λ * sum(scale * exposures)
        let scaled_exp = all_exp.mul(scale_expanded);
        let penalty = scaled_exp.sum();
        
        penalty.mul_scalar(l1_lambda).reshape([1])
    }
    
    /// Compute L2 regularization penalty (excludes intercept)
    pub fn l2_penalty(&self, sample_nulls: &Tensor<B, 1>, l2_lambda: f32) -> Tensor<B, 1> {
        if l2_lambda == 0.0 {
            return Tensor::zeros([1], &self.log_exposures.device());
        }
        
        // Only penalize non-intercept exposures
        let exposures = self.log_exposures.val().exp();
        let exp_squared = exposures.powf_scalar(2.0);
        
        // Scale penalty
        let penalty_scale = sample_nulls.clone();
        let [_, n_samples] = exp_squared.dims();
        let scale_expanded = penalty_scale.reshape([1, n_samples])
            .repeat(&[exp_squared.dims()[0], 1]);
        
        let scaled_sq = exp_squared.mul(scale_expanded);
        let penalty = scaled_sq.sum();
        
        penalty.mul_scalar(l2_lambda).reshape([1])
    }
    
    /// Compute total loss
    pub fn loss(&self, sample_nulls: &Tensor<B, 1>, config: &DeconvConfig) -> Tensor<B, 1> {
        let pred_counts = self.forward(config.insert_size);
        
        // Compute NLL per gene per sample
        let nll_matrix = self.poisson_nll(pred_counts);
        
        // Sum over genes to get per-sample NLL, then sum all samples
        let nll = nll_matrix.sum();
        
        let l1 = self.l1_penalty(sample_nulls, config.l1_lambda);
        let l2 = self.l2_penalty(sample_nulls, config.l2_lambda);
        
        nll.add(l1).add(l2)
    }
    
    /// Count non-zero exposures (for sparsity tracking)
    pub fn count_nonzero(&self) -> f32 {
        // Use sigmoid-like function to approximate step function
        let exposures = self.log_exposures.val();
        let intercept = self.intercept.val();
        let all_exp = Tensor::cat(vec![exposures, intercept], 0);
        
        // Use activation sigmoid
        let nonzero_approx = activation::sigmoid(all_exp).sum();
        
        // Convert to scalar - need to extract the element
        nonzero_approx.into_scalar().elem::<f32>()
    }
    
    /// Get normalized exposures (for output)
    pub fn get_exposures(&self) -> Tensor<B, 2> {
        let exposures = self.log_exposures.val().exp();
        let intercept = self.intercept.val().exp();
        
        Tensor::cat(vec![exposures, intercept], 0)
    }
}

/// Training function
pub fn train_deconvolution<B: AutodiffBackend>(
    mut model: DeconvModel<B>,
    config: DeconvConfig,
    device: &B::Device,
) -> DeconvModel<B> {
    // First fit NULL model (intercept only) to get baseline
    rprintln!("Fitting NULL model...");
    let sample_nulls = fit_null_model(&mut model, &config, device);
    
    rprintln!("NULL model complete. Mean null NLL per sample: {:.2}", 
              sample_nulls.clone().mean().into_scalar().elem::<f32>());
    
    // IMPORTANT: Re-initialize the log_exposures after NULL fitting
    // The NULL model set them to -100, we need to reset them
    let [n_sigs, n_samples] = model.log_exposures.dims();
    let log_exposures = Tensor::random(
        [n_sigs, n_samples],
        burn::tensor::Distribution::Normal(config.init_log_exposure as f64, 0.1),
        device
    );
    model.log_exposures = Param::from_tensor(log_exposures);
    
    rprintln!("Re-initialized exposures for full model training");
    
    // Now fit full model
    rprintln!("Fitting full model...");
    
    let mut optim = AdamConfig::new()
        .init();
    
    let mut last_loss = f32::INFINITY;
    let mut last_nonzero = 0.0;
    
    for iter in 0..config.max_iterations {
        // Compute loss and gradients
        let loss = model.loss(&sample_nulls, &config);
        let grads = loss.backward();
        
        // Update parameters
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.learn_rate, model, grads);
        
        // Check progress every poll_interval iterations
        if iter % config.poll_interval == 0 {
            let current_loss = loss.clone().into_scalar().elem::<f32>();
            let current_nonzero = model.count_nonzero();
            
            let loss_diff = last_loss - current_loss;
            let nz_diff = (last_nonzero - current_nonzero).abs();
            
            // Show some actual exposure values for debugging
            if iter % 1000 == 0 {
                let exp = model.get_exposures();
                let exp_mean = exp.clone().mean().into_scalar().elem::<f32>();
                let exp_max = exp.clone().max().into_scalar().elem::<f32>();
                let exp_min = exp.clone().min().into_scalar().elem::<f32>();
                
                // Check variance across cell types
                let exp_first_sample = exp.clone().slice([0..exp.dims()[0], 0..1]);
                let exp_std = exp_first_sample.clone().var(0).sqrt().into_scalar().elem::<f32>();
                
                rprintln!(
                    "Iter {}: loss={:.6}, nonzero={:.1}, Δloss={:.6}, exp(min/mean/max)=({:.2}/{:.2}/{:.2}), std={:.2}",
                    iter, current_loss, current_nonzero, loss_diff, exp_min, exp_mean, exp_max, exp_std
                );
            } else {
                rprintln!(
                    "Iter {}: loss={:.6}, nonzero={:.1}, Δloss={:.6}, Δnz={:.2}",
                    iter, current_loss, current_nonzero, loss_diff, nz_diff
                );
            }
            
            // Check convergence - but not too early
            if iter > 1000 && loss_diff > 0.0 
                && loss_diff / last_loss.abs() < config.ll_tolerance
                && nz_diff / current_nonzero.max(1.0) < config.sparsity_tolerance
            {
                rprintln!("Converged at iteration {}", iter);
                break;
            }
            
            last_loss = current_loss;
            last_nonzero = current_nonzero;
        }
    }
    
    model
}

/// Fit NULL model (intercept only)
fn fit_null_model<B: AutodiffBackend>(
    model: &mut DeconvModel<B>,
    config: &DeconvConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    // Freeze log_exposures at very negative values
    let [n_sigs, n_samples] = model.log_exposures.dims();
    let frozen_exp = Tensor::zeros([n_sigs, n_samples], device)
        .sub_scalar(100.0);  // Very negative so exp() ≈ 0
    model.log_exposures = Param::from_tensor(frozen_exp);
    
    // Only optimize intercept
    let mut optim = AdamConfig::new()
        .init();
    
    let dummy_nulls = Tensor::zeros([n_samples], device);
    
    let mut last_loss = f32::INFINITY;
    
    for iter in 0..config.max_iterations.min(10000) {  // Cap null fitting at 10k iterations
        let loss = model.loss(&dummy_nulls, config);
        let grads = loss.backward();
        
        // Only update intercept
        let grads = GradientsParams::from_grads(grads, model);
        *model = optim.step(config.learn_rate, model.clone(), grads);
        
        if iter % config.poll_interval == 0 {
            let current_loss = loss.clone().into_scalar().elem::<f32>();
            
            if iter % 1000 == 0 {
                rprintln!("NULL iter {}: loss={:.6}", iter, current_loss);
            }
            
            // Check for convergence
            let loss_change = (last_loss - current_loss).abs();
            if iter > 100 && loss_change / current_loss < 1e-8 {
                rprintln!("NULL model converged at iteration {}", iter);
                break;
            }
            
            last_loss = current_loss;
        }
    }
    
    // Return per-sample null NLL for regularization scaling
    let pred = model.forward(config.insert_size);
    let nll_matrix = model.poisson_nll(pred);
    nll_matrix.sum_dim(0).squeeze(0) // Sum over genes and squeeze to 1D
}

// ============================================================================
// R INTERFACE FUNCTIONS
// ============================================================================

/// Convert R matrix to Burn Tensor
fn rmat_to_tensor<B: Backend>(
    robj: Robj,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    // Get dimensions - returns Vec<Rint> which are i32
    let dims = robj.dim()
        .ok_or_else(|| Error::Other("Matrix must have dimensions".into()))?;
    
    if dims.len() != 2 {
        return Err(Error::Other("Expected 2D matrix".into()));
    }
    
    // Convert Rint to usize
    let nrow = dims[0].inner() as usize;
    let ncol = dims[1].inner() as usize;
    
    rprintln!("▶ rmat_to_tensor: target shape = [{}, {}]", nrow, ncol);
    
    // Get data as f64 vector, then convert to f32
    let data: Vec<f32> = robj
        .as_real_vector()
        .ok_or_else(|| Error::Other("Matrix must be numeric".into()))?
        .iter()
        .map(|&x| x as f32)
        .collect();
    
    rprintln!("  data length = {}", data.len());
    
    // Verify data length matches dimensions
    if data.len() != nrow * ncol {
        return Err(Error::Other(format!(
            "Data length {} doesn't match dimensions {} × {} = {}",
            data.len(), nrow, ncol, nrow * ncol
        ).into()));
    }
    
    // R stores matrices in column-major order, Burn uses row-major
    // So we need to transpose
    let mut row_major = vec![0.0f32; data.len()];
    for i in 0..nrow {
        for j in 0..ncol {
            row_major[i * ncol + j] = data[j * nrow + i];
        }
    }
    
    // Create tensor using the working approach:
    // 1) from_floats → rank-1
    let t1 = Tensor::<B, 1>::from_floats(row_major.as_slice(), device);
    // 2) bump to rank-2 via unsqueeze
    let t2 = t1.unsqueeze::<2>();
    // 3) reshape into [nrow, ncol]
    let t3 = t2.reshape([nrow, ncol]);
    
    Ok(t3)
}

/// Convert R vector to Burn Tensor (1D)
fn rvec_to_tensor<B: Backend>(
    robj: Robj,
    device: &B::Device,
) -> Result<Tensor<B, 1>> {
    let data: Vec<f32> = robj
        .as_real_vector()
        .ok_or_else(|| Error::Other("Vector must be numeric".into()))?
        .iter()
        .map(|&x| x as f32)
        .collect();
    
    // For 1D vectors, from_floats works directly
    Ok(Tensor::<B, 1>::from_floats(data.as_slice(), device))
}

/// Extract scalar f64 from R object
fn robj_to_f64(robj: Robj, default: f64, name: &str) -> f64 {
    robj.as_real_vector()
        .and_then(|v| v.first().copied())
        .unwrap_or_else(|| {
            rprintln!("Warning: Using default value {} for {}", default, name);
            default
        })
}

/// Extract scalar usize from R object
fn robj_to_usize(robj: Robj, default: usize, name: &str) -> usize {
    robj.as_real_vector()
        .and_then(|v| v.first().copied())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            rprintln!("Warning: Using default value {} for {}", default, name);
            default
        })
}

/// Fit deconvolution model
/// 
/// @param sigs Matrix of cellular signatures (genes × cell_types), normalized to sum to 1 per column
/// @param bulk Matrix of bulk RNA-seq counts (genes × samples)
/// @param gene_lengths Vector of gene lengths (same length as nrow(sigs))
/// @param w_vec Vector of gene weights (same length as nrow(sigs)), values in [0,1]
/// @param insert_size Insert size for fragment length conversion (default 500)
/// @param init_log_exp Initial value for log exposures (default -10)
/// @param lr Learning rate (default 0.01)
/// @param l1_lambda L1 regularization lambda (default 0.0)
/// @param l2_lambda L2 regularization lambda (default 0.0)
/// @param max_iter Maximum iterations (default 10000)
/// @param poll_interval Check convergence every N iterations (default 100)
/// @param ll_tol Log-likelihood convergence tolerance (default 1e-6)
/// @param sparsity_tol Sparsity convergence tolerance (default 1e-4)
/// 
/// @return List with:
///   - exposures: Matrix of fitted exposures (cell_types+intercept × samples)
///   - pred_counts: Matrix of predicted counts (genes × samples)
///   - loss_history: Vector of loss values during training
pub fn fit_deconv(
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
) -> Result<List> {
    // Parse backend choice
    let backend_str = backend
        .as_str()
        .unwrap_or("ndarray")
        .to_lowercase();
    
    let backend_type = match backend_str.as_str() {
        "ndarray" | "cpu" => BackendType::NdArray,
        // #[cfg(feature = "wgpu")]
        "wgpu" | "gpu" => BackendType::Wgpu,
        // #[cfg(feature = "cuda")]
        "candle" => BackendType::Candle,
        _ => {
            rprintln!("Unknown backend '{}', using ndarray (CPU)", backend_str);
            BackendType::NdArray
        }
    };
    
    rprintln!("Using backend: {:?}", backend_type);
    
    // Dispatch to appropriate backend
    match backend_type {
        BackendType::Wgpu => fit_deconv_impl::<WgpuBackend>(
            sigs, bulk, gene_lengths, w_vec,
            insert_size, init_log_exp, lr, l1_lambda, l2_lambda,
            max_iter, poll_interval, ll_tol, sparsity_tol
        ),
        BackendType::Candle => fit_deconv_impl::<CandleBackend>(
            sigs, bulk, gene_lengths, w_vec,
            insert_size, init_log_exp, lr, l1_lambda, l2_lambda,
            max_iter, poll_interval, ll_tol, sparsity_tol
        ),
        BackendType::NdArray => fit_deconv_impl::<NdArrayBackend>(
            sigs, bulk, gene_lengths, w_vec,
            insert_size, init_log_exp, lr, l1_lambda, l2_lambda,
            max_iter, poll_interval, ll_tol, sparsity_tol
        ),
        // _ => fit_deconv_impl::<NdArrayBackend>(
        //     sigs, bulk, gene_lengths, w_vec,
        //     insert_size, init_log_exp, lr, l1_lambda, l2_lambda,
        //     max_iter, poll_interval, ll_tol, sparsity_tol
        // ),
    }
}


/// Implementation that works with any backend
fn fit_deconv_impl<B: AutodiffBackend>(
    sigs: Robj,
    bulk: Robj,
    gene_lengths: Robj,
    w_vec: Robj,
    insert_size: Robj,
    init_log_exp: Robj,
    lr: Robj,
    l1_lambda: Robj,
    l2_lambda: Robj,
    max_iter: Robj,
    poll_interval: Robj,
    ll_tol: Robj,
    sparsity_tol: Robj,
) -> Result<List> {
    // Create device
    let device = <B as Backend>::Device::default();
    
    // Convert inputs to tensors
    let signatures: Tensor<B, 2> = rmat_to_tensor(sigs, &device)?;
    let observed_counts: Tensor<B, 2> = rmat_to_tensor(bulk, &device)?;
    let gene_lengths_vec: Tensor<B, 1> = rvec_to_tensor(gene_lengths, &device)?;
    let gene_weights: Tensor<B, 1> = rvec_to_tensor(w_vec, &device)?;
    
    // Get dimensions
    let [n_genes, n_sigs] = signatures.dims();
    let [bulk_genes, n_samples] = observed_counts.dims();
    
    // Validate dimensions
    if n_genes != bulk_genes {
        return Err(Error::Other(format!(
            "Gene dimension mismatch: sigs has {} genes, bulk has {}",
            n_genes, bulk_genes
        ).into()));
    }
    
    if gene_lengths_vec.dims()[0] != n_genes {
        return Err(Error::Other(format!(
            "gene_lengths must have length {} (nrow of sigs)",
            n_genes
        ).into()));
    }
    
    if gene_weights.dims()[0] != n_genes {
        return Err(Error::Other(format!(
            "w_vec must have length {} (nrow of sigs)",
            n_genes
        ).into()));
    }
    
    rprintln!("Gene lengths length: {:?}", gene_lengths_vec.dims());
    rprintln!("Gene weights length: {:?}", gene_weights.dims());
    
    // Expand gene_lengths to match bulk samples dimension
    // gene_lengths_vec is [n_genes], need [n_genes, n_samples]
    let gene_lengths_matrix = gene_lengths_vec
        .clone()
        .reshape([n_genes, 1])           // [n_genes, 1]
        .repeat(&[1, n_samples]);        // [n_genes, n_samples]
    
    rprintln!("Gene lengths matrix shape: {:?}", gene_lengths_matrix.dims());
    
    // Extract scalar parameters
    let config = DeconvConfig {
        insert_size: robj_to_f64(insert_size, 500.0, "insert_size") as f32,
        init_log_exposure: robj_to_f64(init_log_exp, -10.0, "init_log_exp") as f32,
        learn_rate: robj_to_f64(lr, 0.01, "lr"),
        l1_lambda: robj_to_f64(l1_lambda, 0.0, "l1_lambda") as f32,
        l2_lambda: robj_to_f64(l2_lambda, 0.0, "l2_lambda") as f32,
        max_iterations: robj_to_usize(max_iter, 10000, "max_iter"),
        poll_interval: robj_to_usize(poll_interval, 100, "poll_interval"),
        ll_tolerance: robj_to_f64(ll_tol, 1e-6, "ll_tol") as f32,
        sparsity_tolerance: robj_to_f64(sparsity_tol, 1e-4, "sparsity_tol") as f32,
    };
    
    rprintln!("Fitting deconvolution model:");
    rprintln!("  Genes: {}", n_genes);
    rprintln!("  Cell types: {}", n_sigs);
    rprintln!("  Samples: {}", n_samples);
    rprintln!("  Insert size: {}", config.insert_size);
    rprintln!("  Learning rate: {}", config.learn_rate);
    rprintln!("  L1 lambda: {}", config.l1_lambda);
    rprintln!("  L2 lambda: {}", config.l2_lambda);
    rprintln!("  Max iterations: {}", config.max_iterations);
    
    // Check that signatures are normalized
    let sig_sums = signatures.clone().sum_dim(0);
    let sig_sum_mean = sig_sums.clone().mean().into_scalar().elem::<f32>();
    let sig_sum_min = sig_sums.clone().min().into_scalar().elem::<f32>();
    let sig_sum_max = sig_sums.clone().max().into_scalar().elem::<f32>();
    rprintln!("  Signature column sums: mean={:.4}, min={:.4}, max={:.4}", 
              sig_sum_mean, sig_sum_min, sig_sum_max);
    
    if sig_sum_min < 0.95 || sig_sum_max > 1.05 {
        rprintln!("  WARNING: Signatures may not be properly normalized!");
    }
    
    // Check observed counts
    let obs_mean = observed_counts.clone().mean().into_scalar().elem::<f32>();
    let obs_max = observed_counts.clone().max().into_scalar().elem::<f32>();
    rprintln!("  Observed counts: mean={:.2}, max={:.2}", obs_mean, obs_max);
    
    // Create model
    let model = DeconvModel::new(
        signatures,
        gene_lengths_matrix,
        observed_counts,
        gene_weights,
        &config,
        &device,
    );
    
    // Train
    let trained_model = train_deconvolution(model, config.clone(), &device);
    
    // Get results
    let exposures = trained_model.get_exposures();
    let pred_counts = trained_model.forward(config.insert_size);
    
    // Convert exposures to R matrix
    let exp_data: Vec<f32> = exposures
        .to_data()
        .to_vec()
        .unwrap_or_else(|_| {
            panic!("Failed to convert exposures tensor")
        });
    
    let exp_dims = exposures.dims();
    
    // Convert from row-major (Burn) to column-major (R)
    let mut exp_col_major = vec![0.0f64; exp_data.len()];
    for i in 0..exp_dims[0] {
        for j in 0..exp_dims[1] {
            exp_col_major[j * exp_dims[0] + i] = exp_data[i * exp_dims[1] + j] as f64;
        }
    }
    
    let robj_exp = Robj::from(exp_col_major);
    let _ = robj_exp.set_attrib(
        "dim",
        Robj::from(vec![exp_dims[0] as i32, exp_dims[1] as i32])
    );
    
    // Convert predicted counts to R matrix
    let pred_data: Vec<f32> = pred_counts
        .to_data()
        .to_vec()
        .unwrap_or_else(|_| {
            panic!("Failed to convert pred_counts tensor")
        });
    
    let pred_dims = pred_counts.dims();
    
    // Convert from row-major to column-major
    let mut pred_col_major = vec![0.0f64; pred_data.len()];
    for i in 0..pred_dims[0] {
        for j in 0..pred_dims[1] {
            pred_col_major[j * pred_dims[0] + i] = pred_data[i * pred_dims[1] + j] as f64;
        }
    }
    
    let robj_pred = Robj::from(pred_col_major);
    let _ = robj_pred.set_attrib(
        "dim",
        Robj::from(vec![pred_dims[0] as i32, pred_dims[1] as i32])
    );
    
    rprintln!("Training complete!");
    rprintln!("Exposures matrix: {} x {} (cell_types+intercept × samples)", 
              exp_dims[0], exp_dims[1]);
    
    Ok(list!(
        exposures = robj_exp,
        pred_counts = robj_pred
    ))
}
