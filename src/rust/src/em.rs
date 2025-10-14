// EM Algorithm with Intercept for Background/Noise
// The intercept models uniform background noise across all genes

use burn::prelude::*;
use burn::tensor::Tensor;
use extendr_api::prelude::*;  // Add this import

#[derive(Debug, Clone)]
pub struct EMConfig {
    pub max_iterations: usize,
    pub tolerance: f32,
    pub min_exposure: f32,
    pub verbose: bool,
    pub use_intercept: bool,
}

impl Default for EMConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            min_exposure: 1e-10,
            verbose: true,
            use_intercept: true,
        }
    }
}

#[derive(Debug)]
pub struct EMDeconvModel<B: Backend> {
    signatures: Tensor<B, 2>,      // [n_genes, n_celltypes]
    observed_counts: Tensor<B, 2>, // [n_genes, n_samples]
    exposures: Tensor<B, 2>,       // [n_celltypes, n_samples]
    intercept: Tensor<B, 2>,       // [1, n_samples] - NEW: noise level per sample
    #[allow(dead_code)]
    gene_lengths: Tensor<B, 2>,    // [n_genes, n_samples]
    gene_weights: Tensor<B, 1>,    // [n_genes]
    
    // NEW: Precomputed for efficiency
    #[allow(dead_code)]
    intercept_signature: Tensor<B, 2>,  // [n_genes, 1] - uniform across genes
    signatures_with_intercept: Tensor<B, 2>,  // [n_genes, n_celltypes + 1]
    use_intercept: bool,
    config_min_exposure: f32,  // Store config value in model
}

impl<B: Backend> EMDeconvModel<B> {
    pub fn new(
        signatures: Tensor<B, 2>,
        observed_counts: Tensor<B, 2>,
        gene_lengths: Tensor<B, 2>,
        gene_weights: Tensor<B, 1>,
        use_intercept: bool,
        min_exposure: f32,
        device: &B::Device,
    ) -> Self {
        let [n_genes, n_celltypes] = signatures.dims();
        let [_, _n_samples] = observed_counts.dims();

        // eprintln!("DEBUG: n_genes_sig={}, n_celltypes={}, n_genes_counts={}, n_samples={}", n_genes, n_celltypes, n_genes_counts, n_samples);
        // eprintln!("DEBUG: observed_counts shape={:?}", observed_counts.dims());
    
        // Initialize exposures uniformly
        let total_counts = observed_counts.clone().sum_dim(0);
        // eprintln!("DEBUG: total_counts shape={:?}", total_counts.dims());
        let init_exposures = total_counts
            .clone()  // Clone here to reuse below
            // .reshape([1, n_samples])
            .repeat(&[n_celltypes, 1])
            .div_scalar(n_celltypes as f32);
        
        // eprintln!("DEBUG: init_exposures shape={:?}", init_exposures.dims());
        // Initialize intercept (small uniform background)
        let init_intercept = total_counts
            .mul_scalar(0.05);
            // .reshape([1, n_samples]);

        // Create intercept "signature" - uniform across all genes
        let intercept_signature = Tensor::ones([n_genes, 1], device)
            .div_scalar(n_genes as f32);
        
        // Augmented signature matrix with intercept column
        let signatures_with_intercept = if use_intercept {
            Tensor::cat(vec![signatures.clone(), intercept_signature.clone()], 1)
        } else {
            signatures.clone()
        };
        
        Self {
            signatures,
            observed_counts,
            exposures: init_exposures,
            intercept: init_intercept,
            gene_lengths,
            gene_weights,
            intercept_signature,
            signatures_with_intercept,
            use_intercept,
            config_min_exposure: min_exposure,
        }
    }
    
    /// E-step: Compute responsibilities (soft assignments)
    /// E-step with intercept: Compute responsibilities (OPTIMIZED - no loop!)
    fn e_step(&self) -> Tensor<B, 3> {
        let [_n_genes, _n_celltypes] = self.signatures.dims();
        let [_, _n_samples] = self.observed_counts.dims();
        
        // Build full exposure vector including intercept
        let full_exposures = if self.use_intercept {
            Tensor::cat(vec![self.exposures.clone(), self.intercept.clone()], 0)
        } else {
            self.exposures.clone()
        };
        
        // Expected counts: signatures_with_intercept @ full_exposures
        // [n_genes, n_sources] @ [n_sources, n_samples] = [n_genes, n_samples]
        let total_expected = self.signatures_with_intercept.clone()
            .matmul(full_exposures.clone())
            .add_scalar(1e-10);
        
        // VECTORIZED VERSION - no loop!
        // Compute all responsibilities at once using broadcasting
        
        // Reshape for broadcasting:
        // signatures: [n_genes, n_sources] -> [n_genes, n_sources, 1]
        // exposures: [n_sources, n_samples] -> [1, n_sources, n_samples]
        let sig_3d = self.signatures_with_intercept.clone()
            .unsqueeze_dim::<3>(2);  // [n_genes, n_sources, 1]
        
        let exp_3d = full_exposures.clone()
            .unsqueeze_dim::<3>(0);  // [1, n_sources, n_samples]
        
        // Broadcast multiply: [n_genes, n_sources, 1] * [1, n_sources, n_samples]
        // -> [n_genes, n_sources, n_samples]
        let contributions = sig_3d.mul(exp_3d);
        
        // Divide by total_expected: [n_genes, n_sources, n_samples] / [n_genes, 1, n_samples]
        let total_3d = total_expected
            .unsqueeze_dim::<3>(1);  // [n_genes, 1, n_samples]
        
        contributions.div(total_3d)
    }

    /// M-step with intercept: Update exposures (OPTIMIZED - vectorized!)
    fn m_step(&mut self, responsibilities: Tensor<B, 3>) {
        let [n_genes, _n_sources, n_samples] = responsibilities.dims();
        let n_celltypes = self.signatures.dims()[1];
        
        // Expand gene_weights for broadcasting: [n_genes] -> [n_genes, 1, n_samples]
        let weights_3d = self.gene_weights.clone()
            .reshape([n_genes, 1, 1])
            .repeat(&[1, 1, n_samples]);
        
        // Weight the responsibilities: [n_genes, n_sources, n_samples] * [n_genes, 1, n_samples]
        let weighted_resp = responsibilities.clone()
            .mul(weights_3d)
            .mul(self.observed_counts.clone().unsqueeze_dim::<3>(1)); // [n_genes, 1, n_samples]
        
        // Sum over genes: [n_genes, n_sources, n_samples] -> [1, n_sources, n_samples] -> [n_sources, n_samples]
        let all_exposures = weighted_resp
            .sum_dim(0)  // Sum over genes -> [1, n_sources, n_samples]
            .squeeze::<2>(0)  // Remove first dimension -> [n_sources, n_samples]
            .clamp_min(self.config_min_exposure);
        
        // Split into cell type exposures and intercept
        if self.use_intercept {
            // Cell types: first n_celltypes rows
            self.exposures = all_exposures.clone()
                .slice([0..n_celltypes, 0..n_samples]);
            
            // Intercept: last row
            self.intercept = all_exposures
                .slice([n_celltypes..n_celltypes+1, 0..n_samples]);
        } else {
            self.exposures = all_exposures;
        }
    }
    /// Compute Poisson log-likelihood (including intercept contribution)
    fn log_likelihood(&self) -> f32 {
        // Build full exposure vector
        let full_exposures = if self.use_intercept {
            Tensor::cat(vec![self.exposures.clone(), self.intercept.clone()], 0)
        } else {
            self.exposures.clone()
        };
        
        // Predicted counts including noise
        let predicted = self.signatures_with_intercept.clone()
            .matmul(full_exposures)
            .add_scalar(1e-10);
        
        let k = self.observed_counts.clone();
        let log_pred = predicted.clone().log();
        
        let ll = k.clone()
            .mul(log_pred)
            .sub(predicted)
            .mul(self.gene_weights.clone()
                .reshape([self.gene_weights.dims()[0], 1])
                .repeat(&[1, self.observed_counts.dims()[1]]))
            .sum();
        
        ll.into_scalar().elem::<f32>()
    }
    
    #[allow(dead_code)]
    pub fn get_exposures(&self) -> Tensor<B, 2> {
        self.exposures.clone()
    }
    
    #[allow(dead_code)]
    pub fn get_intercept(&self) -> Tensor<B, 2> {
        self.intercept.clone()
    }
    
    /// Get full exposures including intercept (for consistency with gradient descent version)
    pub fn get_exposures_with_intercept(&self) -> Tensor<B, 2> {
        if self.use_intercept {
            Tensor::cat(vec![self.exposures.clone(), self.intercept.clone()], 0)
        } else {
            self.exposures.clone()
        }
    }
    
    pub fn get_predicted_counts(&self) -> Tensor<B, 2> {
        let full_exposures = if self.use_intercept {
            Tensor::cat(vec![self.exposures.clone(), self.intercept.clone()], 0)
        } else {
            self.exposures.clone()
        };
        
        self.signatures_with_intercept.clone().matmul(full_exposures)
    }
}

/*
HOW THE INTERCEPT WORKS IN EM:

The intercept models uniform background noise/contamination across all genes.

Generative Model:
  For each gene i in sample k:
    1. Decide which "source" generated each read:
       - Cell type 1 (with probability ∝ sig_i1 * exp_1k)
       - Cell type 2 (with probability ∝ sig_i2 * exp_2k)
       - ...
       - Background noise (with probability ∝ intercept_sig_i * intercept_k)
    
    2. The background has uniform signature: intercept_sig_i = 1/n_genes
       (all genes equally likely under noise)

E-step Intuition:
  "For gene i with 1000 counts, how many came from noise vs each cell type?"
  
  If noise level is high (intercept_k = 500):
    → More counts attributed to noise
    → Fewer counts attributed to cell types
  
  If gene i has high signature in T-cells but also high noise:
    → Responsibility splits between T-cells and noise
    → EM automatically decides the best split

M-step Intuition:
  "Given the assignments, what's the total noise level?"
  
  If many genes have counts not explained by signatures:
    → Intercept increases (more noise)
  
  If signatures explain counts well:
    → Intercept decreases (less noise)

Comparison to Gradient Descent Intercept:
  Both approaches model the same thing, but:
  
  GD: Learns intercept in log-space, requires careful initialization
  EM: Learns intercept directly, automatically balances with signatures
  
  EM's interpretation: "Intercept competes with signatures for counts"
  This is MORE NATURAL than GD's log-space parameterization

Key Insight:
  The intercept is just another "signature" (uniform across genes)
  with its own exposure level that EM learns automatically!
*/

/// Train using EM algorithm (with or without intercept)
pub fn train_em_deconv<B: Backend>(
    mut model: EMDeconvModel<B>,
    config: EMConfig,
) -> EMDeconvModel<B> {
    if config.verbose {
        if model.use_intercept {
            eprintln!("Starting EM algorithm with intercept (noise modeling)...");
        } else {
            eprintln!("Starting EM algorithm without intercept...");
        }
    }
    
    let mut last_ll = f32::NEG_INFINITY;
    
    for iter in 0..config.max_iterations {
        // E-step: compute responsibilities
        let responsibilities = model.e_step();
        
        // M-step: update exposures
        model.m_step(responsibilities);
        
        // Check convergence every 10 iterations
        if iter % 10 == 0 {
            let current_ll = model.log_likelihood();
            let ll_change = current_ll - last_ll;
            
            if config.verbose && iter % 50 == 0 {
                // Show intercept info if using it
                if model.use_intercept {
                    let intercept_mean = model.intercept.clone().mean().into_scalar().elem::<f32>();
                    let total_exposure = model.exposures.clone().sum().into_scalar().elem::<f32>();
                    let noise_fraction = intercept_mean / (total_exposure / model.exposures.dims()[1] as f32 + intercept_mean);
                    
                    eprintln!(
                        "EM Iter {}: ll={:.2}, Δll={:.2}, noise={:.2} ({:.1}% of signal)",
                        iter, current_ll, ll_change, intercept_mean, noise_fraction * 100.0
                    );
                } else {
                    eprintln!(
                        "EM Iter {}: log-likelihood={:.2}, Δll={:.2}",
                        iter, current_ll, ll_change
                    );
                }
            }
            
            // Check convergence
            if iter > 10 && ll_change.abs() < config.tolerance {
                if config.verbose {
                    eprintln!("EM converged at iteration {} (Δll={:.6})", iter, ll_change);
                }
                break;
            }
            
            // EM should monotonically increase likelihood
            if ll_change < -1e-6 {
                eprintln!("Warning: Likelihood decreased at iteration {} (Δll={})", 
                         iter, ll_change);
            }
            
            last_ll = current_ll;
        }
        
        if iter == config.max_iterations - 1 {
            eprintln!("EM reached maximum iterations ({}) without converging", 
                     config.max_iterations);
        }
    }
    
    model
}

// =========================================
// IMPROVED EM WITH REGULARIZATION
// =========================================

/// EM with L1 regularization (promotes sparsity)
/// This is like adding a prior: exposures ~ Laplace(0, λ)
pub fn train_em_with_l1<B: Backend>(
    mut model: EMDeconvModel<B>,
    config: EMConfig,
    l1_lambda: f32,
) -> EMDeconvModel<B> {
    if config.verbose {
        eprintln!("Starting EM algorithm with L1 regularization (λ={})...", l1_lambda);
    }
    
    let mut last_objective = f32::NEG_INFINITY;
    
    for iter in 0..config.max_iterations {
        // E-step: same as before
        let responsibilities = model.e_step();
        
        // M-step with soft-thresholding for L1
        model.m_step_with_l1(responsibilities, l1_lambda);
        
        if iter % 10 == 0 {
            let ll = model.log_likelihood();
            let l1_penalty = model.exposures.clone().abs().sum().into_scalar().elem::<f32>() * l1_lambda;
            let objective = ll - l1_penalty;
            let obj_change = objective - last_objective;
            
            if config.verbose && iter % 50 == 0 {
                eprintln!(
                    "EM Iter {}: objective={:.2} (ll={:.2}, L1={:.2}), Δobj={:.2}",
                    iter, objective, ll, l1_penalty, obj_change
                );
            }
            
            if iter > 10 && obj_change.abs() < config.tolerance {
                if config.verbose {
                    eprintln!("EM converged at iteration {}", iter);
                }
                break;
            }
            
            last_objective = objective;
        }
    }
    
    model
}

impl<B: Backend> EMDeconvModel<B> {
    /// M-step with soft-thresholding for L1 penalty
    fn m_step_with_l1(&mut self, responsibilities: Tensor<B, 3>, l1_lambda: f32) {
        // First do standard M-step
        self.m_step(responsibilities);
        
        // Then apply soft-thresholding: shrink exposures toward zero
        // soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)
        let threshold = l1_lambda;
        
        self.exposures = self.exposures.clone()
            .sub_scalar(threshold)
            .clamp_min(0.0);  // This implements soft-thresholding for positive values
    }
}

// =========================================
// R INTERFACE INTEGRATION
// =========================================

// Import helper functions from your signal module
use crate::signal::{rmat_to_tensor, rvec_to_tensor, robj_to_bool, robj_to_f64, robj_to_usize};

pub fn fit_deconv_em(
    sigs: Robj,
    bulk: Robj,
    gene_lengths: Robj,
    w_vec: Robj,
    max_iter: Robj,
    tolerance: Robj,
    l1_lambda: Robj,
    verbose: Robj,
) -> Result<List> {
    // Use NdArray backend (CPU)
    type NdArrayBackend = burn::backend::NdArray<f32>;
    let device = <NdArrayBackend as Backend>::Device::default();
    
    let signatures = rmat_to_tensor::<NdArrayBackend>(sigs, &device)?;
    let observed_counts = rmat_to_tensor(bulk, &device)?;
    let gene_lengths_vec = rvec_to_tensor(gene_lengths, &device)?;
    let gene_weights = rvec_to_tensor(w_vec, &device)?;
    // eprintln!("DEBUG: gene_lengths_vec shape={:?}", gene_lengths_vec.dims());
    // eprintln!("DEBUG: gene_weights shape={:?}", gene_weights.dims());

    // Expand gene_lengths to matrix
    let [n_genes, _] = signatures.dims();
    let [_, n_samples] = observed_counts.dims();
    // eprintln!("DEBUG: About to reshape gene_lengths_vec [{}] to [{}, 1]", gene_lengths_vec.dims()[0], n_genes);

    let gene_lengths_matrix = gene_lengths_vec
        .reshape([n_genes, 1])
        .repeat(&[1, n_samples]);
    
    let config = EMConfig {
        max_iterations: robj_to_usize(max_iter, 1000, "max_iter"),
        tolerance: robj_to_f64(tolerance, 1e-6, "tolerance") as f32,
        min_exposure: 1e-10,
        verbose: robj_to_bool(verbose, true, "verbose"),
        use_intercept: true,  // Always use intercept for now
    };
    
    let model = EMDeconvModel::new(
        signatures,
        observed_counts,
        gene_lengths_matrix,
        gene_weights,
        config.use_intercept,
        config.min_exposure,
        &device,
    );
    
    // Train with or without L1
    let l1 = robj_to_f64(l1_lambda, 0.0, "l1_lambda") as f32;
    let trained_model = if l1 > 0.0 {
        train_em_with_l1(model, config, l1)
    } else {
        train_em_deconv(model, config)
    };
    
    let exposures = trained_model.get_exposures_with_intercept();
    let pred_counts = trained_model.get_predicted_counts();
    
    // Convert back to R matrices (reuse conversion logic from signal module)
    let exp_data: Vec<f32> = exposures.to_data().to_vec()
        .unwrap_or_else(|_| panic!("Failed to convert exposures"));
    let exp_dims = exposures.dims();
    
    let mut exp_col_major = vec![0.0f64; exp_data.len()];
    for i in 0..exp_dims[0] {
        for j in 0..exp_dims[1] {
            exp_col_major[j * exp_dims[0] + i] = exp_data[i * exp_dims[1] + j] as f64;
        }
    }
    
    let mut robj_exp = Robj::from(exp_col_major);
    let _ = robj_exp.set_attrib("dim", Robj::from(vec![exp_dims[0] as i32, exp_dims[1] as i32]));
    
    let pred_data: Vec<f32> = pred_counts.to_data().to_vec()
        .unwrap_or_else(|_| panic!("Failed to convert pred_counts"));
    let pred_dims = pred_counts.dims();
    
    let mut pred_col_major = vec![0.0f64; pred_data.len()];
    for i in 0..pred_dims[0] {
        for j in 0..pred_dims[1] {
            pred_col_major[j * pred_dims[0] + i] = pred_data[i * pred_dims[1] + j] as f64;
        }
    }
    
    let mut robj_pred = Robj::from(pred_col_major);
    let _ = robj_pred.set_attrib("dim", Robj::from(vec![pred_dims[0] as i32, pred_dims[1] as i32]));
    
    Ok(list!(
        exposures = robj_exp,
        pred_counts = robj_pred
    ))
}

/*
COMPARISON: EM vs Gradient Descent

EM Algorithm:
✅ No hyperparameters (learning rate, momentum, etc.)
✅ Guaranteed monotonic improvement
✅ Natural interpretation (missing data framework)
✅ Fast convergence (typically 100-300 iterations)
✅ No gradient computation overhead
✅ Simple to implement

Gradient Descent:
❌ Need to tune learning rate
❌ Can diverge or oscillate
❌ Less interpretable
❌ Typically 1000-3000 iterations
❌ Backpropagation overhead
❌ More complex

Expected Performance:
- EM: 100-300 iterations → ~3-5 seconds
- GD: 1000-2000 iterations → ~10-15 seconds
- Speedup: 3-5x

Quality:
- Both converge to similar local optima
- EM may be slightly more stable
- Results should be nearly identical
*/