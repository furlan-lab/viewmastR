use extendr_api::prelude::*;
use burn::prelude::*;
use burn::tensor::{Tensor, backend::{Backend, AutodiffBackend}, activation};
use burn::module::Param;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

// Backend type aliases
type NdArrayBackend = burn::backend::Autodiff<burn::backend::NdArray>;
type WgpuBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
type CandleBackend = burn::backend::Autodiff<burn::backend::Candle>;

#[derive(Debug, Clone)]
enum BackendType {
    NdArray,
    Wgpu,
    Candle
}

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
    pub verbose: bool
}

impl Default for DeconvConfig {
    fn default() -> Self {
        Self {
            insert_size: 500.0,
            init_log_exposure: -5.0,
            learn_rate: 0.01,
            l1_lambda: 0.0,
            l2_lambda: 0.0,
            max_iterations: 10_000_000,
            poll_interval: 100,
            ll_tolerance: 1e-6,
            sparsity_tolerance: 1e-4,
            verbose: true,
        }
    }
}

#[derive(Module, Debug)]
pub struct DeconvModel<B: Backend> {
    log_exposures: Param<Tensor<B, 2>>,
    intercept: Param<Tensor<B, 2>>,
    
    #[module(skip)]
    signatures: Tensor<B, 2>,
    #[module(skip)]
    gene_lengths: Tensor<B, 2>,
    #[module(skip)]
    observed_counts: Tensor<B, 2>,
    #[module(skip)]
    gene_weights: Tensor<B, 1>,
    #[module(skip)]
    gene_weights_expanded: Tensor<B, 2>,
    #[module(skip)]
    intercept_signature: Tensor<B, 2>,
    #[module(skip)]
    conversion_factor: Tensor<B, 2>,
    #[module(skip)]
    signatures_with_int: Tensor<B, 2>,
    #[module(skip)]
    k_log_k_minus_k: Tensor<B, 2>,
}

impl<B: Backend> DeconvModel<B> {
    pub fn new(
        signatures: Tensor<B, 2>,
        gene_lengths: Tensor<B, 2>,
        observed_counts: Tensor<B, 2>,
        gene_weights: Tensor<B, 1>,
        config: &DeconvConfig,
        device: &B::Device,
    ) -> Self {
        let [n_genes, n_sigs] = signatures.dims();
        let [_, n_samples] = observed_counts.dims();
        
        let log_exposures = Tensor::random(
            [n_sigs, n_samples],
            burn::tensor::Distribution::Normal(config.init_log_exposure as f64, 0.1),
            device
        );
        let intercept = Tensor::zeros([1, n_samples], device);
        
        let intercept_signature = Tensor::ones([n_genes, 1], device)
            .div_scalar(n_genes as f32);
        
        let signatures_with_int = Tensor::cat(
            vec![signatures.clone(), intercept_signature.clone()],
            1
        );
        
        let gene_weights_expanded = gene_weights.clone()
            .reshape([n_genes, 1])
            .repeat(&[1, n_samples]);
        
        let conversion_factor = gene_lengths.clone()
            .div_scalar(config.insert_size);
        
        let k = observed_counts.clone();
        let k_safe = k.clone().add_scalar(1e-10);
        let k_log_k_minus_k = k.clone().mul(k_safe.log()).sub(k.clone());
        
        Self {
            log_exposures: Param::from_tensor(log_exposures),
            intercept: Param::from_tensor(intercept),
            signatures,
            gene_lengths,
            observed_counts,
            gene_weights,
            gene_weights_expanded,
            intercept_signature,
            conversion_factor,
            signatures_with_int,
            k_log_k_minus_k,
        }
    }
    
    pub fn forward(&self) -> Tensor<B, 2> {
        let exposures = self.log_exposures.val().exp();
        let intercept_exp = self.intercept.val().exp();
        let all_exposures = Tensor::cat(vec![exposures, intercept_exp], 0);
        
        let pred_mols = self.signatures_with_int.clone().matmul(all_exposures);
        pred_mols.mul(self.conversion_factor.clone())
    }
    
    pub fn poisson_nll(&self, pred_counts: Tensor<B, 2>) -> Tensor<B, 2> {
        let k = self.observed_counts.clone();
        let log_pred = pred_counts.clone().add_scalar(1e-10).log();
        let var_part = k.mul(log_pred).sub(pred_counts);
        let nll_matrix = self.k_log_k_minus_k.clone().sub(var_part);
        nll_matrix.mul(self.gene_weights_expanded.clone())
    }


    pub fn l1_penalty(&self, _sample_nulls: &Tensor<B, 1>, l1_lambda: f32) -> Tensor<B, 1> {
        if l1_lambda == 0.0 {
            return Tensor::zeros([1], &self.log_exposures.device());
        }
        
        // Simple L1 on exposures
        let exposures = self.log_exposures.val().exp();
        let intercept_exp = self.intercept.val().exp();
        
        let penalty = exposures.sum().add(intercept_exp.sum());
        penalty.mul_scalar(l1_lambda).reshape([1])
    }

    pub fn l2_penalty(&self, _sample_nulls: &Tensor<B, 1>, l2_lambda: f32) -> Tensor<B, 1> {
        if l2_lambda == 0.0 {
            return Tensor::zeros([1], &self.log_exposures.device());
        }
        
        // Simple L2 on exposures
        let exposures = self.log_exposures.val().exp();
        let intercept_exp = self.intercept.val().exp();
        
        let penalty = exposures.powf_scalar(2.0).sum()
            .add(intercept_exp.powf_scalar(2.0).sum());
        penalty.mul_scalar(l2_lambda).reshape([1])
    }

    pub fn loss(&self, sample_nulls: &Tensor<B, 1>, config: &DeconvConfig) -> Tensor<B, 1> {
        let pred_counts = self.forward();
        let nll_matrix = self.poisson_nll(pred_counts);
        let nll = nll_matrix.sum();
        
        // Scale regularization to make lambda values intuitive
        // Target: lambda = 1.0 should give ~1-5% penalty relative to NLL
        let nll_magnitude = nll.clone().into_scalar().elem::<f32>();
        
        // Estimate typical exposure sum for scaling
        // For most datasets: sum(exposures) is proportional to total counts / n_genes
        // This makes the penalty scale-invariant across different data sizes
        let exposure_scale = nll_magnitude / 100.0;  // Empirical scaling factor
        
        let l1 = self.l1_penalty(sample_nulls, config.l1_lambda * exposure_scale);
        let l2 = self.l2_penalty(sample_nulls, config.l2_lambda * exposure_scale);
        
        nll.add(l1).add(l2)
    }
    pub fn count_nonzero(&self) -> f32 {
        let exposures = self.log_exposures.val();
        let intercept = self.intercept.val();
        let all_exp = Tensor::cat(vec![exposures, intercept], 0);
        activation::sigmoid(all_exp).sum().into_scalar().elem::<f32>()
    }
    
    pub fn get_exposures(&self) -> Tensor<B, 2> {
        let exposures = self.log_exposures.val().exp();
        let intercept = self.intercept.val().exp();
        Tensor::cat(vec![exposures, intercept], 0)
    }
}

pub fn train_deconvolution<B: AutodiffBackend>(
    mut model: DeconvModel<B>,
    config: DeconvConfig,
    device: &B::Device,
) -> DeconvModel<B> {

    
    let sample_nulls = fit_null_model(&mut model, &config, device);
    if config.verbose {
        eprintln!("NULL model complete. Mean null NLL: {:.2}", 
              sample_nulls.clone().mean().into_scalar().elem::<f32>());
    }
    
    let [n_sigs, n_samples] = model.log_exposures.dims();
    let log_exposures = Tensor::random(
        [n_sigs, n_samples],
        burn::tensor::Distribution::Normal(config.init_log_exposure as f64, 0.1),
        device
    );
    model.log_exposures = Param::from_tensor(log_exposures);
    
    if config.verbose {
        eprintln!("Fitting full model...");
    }
    
    let mut optim = AdamConfig::new().init();
    let mut last_loss = f32::INFINITY;
    let mut last_nonzero = 0.0;
    let mut no_improvement_count = 0;
    
    for iter in 0..config.max_iterations {
        let loss = model.loss(&sample_nulls, &config);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.learn_rate, model, grads);
        
        if iter % config.poll_interval == 0 {
            let current_loss = loss.clone().into_scalar().elem::<f32>();
            let current_nonzero = model.count_nonzero();
            let loss_diff = last_loss - current_loss;
            let nz_diff = (last_nonzero - current_nonzero).abs();
            
            // if iter % 1000 == 0 {
            //     let exp = model.get_exposures();
            //     let exp_mean = exp.clone().mean().into_scalar().elem::<f32>();
            //     let exp_max = exp.clone().max().into_scalar().elem::<f32>();
            //     let exp_min = exp.clone().min().into_scalar().elem::<f32>();
                
            //     // DEBUG: Show regularization penalty breakdown
            //     let pred_for_nll = model.forward();
            //     let nll_matrix = model.poisson_nll(pred_for_nll.clone());
            //     let nll_only = nll_matrix.sum().into_scalar().elem::<f32>();
            //     let l1_penalty = model.l1_penalty(&sample_nulls, config.l1_lambda).into_scalar().elem::<f32>();
            //     let l2_penalty = model.l2_penalty(&sample_nulls, config.l2_lambda).into_scalar().elem::<f32>();
            //     let exp_sum = exp.clone().sum().into_scalar().elem::<f32>();
                
            //     eprintln!(
            //         "Iter {}: loss={:.2e} [NLL={:.2e}, L1={:.2e}, L2={:.2e}], exp_sum={:.2e}, nonzero={:.1}, Δloss={:.6}",
            //         iter, current_loss, nll_only, l1_penalty, l2_penalty, exp_sum, current_nonzero, loss_diff
            //     );
            //     eprintln!(
            //         "  exp(min/mean/max)=({:.4}/{:.4}/{:.4}), L1_lambda={}, L2_lambda={}",
            //         exp_min, exp_mean, exp_max, config.l1_lambda, config.l2_lambda
            //     );
            // } else {
            //     eprintln!(
            //         "Iter {}: loss={:.6}, nonzero={:.1}, Δloss={:.6}, Δnz={:.2}",
            //         iter, current_loss, current_nonzero, loss_diff, nz_diff
            //     );
            // }
            
            if config.verbose {
                eprintln!(
                    "Iter {}: loss={:.6}, nonzero={:.1}, Δloss={:.6}, Δnz={:.2}",
                    iter, current_loss, current_nonzero, loss_diff, nz_diff
                );
            }
            if loss_diff < 0.0 {
                no_improvement_count += 1;
            } else {
                no_improvement_count = 0;
            }
            
            if no_improvement_count > 10 {
                eprintln!("No improvement for {} checks, stopping early", no_improvement_count);
                break;
            }
            
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
        if iter == config.max_iterations - 1 {
            eprintln!("Reached maximum iterations ({}) without converging", config.max_iterations);
        }
    }
    
    model
}

fn fit_null_model<B: AutodiffBackend>(
    model: &mut DeconvModel<B>,
    config: &DeconvConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [n_sigs, n_samples] = model.log_exposures.dims();
    let frozen_exp = Tensor::zeros([n_sigs, n_samples], device)
        .sub_scalar(100.0);
    model.log_exposures = Param::from_tensor(frozen_exp);
    
    let mut optim = AdamConfig::new().init();
    let dummy_nulls = Tensor::zeros([n_samples], device);
    let mut last_loss = f32::INFINITY;
    
    for iter in 0..config.max_iterations.min(10000) {
        let loss = model.loss(&dummy_nulls, config);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, model);
        *model = optim.step(config.learn_rate, model.clone(), grads);
        
        if iter % config.poll_interval == 0 {
            let current_loss = loss.clone().into_scalar().elem::<f32>();
            
            if iter % 1000 == 0 {
                eprintln!("NULL iter {}: loss={:.6}", iter, current_loss);
            }
            
            let loss_change = (last_loss - current_loss).abs();
            if iter > 100 && loss_change / current_loss < 1e-8 {
                eprintln!("NULL model converged at iteration {}", iter);
                break;
            }
            
            last_loss = current_loss;
        }
    }
    
    let pred = model.forward();
    let nll_matrix = model.poisson_nll(pred);
    nll_matrix.sum_dim(0).squeeze(0)
}

// R INTERFACE

pub fn rmat_to_tensor<B: Backend>(robj: Robj, device: &B::Device) -> Result<Tensor<B, 2>> {
    let dims = robj.dim()
        .ok_or_else(|| Error::Other("Matrix must have dimensions".into()))?;
    
    if dims.len() != 2 {
        return Err(Error::Other("Expected 2D matrix".into()));
    }
    
    let nrow = dims[0].inner() as usize;
    let ncol = dims[1].inner() as usize;
    
    // eprintln!("▶ rmat_to_tensor: target shape = [{}, {}]", nrow, ncol);
    
    let data: Vec<f32> = robj
        .as_real_vector()
        .ok_or_else(|| Error::Other("Matrix must be numeric".into()))?
        .iter()
        .map(|&x| x as f32)
        .collect();
    
    // eprintln!("  data length = {}", data.len());
    
    if data.len() != nrow * ncol {
        return Err(Error::Other(format!(
            "Data length {} doesn't match dimensions {} × {} = {}",
            data.len(), nrow, ncol, nrow * ncol
        ).into()));
    }
    
    let mut row_major = vec![0.0f32; data.len()];
    for i in 0..nrow {
        for j in 0..ncol {
            row_major[i * ncol + j] = data[j * nrow + i];
        }
    }
    
    let t1 = Tensor::<B, 1>::from_floats(row_major.as_slice(), device);
    let t2 = t1.unsqueeze::<2>();
    let t3 = t2.reshape([nrow, ncol]);
    
    Ok(t3)
}

pub fn rvec_to_tensor<B: Backend>(robj: Robj, device: &B::Device) -> Result<Tensor<B, 1>> {
    let data: Vec<f32> = robj
        .as_real_vector()
        .ok_or_else(|| Error::Other("Vector must be numeric".into()))?
        .iter()
        .map(|&x| x as f32)
        .collect();
    
    Ok(Tensor::<B, 1>::from_floats(data.as_slice(), device))
}

pub fn robj_to_bool(robj: Robj, default: bool, name: &str) -> bool {
    robj.as_logical_vector()
        .and_then(|v| v.first().copied())
        .map(|rbool| rbool.is_true())  // Convert Rbool to bool
        .unwrap_or_else(|| {
            rprintln!("Warning: Using default value {} for {}", default, name);
            default
        })
}

pub fn robj_to_f64(robj: Robj, default: f64, name: &str) -> f64 {
    robj.as_real_vector()
        .and_then(|v| v.first().copied())
        .unwrap_or_else(|| {
            eprintln!("Warning: Using default value {} for {}", default, name);
            default
        })
}

pub fn robj_to_usize(robj: Robj, default: usize, name: &str) -> usize {
    robj.as_real_vector()
        .and_then(|v| v.first().copied())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            eprintln!("Warning: Using default value {} for {}", default, name);
            default
        })
}


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
    verbose: Robj
) -> Result<List> {
    let backend_str = backend.as_str().unwrap_or("ndarray").to_lowercase();
    
    let backend_type = match backend_str.as_str() {
        "ndarray" | "cpu" => BackendType::NdArray,
        "wgpu" | "gpu" => BackendType::Wgpu,
        "candle" => BackendType::Candle,
        _ => {
            rprintln!("Unknown backend '{}', using ndarray", backend_str);
            BackendType::NdArray
        }
    };
    
    // eprintln!("Using backend: {:?}", backend_type);
    
    match backend_type {
        BackendType::Wgpu => fit_deconv_impl::<WgpuBackend>(
            sigs, bulk, gene_lengths, w_vec,
            insert_size, init_log_exp, lr, l1_lambda, l2_lambda,
            max_iter, poll_interval, ll_tol, sparsity_tol, verbose
        ),
        BackendType::Candle => fit_deconv_impl::<CandleBackend>(
            sigs, bulk, gene_lengths, w_vec,
            insert_size, init_log_exp, lr, l1_lambda, l2_lambda,
            max_iter, poll_interval, ll_tol, sparsity_tol, verbose
        ),
        BackendType::NdArray => fit_deconv_impl::<NdArrayBackend>(
            sigs, bulk, gene_lengths, w_vec,
            insert_size, init_log_exp, lr, l1_lambda, l2_lambda,
            max_iter, poll_interval, ll_tol, sparsity_tol, verbose
        ),
    }
}

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
    verbose: Robj,
) -> Result<List> {
    let device = <B as Backend>::Device::default();
    
    let signatures: Tensor<B, 2> = rmat_to_tensor(sigs, &device)?;
    let observed_counts: Tensor<B, 2> = rmat_to_tensor(bulk, &device)?;
    let gene_lengths_vec: Tensor<B, 1> = rvec_to_tensor(gene_lengths, &device)?;
    let gene_weights: Tensor<B, 1> = rvec_to_tensor(w_vec, &device)?;
    
    let [n_genes, _n_sigs] = signatures.dims();
    let [bulk_genes, n_samples] = observed_counts.dims();
    
    if n_genes != bulk_genes {
        return Err(Error::Other(format!(
            "Gene dimension mismatch: sigs={}, bulk={}",
            n_genes, bulk_genes
        ).into()));
    }
    
    let gene_lengths_matrix = gene_lengths_vec
        .clone()
        .reshape([n_genes, 1])
        .repeat(&[1, n_samples]);
    
    let config = DeconvConfig {
        insert_size: robj_to_f64(insert_size, 500.0, "insert_size") as f32,
        init_log_exposure: robj_to_f64(init_log_exp, -5.0, "init_log_exp") as f32,
        learn_rate: robj_to_f64(lr, 0.01, "lr"),
        l1_lambda: robj_to_f64(l1_lambda, 0.0, "l1_lambda") as f32,
        l2_lambda: robj_to_f64(l2_lambda, 0.0, "l2_lambda") as f32,
        max_iterations: robj_to_usize(max_iter, 10000, "max_iter"),
        poll_interval: robj_to_usize(poll_interval, 100, "poll_interval"),
        ll_tolerance: robj_to_f64(ll_tol, 1e-6, "ll_tol") as f32,
        sparsity_tolerance: robj_to_f64(sparsity_tol, 1e-4, "sparsity_tol") as f32,
        verbose: robj_to_bool(verbose, true, "verbose")
    };
    
    // eprintln!("Model: {} genes, {} cell types, {} samples", n_genes, n_sigs, n_samples);
    // println!("Config: L1={}, L2={}, lr={}, insert_size={}", 
            //   config.l1_lambda, config.l2_lambda, config.learn_rate, config.insert_size);
    
    let model = DeconvModel::new(
        signatures,
        gene_lengths_matrix,
        observed_counts,
        gene_weights,
        &config,
        &device,
    );
    
    let trained_model = train_deconvolution(model, config.clone(), &device);
    let exposures = trained_model.get_exposures();
    let pred_counts = trained_model.forward();
    
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
    
    eprintln!("Training complete!");
    
    Ok(list!(
        exposures = robj_exp,
        pred_counts = robj_pred
    ))
}
