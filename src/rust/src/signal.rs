// // use std::os::unix::raw::dev_t;
// use chrono::Utc;
// use burn::tensor::activation::sigmoid; 
// // use burn::tensor::TensorOps; 
// use burn::{
//     // prelude::Device,
//     // backend::ndarray::{NdArrayDevice},
//     module::Param,
//     optim::{AdamConfig, Optimizer, GradientsParams},      // <- import the trait and Optimizer
//     tensor::{
//         backend::{AutodiffBackend, Backend},
//         Tensor,
//     },
// };
// use burn_derive::Module as DeriveModule;
// // use serde::de;                 // <- only the derive macro
// /* ------------------------------------------------------------------ */
// /*  Hyper-parameters & constants                                      */
// /* ------------------------------------------------------------------ */

// #[derive(Clone)]
// pub struct Params {
//     // pub n_genes: usize,
//     pub n_types: usize,
//     pub n_samps: usize,
//     pub init_log_exp: f32,
//     pub l1: f32,
//     pub l2: f32,
//     pub lr: f64,
//     pub epochs: usize,
// }

// #[derive(Clone)]
// pub struct Consts<B: Backend> {
//     pub sp: Tensor<B, 2>,
//     pub c: Tensor<B, 2>,
//     pub k: Tensor<B, 2>,
//     pub w: Tensor<B, 2>,
//     pub lg_kp1: Tensor<B, 2>,
// }

// /* ------------------------------------------------------------------ */
// /*  Model                                                             */
// /* ------------------------------------------------------------------ */

// #[derive(DeriveModule, Debug)]
// pub struct Deconv<B: Backend> {
//     /* #[param] is optional with burn-derive 0.11 — the field type is
//        enough for the macro to register it.                           */
//     log_e: Param<Tensor<B, 2>>,
// }

// impl<B: Backend> Deconv<B> {
//     pub fn new(p: &Params, device: &B::Device) -> Self {
//         let mut init = Tensor::<B,2>::zeros( [p.n_types + 1, p.n_samps], device);
//         // let mut init = Tensor::<B,2>::zeros(device, [p.n_types + 1, p.n_samps]);
//         if p.init_log_exp != 0.0 {
//             init = init + p.init_log_exp;
//         }
//         Self {
//             log_e: Param::from_tensor(init)
//         }
//     }

//     #[inline(always)]
//     pub fn exposures(&self) -> Tensor<B, 2> {
//         self.log_e.val().exp()
//     }

//     #[inline(always)]
//     fn predict(&self, c: &Consts<B>) -> Tensor<B, 2> {
//         c.sp.clone().matmul(self.exposures()) * c.c.clone()
//     }

//     // pub fn loss(&self, c: &Consts<B>, p: &Params) -> Tensor<B, 1> {
//     //     /* Poisson NLL ------------------------------------------------ */
//     //     let y  = self.predict(c);
//     //     let k  = c.k.clone();
//     //     let ll = k.clone() * y.clone().log() - y;
//     //     // let term = (c.lg_kp1.clone() - ll) * c.w.clone().unsqueeze::<2>();
//     //     let term = (c.lg_kp1.clone() - ll) * c.w.clone();
//     //     let nll  = term.sum();

//     //     /* Elastic-net penalties -------------------------------------- */
//     //     let e   = self.exposures();
//     //     let l1 = e.clone().abs().sum_dim(0).sum() * p.l1;  // sum over all dims in one go
//     //     // let l1  = e.clone().abs().sum() * p.l1; // clone → avoid move
//     //     // let n_samples = e.dims()[1];
//     //     // let l2 = e.slice([0..p.n_types, 0..n_samples])
//     //     //         .powf_scalar(2.0)
//     //     //         .sum() * p.l2;
//     //     let l2 = e.clone().slice([0..p.n_types, 0..e.dims()[1]])
//     //             .powf_scalar(2.)
//     //             .sum() * p.l2;
//     //             nll + l1 + l2
//     // }
//     pub fn loss(&self, c: &Consts<B>, p: &Params) -> Tensor<B, 1> {
//         // ---- prediction with clamp for stability ----
//         let eps = 1e-9;
//         let y   = self.predict(c).clamp_min(eps);

//         // ---- Poisson terms ----
//         let k   = c.k.clone();
//         let ll  = k.clone() * y.clone().log() - y;

//         // If c.w is already (genes, samples), keep as-is; otherwise unsqueeze to match.
//         let term = (c.lg_kp1.clone() - ll) * c.w.clone();

//         let nll  = term.sum();

//         // ---- Elastic-net on NON-intercept rows ----
//         let e       = self.exposures();
//         let n_samps = e.dims()[1];

//         let l1 = e.clone()
//             .slice([0..p.n_types, 0..n_samps])
//             .abs()
//             .sum() * p.l1;

//         let l2 = e.clone()
//             .slice([0..p.n_types, 0..n_samps])
//             .powf_scalar(2.0)
//             .sum() * p.l2;

//         nll + l1 + l2
//     }


// }

// /* ------------------------------------------------------------------ */
// /*  Training loop                                                     */
// /* ------------------------------------------------------------------ */

// pub fn train<BB>(
//     consts: Consts<BB>,
//     params: Params,
// ) -> (Deconv<BB>, Vec<f32>)
// where
//     BB: AutodiffBackend + Backend<FloatElem = f32>,       // you get autodiff + tensor ops
//     BB::Device: Default,                 // so you can do `BB::Device::default()`
//     BB::FloatElem: Into<f32>,

// {
//     let device = <BB as Backend>::Device::default();
//     let mut model = Deconv::<BB>::new(&params, &device);

//     // --- Adam optimiser --------------------------------------------------
//     //     Generic params <B, M> are inferred from the variable's type.
//     let mut opt = AdamConfig::new().init();

//     let mut hist = Vec::with_capacity(params.epochs);


//     // use burn::module::ParamId;   // for ParamId


//     // /* null-model phase – optimise ONLY intercept row ------------------- */
//     // let r0          = params.n_types;    // last row index
//     // let null_epochs = 800;
//     // let mut opt_null = AdamConfig::new().init();
//     // eprintln!(
//     //     "Null model phase: {} epochs, lr = {:.3e}",
//     //     null_epochs, params.lr
//     // );
//     // let pid_intercept: ParamId = model.log_e.id;      // ParamId once
//     // eprintln!(
//     //     "Null model phase: intercept param id = {}",
//     //     pid_intercept
//     // );
//     // for _ in 0..null_epochs {
//     //     // ── inside the for _ in 0..null_epochs loop ────────────────
//     //     let loss    = model.loss(&consts, &params);
//     //     let mut raw = loss.backward();

//     //     /* 1️⃣ extract only the intercept gradient ------------------- */
//     //     let mut gs_one =
//     //         GradientsParams::from_params(&mut raw, &model, &[pid_intercept]);

//     //     /* 2️⃣ take ownership, zero non-intercept rows --------------- */
//     //     let mut grad = gs_one
//     //         .remove::<BB, 2>(pid_intercept)
//     //         .expect("intercept gradient missing");

//     //     let zeros =
//     //         Tensor::<BB, 2>::zeros([r0, params.n_samps], &grad.device());
//     //     grad = grad.slice_assign([0..r0, 0..params.n_samps], zeros);

//     //     /* 3️⃣ build a FRESH container with the single tensor -------- */
//     //     let mut gs_clean = GradientsParams::new();
//     //     gs_clean.register::<BB, 2>(pid_intercept, grad);

//     //     /* 4️⃣ optimiser step ---------------------------------------- */
//     //     model = opt_null.step(params.lr, model, gs_clean);

//     // }


//     /* --------------------------------------------------------------- */
//     /*  Closed-form fit of the intercept row (= “null model”)          */
//     /* --------------------------------------------------------------- */
//     // {
//     //     let r0 = params.n_types;                       // last row index

//     //     /* 1.  per-sample observed totals  k_tot_j -------------------- */
//     //     let obs_tot = consts.k.clone().sum_dim(0);             // shape (n_samps,)

//     //     /* 2.  per-sample baseline totals  Σ_i s0_i * c_ij ------------ */
//     //     // intercept signature column  s0_i   (n_genes × 1)
//     //     let s0 = consts.sp
//     //         .clone()
//     //         .slice([0..consts.sp.dims()[0], r0..r0 + 1]);   // (genes,1)

//     //     let base = (s0 * consts.c.clone()).sum_dim(0);      // (n_samps,)

//     //     /* 3.  MLE  e0_j = k_tot_j / base_j  -------------------------- */
//     //     let e0_log = (obs_tot / base).log();                // (n_samps,)

//     //     // 2. start from a detached clone of the original param
//     //     let log_e_init = model.log_e.val().clone().detach();

//     //     // 3. write the intercept row, THEN detach the result
//     //     let log_e = log_e_init
//     //         .slice_assign([r0..r0 + 1, 0..params.n_samps], e0_log.unsqueeze())
//     //         .detach();                           //  ← add this

//     //     // 4. wrap as a new Param (now a true leaf)
//     //     model.log_e = Param::from_tensor(log_e);
//     // }
//     {
//         let r0 = params.n_types;  // intercept row index

//         // k_tot_j
//         let obs_tot = consts.k.clone().sum_dim(0);  // (n_samps,)

//         // s0_i (genes × 1)
//         let s0 = consts.sp.clone().slice([0..consts.sp.dims()[0], r0..r0 + 1]);

//         // base_j = Σ_i s0_i * c_ij
//         let base = (s0 * consts.c.clone()).sum_dim(0);  // (n_samps,)

//         let eps = 1e-12;
//         let e0_log = (obs_tot / base.clamp_min(eps)).log(); // (n_samps,)

//         // detach-rewrite param row
//         let log_e_init = model.log_e.val().clone().detach();
//         let log_e = log_e_init
//             .slice_assign([r0..r0 + 1, 0..params.n_samps], e0_log.unsqueeze())
//             .detach();
//         model.log_e = Param::from_tensor(log_e);
//     }

//     for epoch in 0..params.epochs {
//         // forward + loss
//         let loss  = model.loss(&consts, &params);
//         let o_val: f64 = loss.clone().into_scalar().into();

//         // 1. backward pass – returns a plain Gradients map
//         let raw_grads = loss.backward();

//         // 2. wrap so Adam can match them to ParamIds
//         let grads = GradientsParams::from_grads(raw_grads, &model);
//         // if let Some(grad) = raw_grads.get(&model.log_e) {
//         //     eprintln!("mean |∇log_e| = {}", grad.abs().mean().into_scalar());
//         // }
//         // 3. update weights in-place
//         model = opt.step(params.lr as f64, model, grads);

//         hist.push(loss.clone().into_scalar().into());
//         if epoch % 100 == 0 {
//                 /* -------- diagnostics -------- */
//             // 1. active-coefficient count   (exclude the intercept row)
//             let log_e  = model.log_e.val().clone();                // shape (sigs+1, n)
//             let cnt_t = sigmoid(
//                 log_e.slice([0..params.n_types, 0..params.n_samps])
//             ).sum();                                            // scalar tensor
//             let cnt     = cnt_t.into_scalar() as f32;
//             let prev_o   = o_val;
//             let prev_cnt = cnt;
//             // // 2. average NLL per cell
//             // let y_hat   = model.predict(&consts);
//             // let k       = consts.k.clone();
//             // let ll_cell = k.clone() * y_hat.clone().log() - y_hat.clone();
//             // let nll_tot = (consts.lg_kp1.clone() - ll_cell) * consts.w.clone();
//             // let cells = (k.dims()[0] * k.dims()[1]) as f32;  
//             // let avg_nll = nll_tot.sum().into_scalar() / cells;   

//             // avg NLL as weight-normalized
//             let y_hat   = model.predict(&consts).clamp_min(1e-9);
//             let k       = consts.k.clone();
//             let ll_cell = k.clone() * y_hat.clone().log() - y_hat.clone();
//             let nll_tot = (consts.lg_kp1.clone() - ll_cell) * consts.w.clone();

//             let w_sum   = consts.w.clone().sum().into_scalar().max(1e-12);
//             let avg_nll = nll_tot.sum().into_scalar() / w_sum;
//             // 3. average coverage ratio
//             let pred_tot = y_hat.sum_dim(0);   // sum over genes → (n,)
//             let obs_tot  = k.sum_dim(0);
//             let avg_cov  = (pred_tot / obs_tot).mean().into_scalar();

//             // 4. deltas
//             let d_o   = if prev_o.is_nan() { 0.0 } else { o_val - prev_o };
//             let d_cnt = if prev_cnt.is_nan() { 0.0 } else { cnt   - prev_cnt };
//             // eprintln!("epoch {epoch:4}  loss = {}", loss.into_scalar());
//             eprintln!(
//                 "[{}] epoch {:5}, Obj = {:.5e}, ActiveCoeff ={:.2}, delta Obj={:+.3}, \
//                 delta ActiveCoeff={:+.3}, nSigsAvg={:.4}/{}, avgNLL={:.2}, avgCov={:.6}",
//                 Utc::now().format("%Y-%m-%d %H:%M:%S%.6f"),
//                 epoch,
//                 o_val,
//                 cnt,
//                 d_o,
//                 d_cnt,
//                 cnt / params.n_samps as f32,
//                 params.n_types + 1,
//                 avg_nll,
//                 avg_cov
//             );
//             }
//     }
//     (model, hist)
// }
