// use burn::{
//     module::{Module, Param, ParamId},
//     optim::{AdamConfig, Optimizer},      // <- import the trait
//     tensor::{
//         backend::{AutodiffBackend, Backend},
//         Tensor,
//     },
// };
// use burn_derive::Module as DeriveModule;                 // <- only the derive macro
// use burn_autodiff::backend::AutodiffBackend;                             // <- needed for `.backward()`
// /* ------------------------------------------------------------------ */
// /*  Hyper-parameters & constants                                      */
// /* ------------------------------------------------------------------ */

// #[derive(Clone)]
// pub struct Params {
//     pub n_genes: usize,
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
//     pub w: Tensor<B, 1>,
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
//     pub fn new(p: &Params) -> Self {
//         let mut init = Tensor::<B, 2>::zeros([p.n_types + 1, p.n_samps]);
//         if p.init_log_exp != 0.0 {
//             init = init + p.init_log_exp;
//         }
//         Self {
//             log_e: Param::new(ParamId::new(), init),
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

//     pub fn loss(&self, c: &Consts<B>, p: &Params) -> Tensor<B, 1> {
//         /* Poisson NLL ------------------------------------------------ */
//         let y  = self.predict(c);
//         let k  = c.k.clone();
//         let ll = k.clone() * y.clone().log() - y;
//         let term = (c.lg_kp1.clone() - ll) * c.w.clone().unsqueeze::<2>();
//         let nll  = term.sum();

//         /* Elastic-net penalties -------------------------------------- */
//         let e   = self.exposures();
//         let l1  = e.clone().abs().sum() * p.l1; // clone → avoid move
//         let n_samples = e.dims()[1];
//         let l2  = e
//             .slice([0..p.n_types, 0..n_samples])
//             .powf(2.0)
//             .sum()
//             * p.l2;

//         nll + l1 + l2
//     }
// }

// /* ------------------------------------------------------------------ */
// /*  Training loop                                                     */
// /* ------------------------------------------------------------------ */xs

// pub fn train<B>(
//     consts: Consts<B>,
//     params: Params,
// ) -> (Deconv<B>, Vec<f32>)
// where
//     B: AutodiffBackend,
//     B::FloatElem: Into<f32>,
// {
//     let device = B::Device::default();
//     let mut model = Deconv::<B>::new(&params).to_device(&device);
//     let mut opt = AdamConfig::new().init();

//     let mut hist = Vec::with_capacity(params.epochs);
//     for _ in 0..params.epochs {
//         let loss  = model.loss(&consts, &params);
//         let grads = loss.backward();
//         opt.step(params.lr, grads, GradientsParams::default());
//         hist.push(loss.into_scalar().into());
//     }
//     (model, hist)
// }
