
// use crate::common::*;
// use crate::scrna_mlr::ModelConfig as MlrCfg;
// use crate::scrna_ann ::ModelConfig as AnnCfg;
// use crate::scrna_ann2l::ModelConfig as Ann2Cfg;
// use burn::{
//     data::{dataloader::DataLoaderBuilder, dataset::{InMemDataset, transform::MapperDataset}},
//     module::Module,
//     record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
//     tensor::{Tensor, DType, backend::Backend as BurnBackend},
// };
// use num_traits::ToPrimitive;



// #[derive(Clone, Debug)]
// pub enum NetKind {
//     Mlr,
//     Ann  { hidden: usize },
//     Ann2 { hidden1: usize, hidden2: usize },
// }


// fn run_inference<B, F>(
//     mut f      : F,
//     device     : <B as BurnBackend>::Device,
//     query      : Vec<SCItemRaw>,
//     batch_size : usize,
// ) -> Vec<f32>
// where
//     B: BurnBackend + 'static,
//     B::FloatElem: ToPrimitive,      // ← was Elem, now correct
//     F: FnMut(Tensor<B, 2>) -> Tensor<B, 2>,
// {
//     let dataset = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
//     let loader  = DataLoaderBuilder::new(SCBatcher::<B>::new(device))
//         .batch_size(batch_size)
//         .build(dataset);

//     //  let mut first_batch = true;
//     loader
//         .iter()
//         .flat_map(|batch| {

//             // if first_batch {
//             //     eprintln!("First batch input shape: {:?}", batch.counts.dims());
//             //     first_batch = false;
//             // }

//             // 1.  Run the model
//             let mut data = f(batch.counts).into_data();   // TensorData

//             // 2.  Make sure the buffer is f32-typed
//             if data.dtype != DType::F32 {
//                 data = data.convert::<f32>();             // still TensorData
//             }

//             // 3.  Take the buffer out as a Vec<f32>
//             let vec = data.into_vec::<f32>().unwrap();    // Vec<f32>

//             vec.into_iter()
//         })
//         .collect()
// }


// pub fn infer<B>(
//     model_path   : &str,
//     net          : NetKind,
//     num_classes  : usize,
//     num_features : usize,
//     query        : Vec<SCItemRaw>,
//     batch_size   : Option<usize>,
//     device : <B as BurnBackend>::Device,
// ) -> Vec<f32>
// where
//     B: BurnBackend + 'static,
//     B::FloatElem: ToPrimitive,
// {
//     let bs = batch_size.unwrap_or(64);

//     match net {
//         NetKind::Mlr => {
//             let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//                            .load(model_path.into(), &device)
//                            .expect("load MLR weights");
//             let model = MlrCfg::new(num_classes)
//                            .init(num_features, device.clone())
//                            .load_record(rec);
//             // eprintln!("Model initialized with:");
//             // eprintln!("  num_features: {}", num_features);
//             // eprintln!("  num_classes: {}", num_classes);
//             run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
//         }
//         // NetKind::Mlr => {
//         //     eprintln!("=== Debug Info ===");
//         //     eprintln!("Model path: {}", model_path);
            
//         //     // Load the record to inspect it
//         //     let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//         //         .load(model_path.into(), &device)
//         //         .expect("load MLR weights");
            
//         //     // Try creating a fresh model and see what happens
//         //     let fresh_model = MlrCfg::new(num_classes)
//         //         .init(num_features, device.clone());
            
//         //     // Test the fresh model first
//         //     let test_input = Tensor::<B, 2>::zeros([1, num_features], &device);
//         //     eprintln!("Testing fresh model (before loading weights)...");
            
//         //     let fresh_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
//         //         fresh_model.forward(test_input.clone())
//         //     }));
            
//         //     match fresh_result {
//         //         Ok(output) => {
//         //             eprintln!("Fresh model works! Output shape: {:?}", output.dims());
//         //             eprintln!("Now loading saved weights...");
                    
//         //             // Now try loading the weights
//         //             let loaded_model = fresh_model.load_record(rec);
                    
//         //             let loaded_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
//         //                 loaded_model.forward(test_input.clone())
//         //             }));
                    
//         //             match loaded_result {
//         //                 Ok(output2) => {
//         //                     eprintln!("Loaded model works! Output shape: {:?}", output2.dims());
                            
//         //                     // Proceed with actual inference
//         //                     run_inference::<B, _>(move |x| loaded_model.forward(x), device, query, bs)
//         //                 },
//         //                 Err(_) => {
//         //                     panic!("Model fails after loading weights - dimension mismatch in saved file");
//         //                 }
//         //             }
//         //         },
//         //         Err(_) => {
//         //             panic!("Fresh model fails - initialization problem");
//         //         }
//         //     }
//         // }

//         NetKind::Ann { hidden } => {
//             let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//                            .load(model_path.into(), &device)
//                            .expect("load ANN-1L weights");
//             let model = AnnCfg::new(num_classes, 0, hidden, 0.0)
//                            .init(num_features, device.clone())
//                            .load_record(rec);

//             run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
//         }

//         NetKind::Ann2 { hidden1, hidden2 } => {
//             let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//                            .load(model_path.into(), &device)
//                            .expect("load ANN-2L weights");
//             let model = Ann2Cfg::new(num_classes, 0, hidden1, hidden2, 0.0)
//                            .init(num_features, device.clone())
//                            .load_record(rec);

//             run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
//         }
//     }
// }


// pub fn infer_nd_parallel(
//     model_path: &str,
//     net: NetKind,
//     num_classes: usize,
//     num_features: usize,
//     query: Vec<SCItemRaw>,
//     batch_size: Option<usize>,
//     num_threads: usize,
//     verbose: bool,
// ) -> Vec<f32> {
//     use burn::backend::ndarray::{NdArray, NdArrayDevice};
//     use rayon::prelude::*;

//     type B = NdArray<f32, i32>;

//     let n_cells = query.len();
//     let chunk_size = (n_cells / num_threads).max(1000);
//     let bs = batch_size.unwrap_or(64);

//     if verbose {
//         eprintln!(
//             "Parallel inference: {} cells, {} threads, chunk_size {}",
//             n_cells, num_threads, chunk_size
//         );
//     }

//     // Split into chunks
//     let chunks: Vec<Vec<SCItemRaw>> = query
//         .chunks(chunk_size)
//         .map(|c| c.to_vec())
//         .collect();

//     let n_chunks = chunks.len();
//     if verbose {
//         eprintln!("Split into {} chunks", n_chunks);
//     }

//     let pool = rayon::ThreadPoolBuilder::new()
//         .num_threads(num_threads)
//         .build()
//         .expect("Failed to build rayon thread pool");

//     pool.install(|| {
//         chunks
//             .into_par_iter()
//             .enumerate()
//             .flat_map(|(i, chunk)| {
//                 if verbose {
//                     eprintln!("Processing chunk {}/{}", i + 1, n_chunks);
//                 }
//                 let device = NdArrayDevice::default();
//                 infer::<B>(
//                     model_path,
//                     net.clone(),
//                     num_classes,
//                     num_features,
//                     chunk,
//                     Some(bs),
//                     device,
//                 )
//             })
//             .collect()
//     })
// }


// pub fn infer_wgpu(
//     model_path   : &str,
//     net          : NetKind,
//     num_classes  : usize,
//     num_features : usize,
//     query        : Vec<SCItemRaw>,
//     batch_size   : Option<usize>,
// ) -> Vec<f32> {
//     use burn::backend::wgpu::{Wgpu, WgpuDevice};
//     type B = Wgpu<f32, i32>;
//     infer::<B>(model_path,
//         net,
//         num_classes,
//         num_features,
//         query,
//         batch_size, WgpuDevice::default())
// }

// pub fn infer_nd(
//     model_path   : &str,
//     net          : NetKind,
//     num_classes  : usize,
//     num_features : usize,
//     query        : Vec<SCItemRaw>,
//     batch_size   : Option<usize>,
// ) -> Vec<f32> {
//     use burn::backend::ndarray::{NdArray, NdArrayDevice}; 
//     type B = NdArray<f32, i32>;
//     infer::<B>(model_path,
//         net,
//         num_classes,
//         num_features,
//         query,
//         batch_size, NdArrayDevice::default())
// }

// pub fn infer_candle(
//     model_path   : &str,
//     net          : NetKind,
//     num_classes  : usize,
//     num_features : usize,
//     query        : Vec<SCItemRaw>,
//     batch_size   : Option<usize>,
// ) -> Vec<f32> {
//     use burn::backend::candle::{Candle, CandleDevice}; 
//     type B = Candle<f32, i64>;
//     infer::<B>(model_path,
//         net,
//         num_classes,
//         num_features,
//         query,
//         batch_size, CandleDevice::default())
// }

