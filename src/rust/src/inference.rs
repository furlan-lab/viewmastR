use crate::common::*;
use crate::scrna_mlr::ModelConfig as MlrCfg;
use crate::scrna_ann ::ModelConfig as AnnCfg;
use crate::scrna_ann2l::ModelConfig as Ann2Cfg;

/// What kind of network do we want to load?
#[derive(Debug)]
pub enum NetKind {
    Mlr,
    Ann  { hidden: usize },
    Ann2 { hidden1: usize, hidden2: usize },
}


use burn::{
    // backend::candle::{Candle, CandleDevice},
    // backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice},
    data::{dataloader::DataLoaderBuilder, dataset::{InMemDataset, transform::MapperDataset}},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::Tensor,
};

use num_traits::ToPrimitive;

// ─────────────────────────────────────────────────────────────
// 1.  Bring the **trait** into scope and give it an unambiguous name。
// ─────────────────────────────────────────────────────────────
use burn::tensor::{backend::Backend as BurnBackend};

// ─────────────────────────────────────────────────────────────
// 2.  DELETE the old hard-wired alias; it shadowed the trait.
//      type Backend = Wgpu<AutoGraphicsApi, f32>;
// ─────────────────────────────────────────────────────────────


// ─────────────────────────────────────────────────────────────
// Generic “engine” — now truly backend-agnostic
// ─────────────────────────────────────────────────────────────
fn run_inference<B, F>(
    mut f      : F,
    device     : <B as BurnBackend>::Device,
    query      : Vec<SCItemRaw>,
    batch_size : usize,
) -> Vec<f32>
where
    B: BurnBackend + 'static,
    B::FloatElem: ToPrimitive,      // ← was Elem, now correct
    F: FnMut(Tensor<B, 2>) -> Tensor<B, 2>,
{
    let dataset = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    let loader  = DataLoaderBuilder::new(SCBatcher::<B>::new(device))
        .batch_size(batch_size)
        .build(dataset);

    loader
        .iter()
        .flat_map(|batch| {
            f(batch.counts)
                .to_data()
                .value
                .into_iter()
                .map(|x| x.to_f32().unwrap())
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────
// Public API — generic over any backend `B`
// ─────────────────────────────────────────────────────────────
pub fn infer<B>(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
    device : <B as BurnBackend>::Device,
) -> Vec<f32>
where
    B: BurnBackend + 'static,
    B::FloatElem: ToPrimitive,      // ← same change here
{
    let bs = batch_size.unwrap_or(64);

    match net {
        NetKind::Mlr => {
            let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                           .load(model_path.into())
                           .expect("load MLR weights");
            let model = MlrCfg::new(num_classes)
                           .init(num_features)
                           .load_record(rec);

            run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
        }

        NetKind::Ann { hidden } => {
            let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                           .load(model_path.into())
                           .expect("load ANN-1L weights");
            let model = AnnCfg::new(num_classes, 0, hidden, 0.0)
                           .init(num_features)
                           .load_record(rec);

            run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
        }

        NetKind::Ann2 { hidden1, hidden2 } => {
            let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                           .load(model_path.into())
                           .expect("load ANN-2L weights");
            let model = Ann2Cfg::new(num_classes, 0, hidden1, hidden2, 0.0)
                           .init(num_features)
                           .load_record(rec);

            run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
        }
    }
}


// GPU (WGPU)
pub fn infer_wgpu(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
) -> Vec<f32> {
    use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
    type B = Wgpu<AutoGraphicsApi, f32>;
    infer::<B>(model_path,
        net,
        num_classes,
        num_features,
        query,
        batch_size, WgpuDevice::default())
}

// CPU (NdArray)
pub fn infer_nd(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
) -> Vec<f32> {
    use burn::backend::ndarray::{NdArray, NdArrayDevice}; 
    type B = NdArray<f32>;
    infer::<B>(model_path,
        net,
        num_classes,
        num_features,
        query,
        batch_size, NdArrayDevice::default())
}

// candle
pub fn infer_candle(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
) -> Vec<f32> {
    use burn::backend::candle::{Candle, CandleDevice}; 
    type B = Candle<f32>;
    infer::<B>(model_path,
        net,
        num_classes,
        num_features,
        query,
        batch_size, CandleDevice::default())
}



// You can add Candle, tch, etc. the exact same way.


// use crate::common::*;
// use crate::scrna_mlr::ModelConfig as MLR_ModelConfig;
// use crate::scrna_ann::ModelConfig as ANN_ModelConfig;
// use crate::scrna_ann2l::ModelConfig as ANN_2_ModelConfig;

// use num_traits::ToPrimitive;

// use burn::{
//     backend::wgpu::{WgpuDevice, Wgpu, AutoGraphicsApi},
//     data::{dataloader::DataLoaderBuilder, dataset::InMemDataset, dataset::transform::MapperDataset},
//     record::{NamedMpkFileRecorder, FullPrecisionSettings, Recorder},
//     module::Module
// };

// pub fn infer_helper_mlr(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, batch_size: Option<usize>) -> Vec<f32>{
//     type MyBackend = Wgpu<AutoGraphicsApi, f32>;
//     let device = WgpuDevice::default();
//     let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//         .load(model_path.into())
//         .expect("Failed to load model weights");

//     // Directly initialize a new model with the loaded record/weights
//     let config_model = MLR_ModelConfig::new(num_classes);
//     let model = config_model.init(num_features).load_record(record);
//     let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
//     MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
//     // Create the batchers.
//     let batcher_query = SCBatcher::<MyBackend>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_query = DataLoaderBuilder::new(batcher_query)
//         .batch_size(batch_size.unwrap_or(64))
//         .build(query_dataset);

//     // let model_valid = model.valid();
//     let mut probs = Vec::new();

//     // Assuming dataloader_query is built
//     for batch in dataloader_query.iter() {
//         let output = model.forward(batch.counts);
//         output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
//         // let output_data = output.to_data().value;
//         // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
//     }
//     probs

// }

// pub fn infer_helper_ann(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, hidden_size: usize, batch_size: Option<usize>) -> Vec<f32>{
//     type MyBackend = Wgpu<AutoGraphicsApi, f32>;
//     let device = WgpuDevice::default();
//     let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//         .load(model_path.into())
//         .expect("Failed to load model weights");

//     // Directly initialize a new model with the loaded record/weights
//     let config_model = ANN_ModelConfig::new(num_classes, 0, hidden_size, 0.0);
//     let model = config_model.init(num_features).load_record(record);
//     let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
//     MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
//     // Create the batchers.
//     let batcher_query = SCBatcher::<MyBackend>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_query = DataLoaderBuilder::new(batcher_query)
//         .batch_size(batch_size.unwrap_or(64))
//         .build(query_dataset);

//     // let model_valid = model.valid();
//     let mut probs = Vec::new();

//     // Assuming dataloader_query is built
//     for batch in dataloader_query.iter() {
//         let output = model.forward(batch.counts);
//         output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
//         // let output_data = output.to_data().value;
//         // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
//     }
//     probs

// }


// pub fn infer_helper_ann2l(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, hidden_size1: usize, hidden_size2: usize, batch_size: Option<usize>) -> Vec<f32>{
//     type MyBackend = Wgpu<AutoGraphicsApi, f32>;
//     let device = WgpuDevice::default();
//     let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//         .load(model_path.into())
//         .expect("Failed to load model weights");

//     // Directly initialize a new model with the loaded record/weights
//     let config_model = ANN_2_ModelConfig::new(num_classes, 0, hidden_size1, hidden_size2, 0.0);
//     let model = config_model.init(num_features).load_record(record);
//     let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
//     MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
//     // Create the batchers.
//     let batcher_query = SCBatcher::<MyBackend>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_query = DataLoaderBuilder::new(batcher_query)
//         .batch_size(batch_size.unwrap_or(64))
//         .build(query_dataset);

//     // let model_valid = model.valid();
//     let mut probs = Vec::new();

//     // Assuming dataloader_query is built
//     for batch in dataloader_query.iter() {
//         let output = model.forward(batch.counts);
//         output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
//         // let output_data = output.to_data().value;
//         // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
//     }
//     probs

// }