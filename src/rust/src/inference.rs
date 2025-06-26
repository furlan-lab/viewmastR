//! inference.rs   (only the parts that changed are shown)

use crate::common::*;
use crate::scrna_mlr::{Model as MlrModel, ModelConfig as MlrCfg};
use crate::scrna_ann::{Model as AnnModel, ModelConfig as AnnCfg};
use crate::scrna_ann2l::{Model as Ann2Model, ModelConfig as Ann2Cfg};

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::MapperDataset, InMemDataset},
    },
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{backend::Backend, Tensor},
};

type B = Wgpu<f32, i32>;          // our single backend alias
type Dev = WgpuDevice;            // shorthand

/* ------------------------------------------------------------------------- */
/* 1. A tiny trait so the generic core is allowed to call `forward`          */
/* ------------------------------------------------------------------------- */
trait Infer<Bk: Backend> {
    fn infer(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2>;
}

impl<Bk: Backend> Infer<Bk> for MlrModel<Bk> {
    fn infer(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> { self.forward(x) }
}
impl<Bk: Backend> Infer<Bk> for AnnModel<Bk>  {
    fn infer(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> { self.forward(x) }
}
impl<Bk: Backend> Infer<Bk> for Ann2Model<Bk> {
    fn infer(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> { self.forward(x) }
}

/* ------------------------------------------------------------------------- */
/* 2. The one shared engine                                                  */
/* ------------------------------------------------------------------------- */
fn infer_with_builder<M, Build>(
    path: &str,
    build_model: Build,
    query: Vec<SCItemRaw>,
    batch: usize,
) -> Vec<f32>
where
    M: Module<B> + Infer<B>,
    Build: FnOnce(&Dev) -> M,
{
    // device ---------------------------------------------------------------
    let device = Dev::default();

    // model skeleton -------------------------------------------------------
    let mut model = build_model(&device);

    // --- A) load the weights  --------------------------------------------
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(path.into(), &device)
        .expect("load weights");
    model = model.load_record(record);

    // dataloader -----------------------------------------------------------
    let ds = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    let loader = DataLoaderBuilder::new(SCBatcher::<B>::new(device.clone()))
        .batch_size(batch)
        .build(ds);

    // inference loop -------------------------------------------------------
    let mut probs = Vec::new();
    for batch in loader.iter() {
        probs.extend(
            model.infer(batch.counts)
                .to_data()
                .iter::<f32>()
        );
    }
    probs
}

/* ------------------------------------------------------------------------- */
/* 3. Thin, model-specific façades                                           */
/* ------------------------------------------------------------------------- */
pub fn infer_helper_mlr(
    path: String,
    n_classes: usize,
    n_feats: usize,
    query: Vec<SCItemRaw>,
    batch: Option<usize>,
) -> Vec<f32> {
    infer_with_builder::<MlrModel<B>, _>(
        &path,
        |device| {
            MlrCfg::new(n_classes)
                .init(n_feats, device.clone())            // `init` needs the device
        },
        query,
        batch.unwrap_or(64),
    )
}

pub fn infer_helper_ann(
    path: String,
    n_classes: usize,
    n_feats: usize,
    query: Vec<SCItemRaw>,
    hidden: usize,
    batch: Option<usize>,
) -> Vec<f32> {
    infer_with_builder::<AnnModel<B>, _>(
        &path,
        |device| {
            //  ANN `init` *does not* take the device
            AnnCfg::new(n_classes, 0, hidden, 0.0).init(n_feats, device.clone())
        },
        query,
        batch.unwrap_or(64),
    )
}

pub fn infer_helper_ann2l(
    path: String,
    n_classes: usize,
    n_feats: usize,
    query: Vec<SCItemRaw>,
    h1: usize,
    h2: usize,
    batch: Option<usize>,
) -> Vec<f32> {
    infer_with_builder::<Ann2Model<B>, _>(
        &path,
        |device| {
            Ann2Cfg::new(n_classes, 0, h1, h2, 0.0).init(n_feats, device.clone()) // same: no device
        },
        query,
        batch.unwrap_or(64),
    )
}


// use crate::common::*;
// use crate::scrna_mlr::ModelConfig as MLR_ModelConfig;
// use crate::scrna_ann::ModelConfig as ANN_ModelConfig;
// use crate::scrna_ann2l::ModelConfig as ANN_2_ModelConfig;

// // use num_traits::ToPrimitive;

// use burn::{
//     backend::wgpu::{WgpuDevice, Wgpu},
//     data::{dataloader::DataLoaderBuilder, dataset::InMemDataset, dataset::transform::MapperDataset},
//     record::{NamedMpkFileRecorder, FullPrecisionSettings, Recorder},
//     module::Module
// };
// // use serde::de;

// pub fn infer_helper_mlr(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, batch_size: Option<usize>) -> Vec<f32>{
//     type MyBackend = Wgpu<f32, i32>;
//     let device = WgpuDevice::default();
//     let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//         .load(model_path.into(), &device)
//         .expect("Failed to load model weights");

//     // Directly initialize a new model with the loaded record/weights
//     let config_model = MLR_ModelConfig::new(num_classes);
//     let model = config_model.init(num_features, device.clone()).load_record(record);
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
//         for val in output.to_data().iter::<f32>() {
//             probs.push(val);
//         }
//         // output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
//         // let output_data = output.to_data().value;
//         // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
//     }
//     probs

// }

// pub fn infer_helper_ann(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, hidden_size: usize, batch_size: Option<usize>) -> Vec<f32>{
//     type MyBackend = Wgpu<f32, i32>;
//     let device = WgpuDevice::default();
//     let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//         .load(model_path.into(), &device.clone())
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
//         for val in output.to_data().iter::<f32>() {
//             probs.push(val);
//         }
//         // output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
//         // let output_data = output.to_data().value;
//         // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
//     }
//     probs

// }


// pub fn infer_helper_ann2l(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, hidden_size1: usize, hidden_size2: usize, batch_size: Option<usize>) -> Vec<f32>{
//     type MyBackend = Wgpu<f32, i32>;
//     let device = WgpuDevice::default();
//     let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
//         .load(model_path.into(), &device)
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
//         for val in output.to_data().iter::<f32>() {
//             probs.push(val);
//         }
//         // output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
//         // let output_data = output.to_data().value;
//         // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
//     }
//     probs

// }

