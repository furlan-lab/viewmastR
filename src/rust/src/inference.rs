
use crate::common::*;
use crate::scrna_mlr::ModelConfig as MLR_ModelConfig;
use crate::scrna_ann::ModelConfig as ANN_ModelConfig;
use crate::scrna_ann2l::ModelConfig as ANN_2_ModelConfig;
use crate::scrna_mlr::Model as MLRModel;
use num_traits::ToPrimitive;

use burn::{
    backend::wgpu::{WgpuDevice, Wgpu, AutoGraphicsApi},
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset, dataset::transform::MapperDataset},
    record::{NamedMpkFileRecorder, FullPrecisionSettings, Recorder, Record},
    module::Module
};

pub fn infer_helper_mlr(model_path: String, num_classes: usize, feature_names_vec: Vec<String>, query: Vec<SCItemRaw>, batch_size: Option<usize>) -> Vec<f32>{
    type MyBackend = Wgpu<AutoGraphicsApi, f32>;
    let device = WgpuDevice::default();
    // let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
    //     .load(model_path.into())
    //     .expect("Failed to load model weights");
    let model: MLRModel<MyBackend> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into())
        .expect("Failed to load model weights");
    let feature_names = model.input_feature_names;
    // let num_classes   = model.num_classes;  
    // Directly initialize a new model with the loaded record/weights
    // let config_model = MLR_ModelConfig::new(num_classes, feature_names_vec);
    // let model = config_model.init().load_record(record);
    // let model = MLRModel::init(&device).load_record(record);
    let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
    MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    // Create the batchers.
    let batcher_query = SCBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_query = DataLoaderBuilder::new(batcher_query)
        .batch_size(batch_size.unwrap_or(64))
        .build(query_dataset);

    // let model_valid = model.valid();
    let mut probs = Vec::new();

    // Assuming dataloader_query is built
    for batch in dataloader_query.iter() {
        let output = model.forward(batch.counts);
        output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
        // let output_data = output.to_data().value;
        // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
    }
    probs

}

pub fn infer_helper_ann(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, hidden_size: usize, batch_size: Option<usize>) -> Vec<f32>{
    type MyBackend = Wgpu<AutoGraphicsApi, f32>;
    let device = WgpuDevice::default();
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into())
        .expect("Failed to load model weights");

    // Directly initialize a new model with the loaded record/weights
    let config_model = ANN_ModelConfig::new(num_classes, 0, hidden_size, 0.0);
    let model = config_model.init(num_features).load_record(record);
    let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
    MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    // Create the batchers.
    let batcher_query = SCBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_query = DataLoaderBuilder::new(batcher_query)
        .batch_size(batch_size.unwrap_or(64))
        .build(query_dataset);

    // let model_valid = model.valid();
    let mut probs = Vec::new();

    // Assuming dataloader_query is built
    for batch in dataloader_query.iter() {
        let output = model.forward(batch.counts);
        output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
        // let output_data = output.to_data().value;
        // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
    }
    probs

}


pub fn infer_helper_ann2l(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>, hidden_size1: usize, hidden_size2: usize, batch_size: Option<usize>) -> Vec<f32>{
    type MyBackend = Wgpu<AutoGraphicsApi, f32>;
    let device = WgpuDevice::default();
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into())
        .expect("Failed to load model weights");

    // Directly initialize a new model with the loaded record/weights
    let config_model = ANN_2_ModelConfig::new(num_classes, 0, hidden_size1, hidden_size2, 0.0);
    let model = config_model.init(num_features).load_record(record);
    let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
    MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    // Create the batchers.
    let batcher_query = SCBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_query = DataLoaderBuilder::new(batcher_query)
        .batch_size(batch_size.unwrap_or(64))
        .build(query_dataset);

    // let model_valid = model.valid();
    let mut probs = Vec::new();

    // Assuming dataloader_query is built
    for batch in dataloader_query.iter() {
        let output = model.forward(batch.counts);
        output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
        // let output_data = output.to_data().value;
        // probs.extend(output_data.iter().map(|x| x.to_f32().unwrap()));
    }
    probs

}