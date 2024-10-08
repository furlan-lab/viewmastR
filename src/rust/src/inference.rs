
use crate::common::*;
use crate::scrna_mlr::ModelConfig;

use num_traits::ToPrimitive;

use burn::{
    backend::wgpu::{WgpuDevice, Wgpu, AutoGraphicsApi},
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset, dataset::transform::MapperDataset},
    record::{NamedMpkFileRecorder, FullPrecisionSettings, Recorder},
    module::Module
};

pub fn infer_helper(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>) -> Vec<f32>{
    type MyBackend = Wgpu<AutoGraphicsApi, f32>;
    let device = WgpuDevice::default();
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into())
        .expect("Should be able to load the model weights from the provided file");

    // Directly initialize a new model with the loaded record/weights
    let config_model = ModelConfig::new(num_classes);
    let model = config_model.init(num_features).load_record(record);
    let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
    MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    // Create the batchers.
    let batcher_query = SCBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_query = DataLoaderBuilder::new(batcher_query)
        .batch_size(64)
        .build(query_dataset);

    // let model_valid = model.valid();
    let mut probs = Vec::new();

    // Assuming dataloader_query is built
    for batch in dataloader_query.iter() {
        let output = model.forward(batch.counts);
        output.to_data().value.iter().for_each(|x| probs.push(x.to_f32().expect("failed to unwrap probs")));
    }
    probs

}