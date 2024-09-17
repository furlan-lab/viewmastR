//  #![allow(unused_imports)]
use crate::common::*;
use crate::scrna_mlr::ModelConfig;
use crate::common::{SCBatcher, map_raw};

use burn::{
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{NamedMpkFileRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::Backend,
};

use crate::scrna_mlr::ModelRecord;

pub fn infer_helper<B: Backend>(model_path: String, 
                                num_classes: usize, 
                                num_features: usize, 
                                query: Vec<SCItemRaw>, 
                                device: B::Device) -> (Vec<i32>, Vec<Vec<f32>>)
                                where i32: From<<B as Backend>::IntElem>
                                {
    let record: ModelRecord<B>  = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into(), &device)
        .expect("Should be able to load the model weights from the provided file");

    // Directly initialize a new model with the loaded record/weights
    let config_model = ModelConfig::new(num_classes);
    let model = config_model.init(num_features, &device).load_record(record);
    let mut prediction: Vec<i32> = Vec::new();
    let mut probs: Vec<Vec<f32>> = Vec::new();
    for item in query {
        let batcher = SCBatcher::new(device.clone());
        let batch = batcher.batch(vec![map_raw(&item)]);
        let output = &model.forward(batch.counts);
        probs.push(output.clone().squeeze::<1>(0).into_data().bytes.iter().map(|&x| x as f32).collect::<Vec<f32>>());
        prediction.push(integer_conversion::<B>(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar()).unwrap());
    }
    (prediction, probs)
}