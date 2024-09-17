
// use core::num;

// use std::clone;

use crate::common::*;
use crate::scrna_mlr::{Model, ModelConfig};

use burn::{
    backend::Autodiff,
    backend::wgpu::{WgpuDevice, Wgpu, AutoGraphicsApi},
    data::dataloader::batcher::Batcher,
    // module::Module,
    // nn::{Linear, LinearConfig, ReLU},
    record::{NamedMpkFileRecorder, FullPrecisionSettings, Recorder},
    // tensor::{backend::Backend, Tensor},
};

  
//   fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
//       log_softmax(tensor, dim).exp()
//   }
  
//   fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
//       tensor.clone() - tensor.exp().sum_dim(dim).log()
//   }


// #[derive(Module, Debug)]
// pub struct Model<B: Backend> {
//     linear1: Linear<B>,
//     activation: ReLU,
// }


// impl<B: Backend> Model<B> {
//     // Returns the initialized model using the recorded weights.
//     pub fn init_with<B: Backend>(&self, no_features: usize, record: ModelRecord<B>) -> Model<B> {
//         Model {
//             linear1: LinearConfig::new(no_features, self.num_classes).init_with(record.linear1),
//             activation: ReLU::new(),
//         }
//     }

//     /// Returns the dummy model with randomly initialized weights.
//     pub fn new(device: &Device<B>) -> Model<B> {
//         let l1 = LinearConfig::new(10, 64).init(device);
//         let l2 = LinearConfig::new(64, 2).init(device);
//         Model {
//             linear1: l1,
//             activation: ReLU::new(),
//         }
//     }
// }



pub fn infer_helper(model_path: String, num_classes: usize, num_features: usize, query: Vec<SCItemRaw>) -> (Vec<i32>, Vec<Vec<f32>>){
    type MyBackend = Wgpu<AutoGraphicsApi, f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = WgpuDevice::default();
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path.into())
        .expect("Should be able to load the model weights from the provided file");

    // Directly initialize a new model with the loaded record/weights
    let config_model = ModelConfig::new(num_classes);
    let model: Model<MyAutodiffBackend> = config_model.init_with(num_features, record);
    let mut prediction: Vec<i32> = Vec::new();
    let mut probs: Vec<Vec<f32>> = Vec::new();
    for item in query {
        let batcher = SCBatcher::new(device.clone());
        let batch = batcher.batch(vec![map_raw(&item)]);
        let output = &model.forward(batch.counts);
        // eprintln!("{:?}", output);
        // probs.push(softmax(output.clone(),  num_classes).into_scalar().try_into().unwrap());
        // probs.push(output.clone().into_scalar().try_into().unwrap());
        // println!("Output shape: {:?}", output.clone().shape());
        probs.push(output.clone().squeeze::<1>(0).into_data().value.iter().cloned().collect::<Vec<f32>>());
        prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
    }
    (prediction, probs)
}