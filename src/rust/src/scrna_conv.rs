
// use burn::{
//     backend::Autodiff,
//     backend::wgpu::Wgpu,
//     backend::wgpu::AutoGraphicsApi,
//     config::Config,
//     data::{dataloader::{DataLoaderBuilder, Dataset, batcher::Batcher}, dataset::{InMemDataset, transform::{Mapper,MapperDataset}}},
//     module::{Module, AutodiffModule},
//     nn::{
//         loss::CrossEntropyLoss,
//         conv::{Conv2d, Conv2dConfig},
//         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
//         Linear, LinearConfig, ReLU, Dropout, DropoutConfig,
//     },
//     optim::{Optimizer, AdamConfig, GradientsParams},
//     record::CompactRecorder,
//     tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
//     train::{ClassificationOutput, TrainStep, ValidStep, TrainOutput},
// };


// use std::result::Result;
// use std::vec::Vec;
// // use serde::Deserialize;
// use crate::scrna_mlr::SCItemRaw;


// // const FEATURE_NO: usize = 3600;

// // #[derive(Deserialize, Debug, Clone, PartialEq, TryFromRobj)]
// // #[derive(Deserialize, Debug, Clone, PartialEq)]
// // pub struct SCItemRaw {
// //     // pub counts: Vec<f64>, 
// //     pub data: Vec<f32>, 
// //     pub target: i32,
// // }

// #[derive(Debug, Clone)]
// pub struct SCItem {
//     // pub counts: Vec<f64>, 
//     pub counts: [[f32; 60]; 60], 
//     pub label: i32,
// }


// pub struct SCBatcher<B: Backend> {
//     device: B::Device,
// }

// impl<B: Backend> SCBatcher<B> {
//     pub fn new(device: B::Device) -> Self {
//         Self { device }
//     }
// }



// #[derive(Clone, Debug)]
// pub struct SCBatch<B: Backend> {
//     pub counts: Tensor<B, 3>,
//     pub targets: Tensor<B, 1, Int>,
// }

// // impl<B: Backend> Batcher<SCItem, SCBatch<B>> for SCBatcher<B>  where Data<<B as Backend>::FloatElem, 1>: From<Data<f32, 1>>, Data<<B as Backend>::IntElem, 1>: From<Data<<B as Backend>::FloatElem, 1>> {
// impl<B: Backend> Batcher<SCItem, SCBatch<B>> for SCBatcher<B>  {

//     fn batch(&self, items: Vec<SCItem>) -> SCBatch<B> {
//         let counts = items
//             .iter()
//             .map(|item| Data::<f32, 2>::from(item.counts))
//             .map(|data| Tensor::<B, 2>::from_data(data.convert()))
//             .map(|tensor| tensor.reshape([1, 60, 60]))
//             // Normalize: make between [0,1] and make the mean=0 and std=1
//             // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
//             // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
//             // .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
//             .collect();

//         let targets = items
//             .iter()
//             .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i32).elem()])))
//             .collect();

//         let counts = Tensor::cat(counts, 0).to_device(&self.device);
//         let targets = Tensor::cat(targets, 0).to_device(&self.device);

//         SCBatch { counts, targets }
//     }
// }


// #[derive(Clone, Debug)]
// pub struct MyBatch<B: Backend> {
//     pub images: Tensor<B, 3>,
//     pub targets: Tensor<B, 1, Int>,
// }


// #[derive(Module, Debug)]
// pub struct Model<B: Backend> {
//     conv1: Conv2d<B>,
//     conv2: Conv2d<B>,
//     pool: AdaptiveAvgPool2d,
//     dropout: Dropout,
//     linear1: Linear<B>,
//     linear2: Linear<B>,
//     activation: ReLU,
// }


// #[derive(Config, Debug)]
// pub struct ModelConfig {
//     num_classes: usize,
//     hidden_size: usize,
//     #[config(default = "0.5")]
//     dropout: f64,
// }


// impl ModelConfig {
//     // Returns the initialized model using the recorded weights.
//     // pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
//     //     Model {
//     //         conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
//     //         conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
//     //         pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//     //         activation: ReLU::new(),
//     //         linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
//     //         linear2: LinearConfig::new(self.hidden_size, self.num_classes)
//     //             .init_with(record.linear2),
//     //         dropout: DropoutConfig::new(self.dropout).init(),
//     //     }
//     // }
//     pub fn init<B: Backend>(&self) -> Model<B> {
//         Model {
//             conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
//             conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
//             pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
//             dropout: DropoutConfig::new(self.dropout).init(),
//         }
//     }
// }

// impl<B: Backend> Model<B> {
//     /// # Shapes
//     ///   - Images [batch_size, height, width]
//     ///   - Output [batch_size, num_classes]
//     pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
//         let [batch_size, height, width] = images.dims();

//         // Create a channel at the second dimension.
//         let x = images.reshape([batch_size, 1, height, width]);


//         let x = self.conv1.forward(x); // [batch_size, 8, _, _]
//         let x = self.dropout.forward(x);
//         let x = self.conv2.forward(x); // [batch_size, 16, _, _]
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);

//         let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
//         let x = x.reshape([batch_size, 16 * 8 * 8]);
//         let x = self.linear1.forward(x);
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);

//         self.linear2.forward(x) // [batch_size, num_classes]
//     }
// }

// impl<B: Backend> Model<B> {
//     pub fn forward_classification(
//         &self,
//         images: Tensor<B, 3>,
//         targets: Tensor<B, 1, Int>,
//     ) -> ClassificationOutput<B> {
//         let output = self.forward(images);
//         let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

//         ClassificationOutput::new(loss, output, targets)
//     }
// }

// impl<B: Backend + burn::tensor::backend::AutodiffBackend> TrainStep<SCBatch<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: SCBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let item = self.forward_classification(batch.counts, batch.targets);

//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

// impl<B: Backend> ValidStep<SCBatch<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: SCBatch<B>) -> ClassificationOutput<B> {
//         self.forward_classification(batch.counts, batch.targets)
//     }
// }

// #[derive(Config)]
// pub struct TrainingConfig {
//     pub model: ModelConfig,
//     pub optimizer: AdamConfig,
//     #[config(default = 10)]
//     pub num_epochs: usize,
//     #[config(default = 64)]
//     pub batch_size: usize,
//     #[config(default = 4)]
//     pub num_workers: usize,
//     #[config(default = 42)]
//     pub seed: u64,
//     #[config(default = 1.0e-4)]
//     pub learning_rate: f64,
// }



// // type MappedLocalDataset = MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw>;


// pub struct LocalCountstoMatrix;

// impl Mapper<SCItemRaw, SCItem> for LocalCountstoMatrix {
//     /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
//     fn map(&self, item: &SCItemRaw) -> SCItem {
//         let counts = &item.data;

//         // Convert the image to a 2D array of floats.
//         let mut counts_array = [[0f32; 60]; 60];
//         for (i, pixel) in counts.iter().enumerate() {
//             let x = i % 60;
//             let y = i / 60;
//             counts_array[y][x] = *pixel as f32;
//         }

//         SCItem {
//             counts: counts_array,
//             label: item.target,
//         }
//     }
// }


// pub fn map_raw(item: &SCItemRaw) -> SCItem {
//     let counts = &item.data;

//     // Convert the image to a 2D array of floats.
//     let mut counts_array = [[0f32; 60]; 60];
//     for (i, pixel) in counts.iter().enumerate() {
//         let x = i % 60;
//         let y = i / 60;
//         counts_array[y][x] = *pixel as f32;
//     }

//     SCItem {
//         counts: counts_array,
//         label: item.target,
//     }
// }



// pub struct SCLocalDataset {
//     pub dataset: dyn Dataset<SCItem>,
// }




// impl Dataset<SCItem> for SCLocalDataset {
//     fn get(&self, index: usize) -> Option<SCItem> {
//         self.dataset.get(index)
//         // None
//     }

//     fn len(&self) -> usize {
//         self.dataset.len()
//     }
// }


// // pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: SCItem) {
// //     let config =
// //         TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("A config exists");
// //     let record = CompactRecorder::new()
// //         .load(format!("{artifact_dir}/model").into())
// //         .expect("Failed to load trained model");

// //     let model = config.model.init_with::<B>(record).to_device(&device);

// //     let label = item.label;
// //     let batcher = SCBatcher::new(device);
// //     let batch = batcher.batch(vec![item]);
// //     let output = model.forward(batch.counts);
// //     let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

// //     println!("Predicted {} Expected {}", predicted, label);
// // }




// #[derive(Config)]
// pub struct SCTrainingConfig {
//     #[config(default = 10)]
//     pub num_epochs: usize,
//     #[config(default = 64)]
//     pub batch_size: usize,
//     #[config(default = 4)]
//     pub num_workers: usize,
//     #[config(default = 42)]
//     pub seed: u64,
//     #[config(default = 1e-4)]
//     pub lr: f64,
//     pub model: ModelConfig,
//     pub optimizer: AdamConfig,
// }



// pub fn run_custom(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, hidden_size: usize)->Vec<i32> {
//     let artifact_dir = "/tmp/sc_local";
//     let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
//     let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
//     // let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
//     type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
//     type MyAutodiffBackend = Autodiff<MyBackend>;
//     let device = burn::backend::wgpu::WgpuDevice::default();
//     // Create the configuration.
//     let config_model = ModelConfig::new(num_classes, hidden_size);
//     let config_optimizer = AdamConfig::new();
//     let config = SCTrainingConfig::new(config_model, config_optimizer);

//     MyAutodiffBackend::seed(config.seed);


//     // Create the model and optimizer.
//     let mut model: Model<MyAutodiffBackend> = config.model.init();
//     let mut optim = config.optimizer.init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();

//     // Create the batcher.
//     let batcher_train = SCBatcher::<MyAutodiffBackend>::new(device.clone());
//     let batcher_valid = SCBatcher::<MyBackend>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(train_dataset);

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(test_dataset);

//     // Iterate over our training and validation loop for X epochs.
//     for epoch in 1..config.num_epochs + 1 {
//         // Implement our training loop.
//         for (iteration, batch) in dataloader_train.iter().enumerate() {
//             let output = model.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
//             let accuracy = accuracy(output, batch.targets);

//             eprintln!(
//                 "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
//                 epoch,
//                 iteration,
//                 loss.clone().into_scalar(),
//                 accuracy,
//             );

//             // Gradients for the current backward pass
//             let grads = loss.backward();
//             // Gradients linked to each parameter of the model.
//             let grads = GradientsParams::from_grads(grads, &model);
//             // Update the model using the optimizer.
//             model = optim.step(config.lr, model, grads);
//         }

//         // Get the model without autodiff.
//         let model_valid = model.valid();

//         // Implement our validation loop.
//         for (iteration, batch) in dataloader_test.iter().enumerate() {
//             let output = model_valid.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
//             let accuracy = accuracy(output, batch.targets);

//             eprintln!(
//                 "[Valid - Epoch {} - Iteration {}] Loss {} | Accuracy {}",
//                 iteration,
//                 epoch,
//                 loss.clone().into_scalar(),
//                 accuracy,
//             );
//         }
//     }

//     let mut prediction: Vec<i32> = vec![];
//     for item in query {
//         let batcher = SCBatcher::new(device.clone());
//         let batch = batcher.batch(vec![map_raw(&item)]);
//         let output = &model.forward(batch.counts);
//         prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar());
//     }
//     let _ = &model.clone()
//         .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
//         .expect("Failed to save trained model");
//     return prediction;

// }

// /// Create out own accuracy metric calculation.
// fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
//     let predictions = output.argmax(1).squeeze(1);
//     let num_predictions: usize = targets.dims().iter().product();
//     let num_corrects = predictions.equal(targets).int().sum().into_scalar();

//     num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
// }
