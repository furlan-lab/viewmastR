
use burn::{
    // backend::{candle::Candle, ndarray::{NdArray, NdArrayDevice}, wgpu::{Wgpu, WgpuDevice}, Autodiff},
    config::Config,
    data::{dataloader::{batcher::Batcher, DataLoaderBuilder, Dataset}, dataset::{transform::MapperDataset, InMemDataset}},
    module::{AutodiffModule, Module},
    nn::{
        loss::{CrossEntropyLoss, CrossEntropyLossConfig},
        Linear, LinearConfig, Relu,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{backend::{AutodiffBackend, Backend},Int, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};



use std::result::Result;
use std::vec::Vec;
// use std::convert::TryInto;
use crate::pb::ProgressBar;
use std::time::Instant;
use crate::common::*;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
}


#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
}


impl ModelConfig {
    // pub fn init_with<B: Backend>(&self, no_features: usize, record: ModelRecord<B>) -> Model<B> {
    //     Model {
    //         linear1: LinearConfig::new(no_features, self.num_classes).init_with(record.linear1),
    //         activation: Relu::new(),
    //     }
    // }
    pub fn init<B: Backend>(&self, no_features: usize, device: &B::Device) -> Model<B> {
        Model {
            activation: Relu::new(),
            linear1: LinearConfig::new(no_features, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, dim] = data.dims();

        let x = data.reshape([batch_size, dim]);
        self.linear1.forward(x)
    }
}

// impl<B: Backend> Model<B> {
//     pub fn forward_classification(
//         &self,
//         data: Tensor<B, 2>,
//         targets: Tensor<B, 1, Int>,
//     ) -> ClassificationOutput<B> {
//         let output = self.forward(data);
//         let loss = CrossEntropyLoss::new()
//                                     .init(&output.device())
//                                     .forward(output.clone(), targets.clone());

//         ClassificationOutput::new(loss, output, targets)
//     }
    
// }

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: Backend + burn::tensor::backend::AutodiffBackend> TrainStep<SCBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: SCBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.counts, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<SCBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: SCBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.counts, batch.targets)
    }
}



#[derive(Config)]
struct SCTrainingConfig {
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}


pub fn train<B: Backend, AB: AutodiffBackend>(
    train: Vec<SCItemRaw>,
    test: Vec<SCItemRaw>,
    query: Vec<SCItemRaw>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
    device: B::Device //) -> ModelRExport {
) -> ModelRExport where 
                    f64: From<<B as Backend>::FloatElem>, 
                    i32: From<<B as Backend>::IntElem>,
                    i64: From<<B as Backend>::IntElem> {
    
    // Avoid calling `.unwrap()` repeatedly
    let no_features: usize = train.get(0).expect("Train data is empty").data.len();

    // Dataset creation once, no need to clone matrices multiple times.
    let train_dataset = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
    let test_dataset = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);

    let num_batches_train: usize = train_dataset.len();
    let artifact_dir = directory.clone().unwrap_or_else(|| panic!("Folder not found: {:?}", directory));

    // Model and optimizer initialization
    let config = SCTrainingConfig::new(
        num_epochs,
        learning_rate,
        ModelConfig::new(num_classes),
        AdamConfig::new(),
    );

    let mut model: Model<B> = config.model.init::<B>(no_features, &device);
    let mut optim       = config.optimizer.init::<AB, dyn AutodiffModule<AB>>();

    // Create batchers and dataloaders
    let batcher_train = SCBatcher::new(device.clone());
    let batcher_valid = SCBatcher::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let mut train_accuracy = ModelAccuracy::new();
    let mut test_accuracy = ModelAccuracy::new();

    // Initialize progress bar parameters
    let num_iterations = (num_batches_train / config.batch_size) as u32;
    let progress_bar_length = 40;
    let eta = false;

    // History tracking
    let mut train_history = History::new();
    let mut test_history = History::new();

    let start = Instant::now();

    // Training and validation loop
    for epoch in 1..=config.num_epochs {
        train_accuracy.epoch_reset(epoch);
        test_accuracy.epoch_reset(epoch);

        if verbose {
            eprintln!("[Epoch {} progress...]", epoch);
        }

        let mut bar = ProgressBar::default(num_iterations, progress_bar_length, eta);

        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            if verbose {
                bar.update();
            }

            let output = model.forward(batch.counts);

            // Loss calculation and accuracy tracking
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());

            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            let loss_scalar = loss.clone().into_scalar(); // Calculate scalar loss once here

            train_accuracy.batch_update(num_corrects.into(), num_predictions, loss_scalar.into());

            // Backpropagation and model update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads);
        }

        train_accuracy.epoch_update(&mut train_history);

        // Validation using non-autodiff model
        let model_valid = model.valid();

        for (_iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None, &device)
                .forward(output.clone(), batch.targets.clone());

            let loss_scalar = loss.into_scalar(); // Avoid multiple conversions
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();

            test_accuracy.batch_update(num_corrects.into(), num_predictions, loss_scalar.into());
        }

        test_accuracy.epoch_update(&mut test_history);

        if verbose {
            emit_metrics(&train_accuracy, &test_accuracy);
        }
    }

    // Elapsed time
    let tduration = start.elapsed();

    // Prediction phase
    let mut prediction: Vec<i32> = Vec::with_capacity(query.len()); // Pre-allocate memory
    let batcher = SCBatcher::new(device.clone());

    for item in query {
        let batch = batcher.batch(vec![map_raw(&item)]);
        let output = model.forward(batch.counts);
        let pred = output.argmax(1).into_scalar(); // Skip unnecessary flattening
        prediction.push(pred.try_into().unwrap());
    }

    // Save model
    model.clone()
        .save_file(
            format!("{artifact_dir}/model"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Failed to save trained model");

    // Collect and return the predictions
    ModelRExport {
        lr: config.lr,
        hidden_size: vec![0],
        batch_size: config.batch_size,
        num_epochs: config.num_epochs,
        num_workers: config.num_workers,
        seed: config.seed,
        predictions: prediction,
        train_history: train_history,
        test_history: test_history,
        training_duration: tduration.as_secs_f64(),
    }
}


// pub fn train<B: Backend>(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool, device: B::Device)->ModelRExport {

//     // type MyBackend = Wgpu<AutoGraphicsApi, f32>;
//     // type MyBackend = Wgpu<f32, i32>;
//     // type MyAutodiffBackend = Autodiff<MyBackend>;
//     // let device = WgpuDevice::default();

//     let no_features = train.first().unwrap().data.len();
//     let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
//     let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
//     let num_batches_train = train_dataset.len();
//     let artifact_dir = match directory {
//         Some(directory) => directory,
//         _ => panic!("Folder not found: {:?}", directory)
//     };

//     // Create the configuration.
//     let config_model = ModelConfig::new(num_classes);
//     let config_optimizer = AdamConfig::new();
//     let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

//     // Create the model and optimizer.
//     let mut model = config.model.init(no_features, &device);
//     let mut optim = config.optimizer.init();

//     // Create the batcher.
//     let batcher_train = SCBatcher::new(device.clone());
//     let batcher_valid = SCBatcher::new(device.clone());
//     // let batcher_query = SCBatcher::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(train_dataset);

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(test_dataset);

//     let mut train_accuracy = ModelAccuracy::new();
//     let mut test_accuracy = ModelAccuracy::new();

//     // progress bar items
//     let num_iterations = (num_batches_train / config.batch_size) as u32; 
//     let length = 40;
//     let eta = false;

//     //history stuff
//     let mut train_history: History = History::new();
//     let mut test_history: History = History::new();

//     let start = Instant::now();
//     // Iterate over our training and validation loop for X epochs.
//     for epoch in 1..config.num_epochs + 1 {
//         // Implement our training loop.
//         train_accuracy.epoch = epoch;
//         test_accuracy.epoch = epoch;
//         test_accuracy.epoch_reset(epoch);
//         train_accuracy.epoch_reset(epoch);
//         let mut bar = ProgressBar::default(num_iterations, length, eta);

//         if verbose {eprintln!("[Epoch {} progress...]", epoch)}
//         for (_iteration, batch) in dataloader_train.iter().enumerate() {
//             if verbose {bar.update()}
//             // bar.inc(1);
//             let output = model.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), batch.targets.clone());
//             // let accuracy = accuracy(output, batch.targets);
//             let loss_scalar: f64 = loss.clone().into_scalar();
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects: i32 = predictions.equal(batch.targets).int().sum().into_scalar();
//             train_accuracy.batch_update(num_corrects.into(), num_predictions, loss_scalar.into());

//             // Gradients for the current backward pass
//             let grads = loss.backward();
//             // Gradients linked to each parameter of the model.
//             let grads = GradientsParams::from_grads(grads, &model);
//             // Update the model using the optimizer.
//             model = optim.step(config.lr, model, grads);
//         }
//         train_accuracy.epoch_update(& mut train_history);
//         // bar.finish();
        
//         // Get the model without autodiff.
//         let model_valid = model.valid();

//         // Implement our validation loop.
//         for (_iteration, batch) in dataloader_test.iter().enumerate() {
//             let output = model_valid.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &device).forward(output.clone(), batch.targets.clone());
//             let loss_scalar: &f64 = &loss.into_scalar();
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects: i32 = predictions.equal(batch.targets).int().sum().into_scalar();
//             test_accuracy.batch_update(num_corrects.into(), num_predictions, (*loss_scalar).into());
//             // let accuracy = accuracy(output, batch.targets);
//         }
//         test_accuracy.epoch_update(& mut test_history);
//         if verbose {emit_metrics(&train_accuracy, &test_accuracy)}
//     }
//     let tduration = start.elapsed();

//     let mut prediction: Vec<i32> = vec![];
//     for item in query {
//         let batcher = SCBatcher::new(device.clone());
//         let batch = batcher.batch(vec![map_raw(&item)]);
//         let output = &model.forward(batch.counts);
//         prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
//     }
    
//     // save model
//     let _ = &model.clone()
//         .save_file(format!("{artifact_dir}/model"), &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
//         .expect("Failed to save trained model");

//     //collect and return the predictions
//     return ModelRExport {
//         lr: config.lr,
//         hidden_size: vec![0],
//         batch_size: config.batch_size,
//         num_epochs: config.num_epochs,
//         num_workers: config.num_workers,
//         seed: config.seed,
//         predictions: prediction,
//         train_history: train_history,
//         test_history: test_history,
//         training_duration: tduration.as_secs_f64(),
//     }
// }


// pub fn run_custom_wgpu(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool)->ModelRExport {

//     // type MyBackend = Wgpu<AutoGraphicsApi, f32>;
//     type MyBackend = Wgpu<f32, i32>;
//     type MyAutodiffBackend = Autodiff<MyBackend>;

//     let device = WgpuDevice::default();

//     let no_features = train.first().unwrap().data.len();
//     let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
//     let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
//     let num_batches_train = train_dataset.len();
//     let artifact_dir = match directory {
//         Some(directory) => directory,
//         _ => panic!("Folder not found: {:?}", directory)
//     };

//     // Create the configuration.
//     let config_model = ModelConfig::new(num_classes);
//     let config_optimizer = AdamConfig::new();
//     let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

//     // Create the model and optimizer.
//     let mut model: Model<MyAutodiffBackend> = config.model.init(no_features);
//     let mut optim = config.optimizer.init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();

//     // Create the batcher.
//     let batcher_train = SCBatcher::<MyAutodiffBackend>::new(device.clone());
//     let batcher_valid = SCBatcher::<MyBackend>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(train_dataset);

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(test_dataset);

//     let mut train_accuracy = ModelAccuracy::new();
//     let mut test_accuracy = ModelAccuracy::new();

//     // progress bar items
//     let num_iterations = (num_batches_train / config.batch_size) as u32; 
//     let length = 40;
//     let eta = false;

//     //history stuff
//     let mut train_history: History = History::new();
//     let mut test_history: History = History::new();

//     let start = Instant::now();
//     // Iterate over our training and validation loop for X epochs.
//     for epoch in 1..config.num_epochs + 1 {
//         // Implement our training loop.
//         train_accuracy.epoch = epoch;
//         test_accuracy.epoch = epoch;
//         test_accuracy.epoch_reset(epoch);
//         train_accuracy.epoch_reset(epoch);
//         let mut bar = ProgressBar::default(num_iterations, length, eta);

//         if verbose {eprintln!("[Epoch {} progress...]", epoch)}
//         for (_iteration, batch) in dataloader_train.iter().enumerate() {
//             if verbose {bar.update()}
//             // bar.inc(1);
//             let output = model.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &device).forward(output.clone(), batch.targets.clone());
//             // let accuracy = accuracy(output, batch.targets);
//             let loss_scalar = loss.clone().into_scalar();
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
//             train_accuracy.batch_update(num_corrects.into(), num_predictions, loss_scalar.into());

//             // Gradients for the current backward pass
//             let grads = loss.backward();
//             // Gradients linked to each parameter of the model.
//             let grads = GradientsParams::from_grads(grads, &model);
//             // Update the model using the optimizer.
//             model = optim.step(config.lr, model, grads);
//         }
//         train_accuracy.epoch_update(& mut train_history);
//         // bar.finish();
        
//         // Get the model without autodiff.
//         let model_valid = model.valid();

//         // Implement our validation loop.
//         for (_iteration, batch) in dataloader_test.iter().enumerate() {
//             let output = model_valid.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &device).forward(output.clone(), batch.targets.clone());
//             let loss_scalar = &loss.into_scalar();
//             // if iteration == test_len {
//             //     last_epoch_loss = loss.into_scalar();
//             // }
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
//             test_accuracy.batch_update(num_corrects.into(), num_predictions, (*loss_scalar).into());
//             // let accuracy = accuracy(output, batch.targets);
//         }
//         test_accuracy.epoch_update(& mut test_history);
//         if verbose {emit_metrics(&train_accuracy, &test_accuracy)}
//     }
//     let tduration = start.elapsed();

//     let mut prediction: Vec<i32> = vec![];
//     for item in query {
//         let batcher = SCBatcher::new(device.clone());
//         let batch = batcher.batch(vec![map_raw(&item)]);
//         let output = &model.forward(batch.counts);
//         prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
//     }
    
//     // save model
//     let _ = &model.clone()
//         .save_file(format!("{artifact_dir}/model"), &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
//         .expect("Failed to save trained model");

//     //collect and return the predictions
//     return ModelRExport {
//         lr: config.lr,
//         hidden_size: vec![0],
//         batch_size: config.batch_size,
//         num_epochs: config.num_epochs,
//         num_workers: config.num_workers,
//         seed: config.seed,
//         predictions: prediction,
//         train_history: train_history,
//         test_history: test_history,
//         training_duration: tduration.as_secs_f64(),
//     }
// }



// pub fn run_custom_candle(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool)->ModelRExport {
//     type MyBackend = Candle<f64, i64>;
//     type MyAutodiffBackend = Autodiff<MyBackend>;
//     let device = burn::backend::candle::CandleDevice::default();
    
//     let no_features = train.first().unwrap().data.len();
//     let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
//     let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
//     let num_batches_train = train_dataset.len();
//     let artifact_dir = match directory {
//         Some(directory) => directory,
//         _ => panic!("Folder not found: {:?}", directory)
//     };
    
//     // Create the configuration.
//     let config_model = ModelConfig::new(num_classes);
//     let config_optimizer = AdamConfig::new();
//     let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

//     // Create the model and optimizer.
//     let mut model: Model<MyAutodiffBackend> = config.model.init(no_features, &device);
//     let mut optim = config.optimizer.init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();

//     // Create the batcher.
//     let batcher_train = SCBatcher::<MyAutodiffBackend>::new(device.clone());
//     let batcher_valid = SCBatcher::<MyBackend>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(train_dataset);

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(test_dataset);

//     let mut train_accuracy = ModelAccuracy::new();
//     let mut test_accuracy = ModelAccuracy::new();

//     // progress bar items
//     let num_iterations = (num_batches_train / config.batch_size) as u32; 
//     let length = 40;
//     let eta = false;

//     //history stuff
//     let mut train_history: History = History::new();
//     let mut test_history: History = History::new();

//     let start = Instant::now();
//     // Iterate over our training and validation loop for X epochs.
//     for epoch in 1..config.num_epochs + 1 {
//         // Implement our training loop.
//         train_accuracy.epoch = epoch;
//         test_accuracy.epoch = epoch;
//         test_accuracy.epoch_reset(epoch);
//         train_accuracy.epoch_reset(epoch);
//         let mut bar = ProgressBar::default(num_iterations, length, eta);

//         if verbose {eprintln!("[Epoch {} progress...]", epoch)}
//         for (_iteration, batch) in dataloader_train.iter().enumerate() {
//             if verbose {bar.update()}
//             // bar.inc(1);
//             let output = model.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &device).forward(output.clone(), batch.targets.clone());
//             // let accuracy = accuracy(output, batch.targets);
//             let loss_scalar = loss.clone().into_scalar();
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
//             train_accuracy.batch_update(num_corrects, num_predictions, loss_scalar);

//             // Gradients for the current backward pass
//             let grads = loss.backward();
//             // Gradients linked to each parameter of the model.
//             let grads = GradientsParams::from_grads(grads, &model);
//             // Update the model using the optimizer.
//             model = optim.step(config.lr, model, grads);
//         }
//         train_accuracy.epoch_update(& mut train_history);
//         // bar.finish();
        
//         // Get the model without autodiff.
//         let model_valid = model.valid();

//         // Implement our validation loop.
//         for (_iteration, batch) in dataloader_test.iter().enumerate() {
//             let output = model_valid.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &device).forward(output.clone(), batch.targets.clone());
//             let loss_scalar = &loss.into_scalar();
//             // if iteration == test_len {
//             //     last_epoch_loss = loss.into_scalar();
//             // }
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
//             test_accuracy.batch_update(num_corrects, num_predictions, *loss_scalar);
//             // let accuracy = accuracy(output, batch.targets);
//         }
//         test_accuracy.epoch_update(& mut test_history);
//         if verbose {emit_metrics(&train_accuracy, &test_accuracy)}
//     }
//     let tduration = start.elapsed();

//     let mut prediction: Vec<i32> = vec![];
//     for item in query {
//         let batcher = SCBatcher::new(device.clone());
//         let batch = batcher.batch(vec![map_raw(&item)]);
//         let output = &model.forward(batch.counts);
//         prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
//     }
    
//     // save model
//     let _ = &model.clone()
//         .save_file(format!("{artifact_dir}/model"), &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
//         .expect("Failed to save trained model");

//     //collect and return the predictions
//     return ModelRExport {
//         lr: config.lr,
//         hidden_size: vec![0],
//         batch_size: config.batch_size,
//         num_epochs: config.num_epochs,
//         num_workers: config.num_workers,
//         seed: config.seed,
//         predictions: prediction,
//         train_history: train_history,
//         test_history: test_history,
//         training_duration: tduration.as_secs_f64(),
//     }
// }




// pub fn run_custom_nd(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool)->ModelRExport {
//     type MyBackend = NdArray<f64>;
//     type MyAutodiffBackend = Autodiff<MyBackend>;
//     let device = NdArrayDevice::default();
//     // if cfg!(target_os = "macos") {
//     //     device = LibTorchDevice::Mps;
//     // } else {
//     //     device = LibTorchDevice::Cuda(0);
//     // }
//     let no_features = train.first().unwrap().data.len();
//     let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
//     let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
//     let num_batches_train = train_dataset.len();
//     let artifact_dir = match directory {
//         Some(directory) => directory,
//         _ => panic!("Folder not found: {:?}", directory)
//     };
    
    
//     // let num_batches_test = train_dataset.len();
//     // let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
//     // type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
//     // type MyAutodiffBackend = ADBackendDecorator<MyBackend>;
//     // let device = burn::backend::wgpu::WgpuDevice::default();

//     // Create the configuration.
//     let config_model = ModelConfig::new(num_classes);
//     let config_optimizer = AdamConfig::new();
//     let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

//     // MyAutodiffBackend::seed(config.seed);


//     // Create the model and optimizer.
//     let mut model: Model<MyAutodiffBackend> = config.model.init(no_features);
//     let mut optim = config.optimizer.init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();

//     // Create the batcher.
//     let batcher_train = SCBatcher::<MyAutodiffBackend>::new(device.clone());
//     let batcher_valid = SCBatcher::<MyBackend>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(train_dataset);

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         // .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(test_dataset);

//     let mut train_accuracy = ModelAccuracy::new();
//     let mut test_accuracy = ModelAccuracy::new();

//     // progress bar items
//     let num_iterations = (num_batches_train / config.batch_size) as u32; 
//     let length = 40;
//     let eta = false;

//     //history stuff
//     let mut train_history: History = History::new();
//     let mut test_history: History = History::new();

//     let start = Instant::now();
//     // Iterate over our training and validation loop for X epochs.
//     for epoch in 1..config.num_epochs + 1 {
//         // Implement our training loop.
//         train_accuracy.epoch = epoch;
//         test_accuracy.epoch = epoch;
//         test_accuracy.epoch_reset(epoch);
//         train_accuracy.epoch_reset(epoch);
//         let mut bar = ProgressBar::default(num_iterations, length, eta);

//         if verbose {eprintln!("[Epoch {} progress...]", epoch)}
//         for (_iteration, batch) in dataloader_train.iter().enumerate() {
//             if verbose {bar.update()}
//             // bar.inc(1);
//             let output = model.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &device).forward(output.clone(), batch.targets.clone());
//             // let accuracy = accuracy(output, batch.targets);
//             let loss_scalar = loss.clone().into_scalar();
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
//             train_accuracy.batch_update(num_corrects, num_predictions, loss_scalar);

//             // Gradients for the current backward pass
//             let grads = loss.backward();
//             // Gradients linked to each parameter of the model.
//             let grads = GradientsParams::from_grads(grads, &model);
//             // Update the model using the optimizer.
//             model = optim.step(config.lr, model, grads);
//         }
//         train_accuracy.epoch_update(& mut train_history);
//         // bar.finish();
        
//         // Get the model without autodiff.
//         let model_valid = model.valid();

//         // Implement our validation loop.
//         for (_iteration, batch) in dataloader_test.iter().enumerate() {
//             let output = model_valid.forward(batch.counts);
//             let loss = CrossEntropyLoss::new(None, &device).forward(output.clone(), batch.targets.clone());
//             let loss_scalar = &loss.into_scalar();
//             // if iteration == test_len {
//             //     last_epoch_loss = loss.into_scalar();
//             // }
//             let predictions = output.argmax(1).squeeze(1);
//             let num_predictions: usize = batch.targets.dims().iter().product();
//             let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
//             test_accuracy.batch_update(num_corrects, num_predictions, *loss_scalar);
//             // let accuracy = accuracy(output, batch.targets);
//         }
//         test_accuracy.epoch_update(& mut test_history);
//         if verbose {emit_metrics(&train_accuracy, &test_accuracy)}
//     }
//     let tduration = start.elapsed();

//     let mut prediction: Vec<i32> = vec![];
//     for item in query {
//         let batcher = SCBatcher::new(device.clone());
//         let batch = batcher.batch(vec![map_raw(&item)]);
//         let output = &model.forward(batch.counts);
//         prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
//     }
    
//     // save model
//     let _ = &model.clone()
//         .save_file(format!("{artifact_dir}/model"), &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
//         .expect("Failed to save trained model");

//     //collect and return the predictions
//     return ModelRExport {
//         lr: config.lr,
//         hidden_size: vec![0],
//         batch_size: config.batch_size,
//         num_epochs: config.num_epochs,
//         num_workers: config.num_workers,
//         seed: config.seed,
//         predictions: prediction,
//         train_history: train_history,
//         test_history: test_history,
//         training_duration: tduration.as_secs_f64(),
//     }
// }

