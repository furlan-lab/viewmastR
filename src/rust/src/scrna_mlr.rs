
use burn::{
    backend::Autodiff,
    backend::ndarray::{NdArray, NdArrayDevice},
    backend::wgpu::{WgpuDevice, Wgpu, AutoGraphicsApi},
    backend::candle::Candle,
    config::Config,
    data::{dataloader::{DataLoaderBuilder, Dataset, batcher::Batcher}, dataset::{InMemDataset, transform::MapperDataset}},
    module::{Module, AutodiffModule},
    nn::{
        loss::CrossEntropyLoss,
        Linear, LinearConfig, ReLU,
    },
    optim::{Optimizer, AdamConfig, GradientsParams},
    record::{NamedMpkFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Int, Tensor},
    train::{ClassificationOutput, TrainStep, ValidStep, TrainOutput},
};


use std::result::Result;
use std::vec::Vec;
use std::convert::TryInto;
use crate::pb::ProgressBar;
use std::time::Instant;
use crate::common::*;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    activation: ReLU,
}


#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
}


impl ModelConfig {
    pub fn init_with<B: Backend>(&self, no_features: usize, record: ModelRecord<B>) -> Model<B> {
        Model {
            linear1: LinearConfig::new(no_features, self.num_classes).init_with(record.linear1),
            activation: ReLU::new(),
        }
    }
    pub fn init<B: Backend>(&self, no_features: usize) -> Model<B> {
        Model {
            activation: ReLU::new(),
            linear1: LinearConfig::new(no_features, self.num_classes).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    pub fn forward(&self, images: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, dim] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, dim]);
        self.linear1.forward(x)
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

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


pub fn run_custom_candle(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool)->ModelRExport {
    type MyBackend = Candle<f64, i64>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::candle::CandleDevice::default();
    
    let no_features = train.first().unwrap().data.len();
    let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
    let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
    let num_batches_train = train_dataset.len();
    let artifact_dir = match directory {
        Some(directory) => directory,
        _ => panic!("Folder not found: {:?}", directory)
    };

    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

    // Create the model and optimizer.
    let mut model: Model<MyAutodiffBackend> = config.model.init(no_features);
    let mut optim = config.optimizer.init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();

    // Create the batcher.
    let batcher_train = SCBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = SCBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        // .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        // .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let mut train_accuracy = ModelAccuracy::new();
    let mut test_accuracy = ModelAccuracy::new();

    // progress bar items
    let num_iterations = (num_batches_train / config.batch_size) as u32; 
    let length = 40;
    let eta = false;

    //history stuff
    let mut train_history: History = History::new();
    let mut test_history: History = History::new();

    let start = Instant::now();
    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        train_accuracy.epoch = epoch;
        test_accuracy.epoch = epoch;
        test_accuracy.epoch_reset(epoch);
        train_accuracy.epoch_reset(epoch);
        let mut bar = ProgressBar::default(num_iterations, length, eta);

        if verbose {eprintln!("[Epoch {} progress...]", epoch)}
        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            if verbose {bar.update()}
            // bar.inc(1);
            let output = model.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            // let accuracy = accuracy(output, batch.targets);
            let loss_scalar = loss.clone().into_scalar();
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            train_accuracy.batch_update(num_corrects, num_predictions, loss_scalar);

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }
        train_accuracy.epoch_update(& mut train_history);
        // bar.finish();
        
        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (_iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            let loss_scalar = &loss.into_scalar();
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            test_accuracy.batch_update(num_corrects, num_predictions, *loss_scalar);
        }
        test_accuracy.epoch_update(& mut test_history);
        if verbose {emit_metrics(&train_accuracy, &test_accuracy)}
    }
    let tduration = start.elapsed();

    let mut prediction: Vec<i32> = vec![];
    for item in query {
        let batcher = SCBatcher::new(device.clone());
        let batch = batcher.batch(vec![map_raw(&item)]);
        let output = &model.forward(batch.counts);
        prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
    }
    
    // save model
    let _ = &model.clone()
        .save_file(format!("{artifact_dir}/model"), &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save trained model");

    //collect and return the predictions
    return ModelRExport {
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


pub fn run_custom_wgpu(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool)->ModelRExport {

    type MyBackend = Wgpu<AutoGraphicsApi, f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = WgpuDevice::default();

    let no_features = train.first().unwrap().data.len();
    let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
    let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
    let num_batches_train = train_dataset.len();
    let artifact_dir = match directory {
        Some(directory) => directory,
        _ => panic!("Folder not found: {:?}", directory)
    };

    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

    // Create the model and optimizer.
    let mut model: Model<MyAutodiffBackend> = config.model.init(no_features);
    let mut optim = config.optimizer.init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();

    // Create the batcher.
    let batcher_train = SCBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = SCBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        // .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        // .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let mut train_accuracy = ModelAccuracy::new();
    let mut test_accuracy = ModelAccuracy::new();

    // progress bar items
    let num_iterations = (num_batches_train / config.batch_size) as u32; 
    let length = 40;
    let eta = false;

    //history stuff
    let mut train_history: History = History::new();
    let mut test_history: History = History::new();

    let start = Instant::now();
    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        train_accuracy.epoch = epoch;
        test_accuracy.epoch = epoch;
        test_accuracy.epoch_reset(epoch);
        train_accuracy.epoch_reset(epoch);
        let mut bar = ProgressBar::default(num_iterations, length, eta);

        if verbose {eprintln!("[Epoch {} progress...]", epoch)}
        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            if verbose {bar.update()}
            // bar.inc(1);
            let output = model.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            // let accuracy = accuracy(output, batch.targets);
            let loss_scalar = loss.clone().into_scalar();
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            train_accuracy.batch_update(num_corrects.into(), num_predictions, loss_scalar.into());

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }
        train_accuracy.epoch_update(& mut train_history);
        // bar.finish();
        
        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (_iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            let loss_scalar = &loss.into_scalar();
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            test_accuracy.batch_update(num_corrects.into(), num_predictions, (*loss_scalar).into());
        }
        test_accuracy.epoch_update(& mut test_history);
        if verbose {emit_metrics(&train_accuracy, &test_accuracy)}
    }
    let tduration = start.elapsed();

    let mut prediction: Vec<i32> = vec![];
    for item in query {
        let batcher = SCBatcher::new(device.clone());
        let batch = batcher.batch(vec![map_raw(&item)]);
        let output = &model.forward(batch.counts);
        prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
    }
    
    // save model
    let _ = &model.clone()
        .save_file(format!("{artifact_dir}/model"), &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save trained model");

    //collect and return the predictions
    return ModelRExport {
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



pub fn run_custom_nd(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool)->ModelRExport {
    type MyBackend = NdArray<f64>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = NdArrayDevice::default();
    let no_features = train.first().unwrap().data.len();
    let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
    let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
    let num_batches_train = train_dataset.len();
    let artifact_dir = match directory {
        Some(directory) => directory,
        _ => panic!("Folder not found: {:?}", directory)
    };
    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

    // Create the model and optimizer.
    let mut model: Model<MyAutodiffBackend> = config.model.init(no_features);
    let mut optim = config.optimizer.init::<MyAutodiffBackend, Model<MyAutodiffBackend>>();

    // Create the batcher.
    let batcher_train = SCBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = SCBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        // .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        // .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let mut train_accuracy = ModelAccuracy::new();
    let mut test_accuracy = ModelAccuracy::new();

    // progress bar items
    let num_iterations = (num_batches_train / config.batch_size) as u32; 
    let length = 40;
    let eta = false;

    //history stuff
    let mut train_history: History = History::new();
    let mut test_history: History = History::new();

    let start = Instant::now();
    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        train_accuracy.epoch = epoch;
        test_accuracy.epoch = epoch;
        test_accuracy.epoch_reset(epoch);
        train_accuracy.epoch_reset(epoch);
        let mut bar = ProgressBar::default(num_iterations, length, eta);

        if verbose {eprintln!("[Epoch {} progress...]", epoch)}
        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            if verbose {bar.update()}
            // bar.inc(1);
            let output = model.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            // let accuracy = accuracy(output, batch.targets);
            let loss_scalar = loss.clone().into_scalar();
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            train_accuracy.batch_update(num_corrects, num_predictions, loss_scalar);

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }
        train_accuracy.epoch_update(& mut train_history);
        // bar.finish();
        
        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (_iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            let loss_scalar = &loss.into_scalar();
            // if iteration == test_len {
            //     last_epoch_loss = loss.into_scalar();
            // }
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            test_accuracy.batch_update(num_corrects, num_predictions, *loss_scalar);
            // let accuracy = accuracy(output, batch.targets);
        }
        test_accuracy.epoch_update(& mut test_history);
        if verbose {emit_metrics(&train_accuracy, &test_accuracy)}
    }
    let tduration = start.elapsed();

    let mut prediction: Vec<i32> = vec![];
    for item in query {
        let batcher = SCBatcher::new(device.clone());
        let batch = batcher.batch(vec![map_raw(&item)]);
        let output = &model.forward(batch.counts);
        prediction.push(output.clone().argmax(1).flatten::<1>(0, 1).into_scalar().try_into().unwrap());
    }
    
    // save model
    let _ = &model.clone()
        .save_file(format!("{artifact_dir}/model"), &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save trained model");

    //collect and return the predictions
    return ModelRExport {
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
