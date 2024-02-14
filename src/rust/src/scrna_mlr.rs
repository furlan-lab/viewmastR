
use burn::{
    backend::Autodiff,
    backend::ndarray::{NdArray, NdArrayDevice},
    backend::wgpu::{WgpuDevice, Wgpu, AutoGraphicsApi},
    backend::candle::Candle,
    // backend::libtorch::{LibTorch, LibTorchDevice},
    config::Config,
    data::{dataloader::{DataLoaderBuilder, Dataset, batcher::Batcher}, dataset::{InMemDataset, transform::{Mapper,MapperDataset}}},
    module::{Module, AutodiffModule},
    nn::{
        loss::CrossEntropyLoss,
        Linear, LinearConfig, ReLU,
    },
    optim::{Optimizer, AdamConfig, GradientsParams},
    record::{NamedMpkFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
    train::{ClassificationOutput, TrainStep, ValidStep, TrainOutput},
};


use std::result::Result;
use std::vec::Vec;
use std::convert::TryInto;
// use crate::scrna_ann::SCItemRaw;
// use indicatif::{ProgressStyle, ProgressBar, ProgressDrawTarget};
use crate::pb::ProgressBar;
// use std::collections::HashMap;
// use test::stats::Stats;
use std::time::Instant;
use crate::common::{SCItemRaw, History, ModelRExport, ModelAccuracy, emit_metrics, SCItem};



pub struct SCBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SCBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}



#[derive(Clone, Debug)]
pub struct SCBatch<B: Backend> {
    pub counts: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<SCItem, SCBatch<B>> for SCBatcher<B>  {

    fn batch(&self, items: Vec<SCItem>) -> SCBatch<B> {
        let n: usize = items.first().unwrap().counts.len();
        let counts = items
            .iter()
            .map(|item| Data::<f64, 1>::from(&item.counts[0..n]))
            // .map(|item| Data::item))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, n]))
            // Normalize: make between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            // .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i32).elem()])))
            .collect();

        let counts = Tensor::cat(counts, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        SCBatch { counts, targets }
    }
}


#[derive(Clone, Debug)]
pub struct MyBatch<B: Backend> {
    pub images: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}


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
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
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


pub struct LocalCountstoMatrix;

impl Mapper<SCItemRaw, SCItem> for LocalCountstoMatrix {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &SCItemRaw) -> SCItem {
        let counts = &item.data;

        // // Convert the image to a 2D array of floats.
        // let mut counts_array = [0f32; 3600];
        // for (i, pixel) in counts.iter().enumerate() {
        //     counts_array[i] = *pixel as f32;
        // }

        SCItem {
            counts: counts.to_vec(),
            label: item.target,
        }
    }
}


pub fn map_raw(item: &SCItemRaw) -> SCItem {
    let counts = &item.data;

    SCItem {
        counts: counts.to_vec(),
        label: item.target,
    }
}

pub struct SCLocalDataset {
    pub dataset: dyn Dataset<SCItem>,
}


impl Dataset<SCItem> for SCLocalDataset {
    fn get(&self, index: usize) -> Option<SCItem> {
        self.dataset.get(index)
        // None
    }

    fn len(&self) -> usize {
        self.dataset.len()
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
    
    
    // let num_batches_test = train_dataset.len();
    // let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    // type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    // type MyAutodiffBackend = ADBackendDecorator<MyBackend>;
    // let device = burn::backend::wgpu::WgpuDevice::default();

    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

    // MyAutodiffBackend::seed(config.seed);


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
    
    
    // let num_batches_test = train_dataset.len();
    // let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    // type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    // type MyAutodiffBackend = ADBackendDecorator<MyBackend>;
    // let device = burn::backend::wgpu::WgpuDevice::default();

    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

    // MyAutodiffBackend::seed(config.seed);


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
            // if iteration == test_len {
            //     last_epoch_loss = loss.into_scalar();
            // }
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions: usize = batch.targets.dims().iter().product();
            let num_corrects = predictions.equal(batch.targets).int().sum().into_scalar();
            test_accuracy.batch_update(num_corrects.into(), num_predictions, (*loss_scalar).into());
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



pub fn run_custom_nd(train: Vec<SCItemRaw>, test: Vec<SCItemRaw>, query: Vec<SCItemRaw>, num_classes: usize, learning_rate: f64, num_epochs: usize, directory: Option<String>, verbose: bool)->ModelRExport {
    type MyBackend = NdArray<f64>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = NdArrayDevice::default();
    // if cfg!(target_os = "macos") {
    //     device = LibTorchDevice::Mps;
    // } else {
    //     device = LibTorchDevice::Cuda(0);
    // }
    let no_features = train.first().unwrap().data.len();
    let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
    let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
    let num_batches_train = train_dataset.len();
    let artifact_dir = match directory {
        Some(directory) => directory,
        _ => panic!("Folder not found: {:?}", directory)
    };
    
    
    // let num_batches_test = train_dataset.len();
    // let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    // type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    // type MyAutodiffBackend = ADBackendDecorator<MyBackend>;
    // let device = burn::backend::wgpu::WgpuDevice::default();

    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

    // MyAutodiffBackend::seed(config.seed);


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

// pub fn run() {
//     #[cfg(not(target_os = "macos"))]
//     let device = LibTorchDevice::Cuda(0);
//     #[cfg(target_os = "macos")]
//     let device = LibTorchDevice::Mps;

//     training::run::<Autodiff<LibTorch<f32>>>(device);
// }