use std::convert::TryInto;
use std::time::Instant;

use num_traits::ToPrimitive;

use burn::{
    backend::Autodiff,
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, Dataset},
        dataset::{transform::MapperDataset, InMemDataset},
    },
    module::{AutodiffModule, Module},
    nn::{
        loss::CrossEntropyLoss,
        Linear,
        LinearConfig,
        ReLU,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, backend::Backend},
    // train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::common::*;
use crate::pb::ProgressBar;


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


pub fn run_custom<B>(
    train: Vec<SCItemRaw>,
    test: Vec<SCItemRaw>,
    query: Vec<SCItemRaw>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
    device: B::Device,
) -> ModelRExport
    where
     B: Backend,
     B::Device: Clone, 
     B::IntElem: ToPrimitive,
     B::FloatElem: ToPrimitive,
{

    let no_features = train.first().unwrap().data.len();
    let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
        MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
    let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
        MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
    let num_batches_train = train_dataset.len();
    let artifact_dir = directory.clone().unwrap_or_else(|| panic!("Folder not found: {:?}", directory));
    
    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config =
        SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

    // Create the model and optimizer.
    let mut model: Model<Autodiff<B>> = config.model.init(no_features);
    let mut optim = config.optimizer.init::<Autodiff<B>, Model<Autodiff<B>>>();

    // Create the batchers.
    let batcher_train = SCBatcher::<Autodiff<B>>::new(device.clone());
    let batcher_valid = SCBatcher::<B>::new(device.clone());

    // Create the dataloaders.
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

    // Progress bar items
    let num_iterations = (num_batches_train as f64 / config.batch_size as f64).ceil() as u32; // adjust to estimate ceiling
    let length = 40;
    let eta = false;

    // History tracking
    let mut train_history: History = History::new();
    let mut test_history: History = History::new();
    
    let start = Instant::now();

    // Training and validation loop
    for epoch in 1..=config.num_epochs {
        train_accuracy.epoch_reset(epoch);
        test_accuracy.epoch_reset(epoch);
        let mut bar = ProgressBar::default(num_iterations, length, eta);
        let loss_fn = CrossEntropyLoss::new(None);
        if verbose {
            eprintln!("[Epoch {} progress...]", epoch);
        }

        // Training loop
        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            if verbose {
                bar.update();
            }
            let output = model.forward(batch.counts);
            let loss = loss_fn.forward(output.clone(), batch.targets.clone());
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions = batch.targets.dims()[0];
            let num_corrects = predictions
                .equal(batch.targets)
                .int()
                .sum()
                .into_scalar()
                .to_usize()
                .expect("Conversion to usize failed");

            let loss_scalar = loss.clone()
                .into_scalar()
                .to_f64()
                .expect("Conversion to f64 failed");
            train_accuracy.batch_update(num_corrects.try_into().unwrap(), num_predictions, loss_scalar);

            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }
        train_accuracy.epoch_update(&mut train_history);

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Validation loop
        for (_iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.counts);
            let loss = CrossEntropyLoss::new(None)
                .forward(output.clone(), batch.targets.clone());

            let loss_scalar = loss
                .into_scalar()
                .to_f64()
                .expect("Conversion to f64 failed");
            let predictions = output.argmax(1).squeeze(1);
            let num_predictions = batch.targets.dims().iter().product();
            let num_corrects = predictions
                .equal(batch.targets)
                .int()
                .sum()
                .into_scalar()
                .to_usize()
                .expect("Conversion to usize failed");

            test_accuracy.batch_update(num_corrects.to_i64().expect("Failed to convert usize to i64"), num_predictions, loss_scalar);
        }
        test_accuracy.epoch_update(&mut test_history);

        if verbose {
            emit_metrics(&train_accuracy, &test_accuracy);
        }
    }

    let tduration = start.elapsed();

    let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
    MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    let query_len = query_dataset.len();
    let batcher_query = SCBatcher::<B>::new(device.clone());

    let dataloader_query = DataLoaderBuilder::new(batcher_query)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(query_dataset);

    let model_valid = model.valid();
    let mut predictions = Vec::with_capacity(query_len);

    for batch in dataloader_query.iter() {
        let output = model_valid.forward(batch.counts);
        let batch_predictions = output.argmax(1).squeeze::<1>(1);
        predictions.extend(
            batch_predictions
                .to_data()
                .value.iter()
                .map(|&pred| pred.to_i32().expect("Failed to convert prediction to i32")),
        );
    }

    // Save the model
    model
        .save_file(
            format!("{}/model", artifact_dir),
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
        predictions: predictions,
        train_history,
        test_history,
        training_duration: tduration.as_secs_f64(),
    }
}

pub fn run_custom_nd(
    train: Vec<SCItemRaw>,
    test: Vec<SCItemRaw>,
    query: Vec<SCItemRaw>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> ModelRExport {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    let device = NdArrayDevice::default();
    run_custom::<NdArray<f64>>(
        train,
        test,
        query,
        num_classes,
        learning_rate,
        num_epochs,
        directory,
        verbose,
        device,
    )
}


pub fn run_custom_wgpu(
    train: Vec<SCItemRaw>,
    test: Vec<SCItemRaw>,
    query: Vec<SCItemRaw>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> ModelRExport {
    use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};

    let device = WgpuDevice::default();
    run_custom::<Wgpu<AutoGraphicsApi, f32, i32>>(
        train,
        test,
        query,
        num_classes,
        learning_rate,
        num_epochs,
        directory,
        verbose,
        device,
    )
}


pub fn run_custom_candle(
    train: Vec<SCItemRaw>,
    test: Vec<SCItemRaw>,
    query: Vec<SCItemRaw>,
    num_classes: usize,
    learning_rate: f64,
    num_epochs: usize,
    directory: Option<String>,
    verbose: bool,
) -> ModelRExport {
    use burn::backend::candle::{Candle, CandleDevice};

    let device = CandleDevice::default();
    run_custom::<Candle<f64, i64>>(
        train,
        test,
        query,
        num_classes,
        learning_rate,
        num_epochs,
        directory,
        verbose,
        device,
    )
}
