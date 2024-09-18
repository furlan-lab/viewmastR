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
    optim::{AdamConfig, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, backend::Backend, Int},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
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
    pub fn forward(&self, data: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear1.forward(data)
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        data: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(data);
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
    #[config(default = 16)]
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
    let no_features = train.first().expect("Features not found").data.len();
    let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
        MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
    let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
        MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
    let num_batches_train = train_dataset.len();
    let artifact_dir = directory.clone().unwrap_or_else(|| panic!("Folder not found: {:?}", directory));
    
    // Create the configuration.
    let config_model = ModelConfig::new(num_classes);
    let config_optimizer = AdamConfig::new();
    let config = SCTrainingConfig::new(num_epochs, learning_rate, config_model, config_optimizer);

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
    let num_iterations = (num_batches_train as f64 / config.batch_size as f64).ceil() as u32;
    let batch_report_interval = num_iterations.to_usize().unwrap() - 1;
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
        if verbose {
            eprintln!("[Epoch {} progress...]", epoch);
        }

        // Training loop using `TrainStep`
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            if verbose {
                bar.update();
            }
            

            let output = TrainStep::step(&model, batch); // using the `step` method
            model = optim.step(config.lr, model, output.grads);
            // // Calculate number of correct predictions on the last batch
            if iteration == batch_report_interval {
                let predictions = output.item.output.argmax(1).squeeze(1);
                let num_predictions = output.item.targets.dims()[0];
                let num_corrects = predictions
                    .equal(output.item.targets)
                    .int()
                    .sum()
                    .into_scalar()
                    .to_usize()
                    .expect("Conversion to usize failed");

                // Update accuracy and loss tracking
                train_accuracy.batch_update(num_corrects, num_predictions, output.item.loss.into_scalar().to_f64().expect("Conversion to f64 failed"));
            }
        }

        train_accuracy.epoch_update(&mut train_history);

        // Validation loop using `ValidStep`
        for (_iteration, batch) in dataloader_test.iter().enumerate() {
            let output = ValidStep::step(&model.valid(), batch.clone()); // using the `step` method

            // Calculate number of correct predictions
            let predictions = output.output.argmax(1).squeeze(1);
            let num_predictions = batch.targets.dims()[0];
            let num_corrects = predictions
                .equal(batch.targets)
                .int()
                .sum()
                .into_scalar()
                .to_usize()
                .expect("Conversion to usize failed");

            // Update accuracy and loss tracking
            test_accuracy.batch_update(num_corrects, num_predictions, output.loss.into_scalar().to_f64().expect("Conversion to f64 failed"));
        }
        test_accuracy.epoch_update(&mut test_history);

        if verbose {
            emit_metrics(&train_accuracy, &test_accuracy);
        }
    }

    let tduration = start.elapsed();


    // Query handling and predictions with proper indexing
    let query_with_index: Vec<(usize, SCItemRaw)> = query.into_iter().enumerate().collect();
    let query_len = query_with_index.len();
    let query_dataset: MapperDataset<InMemDataset<(usize, SCItemRaw)>, LocalCountstoMatrixWithIndex, SCItem> =
        MapperDataset::new(InMemDataset::new(query_with_index), LocalCountstoMatrixWithIndex);

    let batcher_query = SCBatcher::<B>::new(device.clone());
    let dataloader_query = DataLoaderBuilder::new(batcher_query)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(query_dataset);  // query_dataset now returns (usize, SCItem)

    let model_valid = model.valid();
    let mut indexed_predictions: Vec<(usize, i32)> = Vec::with_capacity(query_len);

    // Create a Vec to store the original indices
    let original_indices: Vec<usize> = (0..query_len).collect();
    
    // Assuming dataloader_query is built
    for (batch_index, batch) in dataloader_query.iter().enumerate() {
        let output = model_valid.forward(batch.counts);
        let batch_predictions = output.argmax(1).squeeze::<1>(1);
    
        // Collect predictions along with their original indices
        for (i, pred) in original_indices.iter().zip(batch_predictions.to_data().value.iter()) {
            indexed_predictions.push((*i, pred.to_i32().expect("Failed to convert prediction to i32")));
        }
    }
    
    // Sort predictions by the original index to restore the correct order
    indexed_predictions.sort_by_key(|&(index, _)| index);
    let predictions: Vec<i32> = indexed_predictions.into_iter().map(|(_, pred)| pred).collect();
    
    // Sort predictions by the original index to restore the correct order
    indexed_predictions.sort_by_key(|&(index, _)| index);
    let predictions: Vec<i32> = indexed_predictions.into_iter().map(|(_, pred)| pred).collect();

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
