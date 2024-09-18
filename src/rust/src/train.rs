// #[derive(Debug)]
// pub struct RExport {
//     pub config: ModelConfig,
//     pub predictions: Vec<i32>,
//     pub train_history: History,
//     pub test_history: History,
//     pub training_duration: f64,
// }

// use std::time::Instant;
// use std::path::Path;
// use serde::Deserialize;
// use extendr_api::Robj;

// pub struct LearningProcessor {
//     train: Robj,
//     test: Robj,
//     query: Robj,
//     labels: Robj,
//     model: Robj,
//     num_epochs: Robj,
//     directory: Robj,
//     verbose: Robj,
//     backend: String,
// }

// impl LearningProcessor {
//     pub fn new(
//         train: Robj, 
//         test: Robj, 
//         query: Robj, 
//         labels: Robj, 
//         model: Robj, 
//         num_epochs: Robj, 
//         directory: Robj, 
//         verbose: Robj, 
//         backend: Robj,
//     ) -> Self {
//         let backend = match backend.as_str_vector() {
//             Some(string_vec) => string_vec.first().unwrap().to_string(),
//             _ => panic!("Could not find backend: '{:?}'", backend),
//         };
//         if !["wgpu", "candle", "nd"].contains(&backend.as_str()) {
//             panic!("Could not find backend: '{:?}'", backend);
//         }

//         Self {
//             train,
//             test,
//             query,
//             labels,
//             model,
//             num_epochs,
//             directory,
//             verbose,
//             backend,
//         }
//     }

//     pub fn process(&self) -> List {
//         let start = Instant::now();
//         let verbose = self.extract_verbose();
//         let learning_rate = self.extract_learning_rate();
//         let num_epochs = self.extract_num_epochs();
//         let artifact_dir = self.extract_directory();
//         let labelvec = self.labels.as_str_vector().unwrap();

//         let test_raw = self.extract_scitemraw(&self.test, None);
//         let train_raw = self.extract_scitemraw(&self.train, None);
//         let query_raw = self.extract_scitemraw(&self.query, Some(0));

//         let model_export = self.run_backend(train_raw, test_raw, query_raw, labelvec.len(), learning_rate, num_epochs, artifact_dir, verbose);

//         self.build_output(model_export, start)
//     }

//     fn extract_verbose(&self) -> bool {
//         self.verbose
//             .as_logical_vector()
//             .unwrap()
//             .first()
//             .unwrap()
//             .to_bool()
//     }

//     fn extract_learning_rate(&self) -> f64 {
//         *self.model.as_real_vector().unwrap().first().unwrap_or(&0.2) as f64
//     }

//     fn extract_num_epochs(&self) -> usize {
//         *self.num_epochs.as_real_vector().unwrap().first().unwrap_or(&10.0) as usize
//     }

//     fn extract_directory(&self) -> String {
//         let artifact_dir = match self.directory.as_str_vector() {
//             Some(string_vec) => string_vec.first().unwrap().to_string(),
//             _ => panic!("Could not find folder: '{:?}'", self.directory),
//         };

//         if !Path::new(&artifact_dir).exists() {
//             panic!("Could not find folder: '{:?}'", artifact_dir);
//         }

//         artifact_dir
//     }

//     fn extract_scitemraw(&self, data: &Robj, target_value: Option<i32>) -> Vec<SCItemRaw> {
//         extract_scitemraw(data, target_value)
//     }

//     fn run_backend(
//         &self, 
//         train_raw: Vec<SCItemRaw>, 
//         test_raw: Vec<SCItemRaw>, 
//         query_raw: Vec<SCItemRaw>, 
//         label_count: usize, 
//         learning_rate: f64, 
//         num_epochs: usize, 
//         artifact_dir: String, 
//         verbose: bool
//     ) -> ModelRExport {
//         match self.backend.as_str() {
//             "candle" => scrna_mlr::run_custom_candle(train_raw, test_raw, query_raw, label_count, learning_rate, num_epochs, Some(artifact_dir), verbose),
//             "wgpu" => scrna_mlr::run_custom_wgpu(train_raw, test_raw, query_raw, label_count, learning_rate, num_epochs, Some(artifact_dir), verbose),
//             "nd" => scrna_mlr::run_custom_nd(train_raw, test_raw, query_raw, label_count, learning_rate, num_epochs, Some(artifact_dir), verbose),
//             _ => panic!("Unknown backend: {}", self.backend),
//         }
//     }

//     fn build_output(&self, model_export: ModelRExport, start: Instant) -> List {
//         let params = list!(
//             lr = model_export.lr,
//             epochs = model_export.num_epochs,
//             batch_size = model_export.batch_size,
//             workers = model_export.num_workers,
//             seed = model_export.seed
//         );

//         let predictions = list!(model_export.predictions);
//         let history: List = list!(
//             train_acc = model_export.train_history.acc,
//             test_acc = model_export.test_history.acc,
//             train_loss = model_export.train_history.loss,
//             test_loss = model_export.test_history.loss
//         );

//         let duration = start.elapsed();
//         let duration: List = list!(
//             total_duration = duration.as_secs_f64(),
//             training_duration = model_export.training_duration
//         );

//         list!(
//             params = params,
//             predictions = predictions,
//             history = history,
//             duration = duration
//         )
//     }
// }

// pub fn run_custom<B>(
//     train: Vec<SCItemRaw>,
//     test: Vec<SCItemRaw>,
//     query: Vec<SCItemRaw>,
//     config_model: ModelConfig,
//     directory: Option<String>,
//     verbose: bool,
//     device: B::Device,
// ) -> ModelRExport
// where
//     B: Backend,
//     B::Device: Clone,
//     B::IntElem: ToPrimitive,
//     B::FloatElem: ToPrimitive,
// {
//     let no_features = train.first().expect("Features not found").data.len();
//     let train_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
//         MapperDataset::new(InMemDataset::new(train), LocalCountstoMatrix);
//     let test_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
//         MapperDataset::new(InMemDataset::new(test), LocalCountstoMatrix);
//     let num_batches_train = train_dataset.len();
//     let artifact_dir = directory.clone().unwrap_or_else(|| panic!("Folder not found: {:?}", directory));
    
//     // Create the model and optimizer.
//     let mut model: Model<Autodiff<B>> = config.model.init(no_features);
//     let mut optim = config.optimizer.init::<Autodiff<B>, Model<Autodiff<B>>>();

//     // Create the batchers.
//     let batcher_train = SCBatcher::<Autodiff<B>>::new(device.clone());
//     let batcher_valid = SCBatcher::<B>::new(device.clone());

//     // Create the dataloaders.
//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         .num_workers(config.num_workers)
//         .build(train_dataset);

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         .num_workers(config.num_workers)
//         .build(test_dataset);

//     let mut train_accuracy = ModelAccuracy::new();
//     let mut test_accuracy = ModelAccuracy::new();

//     // Progress bar items
//     let num_iterations = (num_batches_train as f64 / config.batch_size as f64).ceil() as u32;
//     let batch_report_interval = num_iterations.to_usize().unwrap() - 1;
//     let length = 40;
//     let eta = false;

//     // History tracking
//     let mut train_history: History = History::new();
//     let mut test_history: History = History::new();

//     let start = Instant::now();

//     // Training and validation loop
//     for epoch in 1..=config.num_epochs {
//         train_accuracy.epoch_reset(epoch);
//         test_accuracy.epoch_reset(epoch);
//         let mut bar = ProgressBar::default(num_iterations, length, eta);
//         if verbose {
//             eprintln!("[Epoch {} progress...]", epoch);
//         }

//         // Training loop using `TrainStep`
//         for (iteration, batch) in dataloader_train.iter().enumerate() {
//             if verbose {
//                 bar.update();
//             }
            

//             let output = TrainStep::step(&model, batch); // using the `step` method
//             model = optim.step(config.lr, model, output.grads);
//             // // Calculate number of correct predictions on the last batch
//             if iteration == batch_report_interval {
//                 let predictions = output.item.output.argmax(1).squeeze(1);
//                 let num_predictions = output.item.targets.dims()[0];
//                 let num_corrects = predictions
//                     .equal(output.item.targets)
//                     .int()
//                     .sum()
//                     .into_scalar()
//                     .to_usize()
//                     .expect("Conversion to usize failed");

//                 // Update accuracy and loss tracking
//                 train_accuracy.batch_update(num_corrects, num_predictions, output.item.loss.into_scalar().to_f64().expect("Conversion to f64 failed"));
//             }
//         }

//         train_accuracy.epoch_update(&mut train_history);

//         // Validation loop using `ValidStep`
//         for (_iteration, batch) in dataloader_test.iter().enumerate() {
//             let output = ValidStep::step(&model.valid(), batch.clone()); // using the `step` method

//             // Calculate number of correct predictions
//             let predictions = output.output.argmax(1).squeeze(1);
//             let num_predictions = batch.targets.dims()[0];
//             let num_corrects = predictions
//                 .equal(batch.targets)
//                 .int()
//                 .sum()
//                 .into_scalar()
//                 .to_usize()
//                 .expect("Conversion to usize failed");

//             // Update accuracy and loss tracking
//             test_accuracy.batch_update(num_corrects, num_predictions, output.loss.into_scalar().to_f64().expect("Conversion to f64 failed"));
//         }
//         test_accuracy.epoch_update(&mut test_history);

//         if verbose {
//             emit_metrics(&train_accuracy, &test_accuracy);
//         }
//     }

//     let tduration = start.elapsed();

//     // Query handling and predictions
//     let query_dataset: MapperDataset<InMemDataset<SCItemRaw>, LocalCountstoMatrix, SCItemRaw> =
//         MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
//     let query_len = query_dataset.len();
//     let batcher_query = SCBatcher::<B>::new(device.clone());

//     let dataloader_query = DataLoaderBuilder::new(batcher_query)
//         .batch_size(config.batch_size)
//         .num_workers(config.num_workers)
//         .build(query_dataset);

//     let model_valid = model.valid();
//     let mut predictions = Vec::with_capacity(query_len);

//     for batch in dataloader_query.iter() {
//         let output = model_valid.forward(batch.counts);
//         let batch_predictions = output.argmax(1).squeeze::<1>(1);
//         predictions.extend(
//             batch_predictions
//                 .to_data()
//                 .value.iter()
//                 .map(|&pred| pred.to_i32().expect("Failed to convert prediction to i32")),
//         );
//     }

//     // Save the model
//     model
//         .save_file(
//             format!("{}/model", artifact_dir),
//             &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
//         )
//         .expect("Failed to save trained model");

//     // Collect and return the predictions
//     RExport {
//         config: config_model,
//         predictions: predictions,
//         train_history,
//         test_history,
//         training_duration: tduration.as_secs_f64(),
//     }
// }


