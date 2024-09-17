// candle does not support AdaptiveAvgPool2d... 
use burn::{
    backend::Autodiff,
    backend::wgpu::Wgpu,
    backend::wgpu::AutoGraphicsApi,
    backend::ndarray::NdArray,
    config::Config,
    data::{dataloader::{DataLoaderBuilder, Dataset, batcher::Batcher}, dataset::{InMemDataset, source::huggingface::{MNISTDataset, MNISTItem}, transform::{Mapper,MapperDataset}}},
    module::{Module, AutodiffModule},
    nn::{
        loss::CrossEntropyLoss,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Linear, LinearConfig, ReLU, Dropout, DropoutConfig,
    },
    optim::{Optimizer, AdamConfig, GradientsParams},
    record::{Recorder, CompactRecorder},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
    train::{metric::{AccuracyMetric, LossMetric}, ClassificationOutput, TrainStep, ValidStep, TrainOutput, LearnerBuilder},
};

use std::error::Error;
use std::result::Result;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::io::Result as OtherResult;
use std::vec::Vec;
use serde::Deserialize;
use std::io::Cursor;
use byteorder::BigEndian;
use byteorder::ReadBytesExt;
use rand::{seq::IteratorRandom, thread_rng};
use extendr_macros::R;
use extendr_api::eval_string;
use flate2::read::GzDecoder;



const WIDTH: usize = 28;
const HEIGHT: usize = 28;



#[derive(Deserialize, Debug, Clone)]
pub struct MNISTItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: usize,
}


pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}



#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Normalize: make between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MNISTBatch { images, targets }
    }
}


// #[derive(Clone, Debug)]
// pub struct MyBatch<B: Backend> {
//     pub images: Tensor<B, 3>,
//     pub targets: Tensor<B, 1, Int>,
// }


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}


#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}


impl ModelConfig {
    /// Returns the initialized model using the recorded weights.
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes)
                .init_with(record.linear2),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);


        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: Backend + burn::tensor::backend::AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

type MappedLocalDataset = MapperDataset<InMemDataset<MNISTItemRaw>, LocalBytesToImage, MNISTItemRaw>;


pub struct LocalBytesToImage;

impl Mapper<MNISTItemRaw, MNISTItem> for LocalBytesToImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &MNISTItemRaw) -> MNISTItem {
        let image = &item.image_bytes;

        // Convert the image to a 2D array of floats.
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in image.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[y][x] = *pixel as f32;
        }

        MNISTItem {
            image: image_array,
            label: item.label,
        }
    }
}


pub struct MNISTLocalDataset {
    pub dataset: MappedLocalDataset,
}


impl Dataset<MNISTItem> for MNISTLocalDataset {
    fn get(&self, index: usize) -> Option<MNISTItem> {
        self.dataset.get(index)
        // None
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}


fn get_library_directory() -> Result<Box<String>, Box<dyn Error>>{
    let path_result = R!(r#"installed.packages()[rownames(installed.packages())=='viewmastR'][2]"#)?;
    let path = match path_result.as_string_vector(){
        Some(st) => st,
        None => panic!("Cannot unwrap as string for Robj: {:?}", path_result),
    };
    Ok(Box::new(path.first().unwrap().to_string()))
}

impl MNISTLocalDataset{
    pub fn new(dataset: &str) -> Self {
        let file_tup;
        if dataset == "train" {
            file_tup = ("train-images.idx3-ubyte.gz", "train-labels.idx1-ubyte.gz")
        } else {
            file_tup = ("t10k-images.idx3-ubyte.gz", "t10k-labels.idx1-ubyte.gz")
        }
        let root_dir_temp  = *get_library_directory().unwrap();
        let root_dir = Path::new(root_dir_temp.as_str());
        let mut filename = root_dir.join(Path::new("viewmastR")).join("extdata").join("mnist").join(Path::new(file_tup.1));
        // .join(Path::new("viewmastR/").join(Path::new(file_tup.0)));
        // eprintln!("path: {:?}", filename);
        let label_data = &LocalMnistData::new(&(File::open(filename).unwrap())).unwrap();
        filename = root_dir.join(Path::new("viewmastR")).join("extdata").join("mnist").join(Path::new(file_tup.0));
        let images_data = &LocalMnistData::new(&(File::open(filename).unwrap())).unwrap();
        let data = local_mnist_to_image(images_data, label_data);
        let dataset: MapperDataset<InMemDataset<MNISTItemRaw>, LocalBytesToImage, MNISTItemRaw> = MapperDataset::new(data.unwrap(), LocalBytesToImage);
        Self{ dataset }
    }

    pub fn train() -> Self {
        Self::new("train")
    }
    pub fn test() -> Self {
        Self::new("test")
    }
}

pub fn train_local<B: Backend + burn::tensor::backend::AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Save without error");

    B::seed(config.seed);

    let batcher_train = MNISTBatcher::<B>::new(device.clone());
    let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    let train_data = MNISTLocalDataset::train();
    // eprintln!("local dataset type: {:?}", print_type_of(&train_data));
    let test_data = MNISTLocalDataset::test();
    // eprintln!("local test dataset type {:?}", print_type_of(&test_data));

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_data);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_data);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer( CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

   let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");
}



#[derive(Debug)]
pub struct LocalMnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl LocalMnistData {
    fn new(f: &File) -> OtherResult<LocalMnistData> {
        // taken from: https://ngoldbaum.github.io/posts/loading-mnist-data-in-rust/
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(LocalMnistData { sizes, data })
    }
}


pub fn local_mnist_to_image(images_data: &LocalMnistData, label_data: &LocalMnistData) -> OtherResult<InMemDataset<MNISTItemRaw>> {
    let mut images: Vec<Vec<u8>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        images.push(image_data);
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut raw_vec: Vec<MNISTItemRaw> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        raw_vec.push(MNISTItemRaw {
            image_bytes: image.clone(),
            label: classification as usize,
        })
    }

    Ok(InMemDataset::new(raw_vec))
}

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
    let config =
        TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("A config exists");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Failed to load trained model");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let label = item.label;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    eprintln!("Predicted {} Expected {}", predicted, label);
}



pub fn run_burn() -> OtherResult<()> {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    // type SelectedMetricsRenderer = CliMetricsRenderer;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/mnist_local";
    let myconfig = TrainingConfig{
        model: ModelConfig{num_classes: 10, hidden_size: 512, dropout: 0.1}, 
        optimizer: AdamConfig::new(),
        num_epochs: 6,
        batch_size: 64,
        num_workers: 4,
        seed: 42,
        learning_rate: 1.0e-4, 
    };

    train_local::<MyAutodiffBackend>(
        artifact_dir,
        myconfig,
        device.clone());

    // eprintln!("get42: {:?}", MNISTLocalDataset::test().get(42));
    let mut rng = thread_rng();
    let v: Vec<usize> = (1..10000).collect();
    let sample: Vec<&usize> = v.iter().choose_multiple(&mut rng, 30);
    for getno in sample {
        infer::<MyBackend>(
            artifact_dir,
            device.clone(),
            MNISTDataset::test()
                .get(getno.clone())
                .unwrap()
        );
    }
    Ok(())
}




#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 6)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}



// fn print_type_of<T>(_: &T)-> String {
//     format!("{}", std::any::type_name::<T>())
// }

// pub fn set_backend(backend_str: String)-> dyn Backend {
//     let backend_out = match &backend_str {
//         "wgpu" => Wgpu<AutoGraphicsApi, f32, i32>,
//         _ => Wgpu<AutoGraphicsApi, f32, i32>,
//     };
//     return backend_out
// }


pub fn run_custom() {
        type MyBackend = NdArray<f32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;
        let device = burn::backend::ndarray::NdArrayDevice::default();

    // Create the configuration.
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);
    let artifact_dir = "/tmp/mnist_local";

    MyAutodiffBackend::seed(config.seed);

    // Create the model and optimizer.
    let mut model: Model<MyAutodiffBackend> = config.model.init();
    let mut optim = config.optimizer.init();

    // Create the batcher.
    let batcher_train = MNISTBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = MNISTBatcher::<MyBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTLocalDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTLocalDataset::test());

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.images);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            eprintln!(
                "[Train - Epoch {} - Iteration {}] Loss {:.4} | Accuracy {:.4} %",
                epoch,
                iteration,
                loss.clone().into_scalar(),
                accuracy,
            );

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.images);
            let loss = CrossEntropyLoss::new(None).forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            eprintln!(
                "[Valid - Epoch {} - Iteration {}] Loss {:.4} | Accuracy {:.4}",
                iteration,
                epoch,
                loss.clone().into_scalar(),
                accuracy,
            );
        }
    }

    let mut rng = thread_rng();
    let v: Vec<usize> = (1..10000).collect();
    let sample: Vec<&usize> = v.iter().choose_multiple(&mut rng, 30);
    for getno in sample {
        let item = MNISTDataset::test()
                                            .get(getno.clone())
                                            .unwrap();
        let label = item.label;
        let batcher = MNISTBatcher::new(device.clone());
        let batch = batcher.batch(vec![item]);
        let output = &model.forward(batch.images);
        let predicted = output.clone().argmax(1).flatten::<1>(0, 1).into_scalar();
    
        eprintln!("Predicted {} Expected {}", predicted, label);
    }
    let _ = &model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");

}

/// Create out own accuracy metric calculation.
fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
    let predictions = output.argmax(1).squeeze(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}
