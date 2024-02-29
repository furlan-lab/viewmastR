use serde::Deserialize;


pub fn mean(numbers: &Vec<f64>) -> f64 {
    numbers.iter().sum::<f64>() as f64 / numbers.len() as f64
}

#[derive(Debug, Clone)]
pub struct SCItem {
    pub counts: Vec<f64>, 
    // pub counts: [f32; 3600], 
    pub label: i32,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct SCItemRaw {
    // pub counts: Vec<f64>, 
    pub data: Vec<f64>, 
    pub target: i32,
}


#[derive(Debug)]
pub struct History {
    pub acc: Vec<f64>,
    pub loss: Vec<f64>,
}

impl History {
    pub fn new() -> Self {
        Self {
            acc: vec![],
            loss: vec![],
        }
    }
}



#[derive(Debug)]

pub struct ModelRExport {
    pub lr: f64,
    pub hidden_size: Vec<usize>,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub predictions: Vec<i32>,
    pub train_history: History,
    pub test_history: History,
    pub training_duration: f64,
    pub probs: Option<Vec<Vec<f32>>>,
}

#[derive(Debug)]
pub struct ModelAccuracy {
    iter_count: usize,
    pub epoch: usize,
    num_correct: usize,
    num_predictions: usize,
    loss: Vec<f64>,
}

// impl ModelAccuracy method to update
impl ModelAccuracy {
    pub fn new() -> Self {
        Self {
            iter_count: 0 as usize,
            epoch: 0 as usize,
            num_correct: 0 as usize,
            num_predictions: 0 as usize,
            loss: vec![],
        }
    }
    pub fn epoch_reset(&mut self, epoch: usize) -> &mut Self {
        self.num_correct = 0;
        self.num_predictions = 0;
        self.epoch = epoch;
        self
    }
    pub fn batch_update(&mut self, num_corrects: i64, num_predictions: usize, loss_scalar: f64) -> &mut Self {
        self.iter_count += 1;
        self.num_correct += num_corrects as usize;
        self.num_predictions += num_predictions;
        self.loss.push(loss_scalar);
        self
    }
    pub fn epoch_update(&mut self, history: & mut History) -> &mut Self {
        history.acc.push(self.num_correct as f64 / self.num_predictions as f64);
        history.loss.push(mean(&self.loss));
        self
    }
}

pub fn emit_metrics(train_accuracy: &ModelAccuracy, test_accuracy: &ModelAccuracy) {
    eprintln!(
        "[Epoch - {} | Training Accuracy {:.3}% | Validation Accuracy {:.3}%]",
        train_accuracy.epoch,
        train_accuracy.num_correct as f32 / train_accuracy.num_predictions as f32 * 100.0,
        test_accuracy.num_correct as f32 / test_accuracy.num_predictions as f32 * 100.0
    );
}

// #[derive(Config)]
// pub struct SCTrainingConfig {
//     pub num_epochs: usize,
//     #[config(default = 64)]
//     pub batch_size: usize,
//     #[config(default = 4)]
//     pub num_workers: usize,
//     #[config(default = 42)]
//     pub seed: u64,
//     pub lr: f64,
//     pub model: ModelConfig,
//     pub optimizer: AdamConfig,
// }

// #[derive(Config, Debug)]
// pub struct ModelConfig {
//     num_classes: usize,
//     num_epochs: usize,
//     hidden_size: usize,
//     #[config(default = "0.5")]
//     dropout: f64,
//     learning_rate: f64,
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
//     pub fn init<B: Backend>(&self, no_features: usize) -> Model<B> {
//         Model {
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(no_features, self.hidden_size).init(),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
//         }
//     }
// }