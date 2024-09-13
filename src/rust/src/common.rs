#![allow(dead_code)]

use serde::Deserialize;
use burn::tensor::{Shape, Tensor, Data};
use burn::backend::wgpu::Wgpu;
use extendr_api::Robj;
use extendr_api::Conversions;

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


// Function to extract vectors from the R list
pub fn extract_vectors(data: &Robj, index: usize) -> Vec<Vec<f64>> {
    let list = data.as_list().unwrap();
    list.iter()
        .map(|(_, item_robj)| {
            let list_items = item_robj.as_list().unwrap();
            list_items[index].as_real_vector().unwrap()
        })
        .collect()
}

// Function to extract scalar values from the R list
pub fn extract_scalars(data: &Robj, index: usize) -> Vec<usize> {
    let list = data.as_list().unwrap();
    list.iter()
        .map(|(_, item_robj)| {
            let list_items = item_robj.as_list().unwrap();
            list_items[index].as_real().unwrap() as usize
        })
        .collect()
}

// Function to flatten data and create a Tensor
pub fn create_tensor(data: Vec<Vec<f64>>) -> Tensor<Wgpu, 2> {
    let flattened_data: Vec<f32> = data.iter().flatten().map(|&x| x as f32).collect();
    let shape = Shape::from(vec![data.len() as i64, data[0].len() as i64]);
    Tensor::<Wgpu, 2>::from_data(Data::new(flattened_data, shape))
}


pub fn extract_scitemraw(data: &Robj, target_value: Option<i32>) -> Vec<SCItemRaw> {
    let sc_from_list = data.as_list().unwrap();
    sc_from_list
        .iter()
        .map(|(_, item_robj)| {
            let list_items = item_robj.as_list().unwrap();
            let data = list_items[0].as_real_vector().unwrap();
            // Check and compute target per iteration, fall back to list_items[1] safely
            let target = target_value.unwrap_or_else(|| {
                list_items.get(1)
                    .and_then(|item| item.as_real_vector())
                    .map(|vec| vec[0] as i32)
                    .unwrap_or_default() // fallback if index 1 or conversion fails
            });
            SCItemRaw { data, target }
        })
        .collect()
}
