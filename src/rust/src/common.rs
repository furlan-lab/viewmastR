#![allow(dead_code)]

use extendr_api::{r, NULL};

use burn::{
    data::{dataloader::{ Dataset, batcher::Batcher}, dataset::transform::Mapper},
    tensor::{backend::Backend, ElementConversion, Int, Tensor},
};


use serde::{Serialize, Deserialize};
use extendr_api::Robj;
use extendr_api::Conversions;
use std::fs::File;
use rmp_serde::{encode::{write_named}, decode::from_read};   // MessagePack writer


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
pub struct RExport {
    pub lr: f64,
    pub hidden_size: Vec<usize>,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub probs: Vec<f32>,
    pub train_history: History,
    pub test_history: History,
    pub training_duration: f64,
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
    pub fn batch_update(&mut self, num_corrects: usize, num_predictions: usize, loss_scalar: f64) -> &mut Self {
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
// pub fn create_tensor(data: Vec<Vec<f64>>) -> Tensor<Wgpu, 2> {
//     let flattened_data: Vec<f32> = data.iter().flatten().map(|&x| x as f32).collect();
//     let shape = Shape::from(vec![data.len() as i64, data[0].len() as i64]);
//     Tensor::<Wgpu, 2>::from_data(Data::new(flattened_data, shape))
// }


pub fn extract_scitemraw(data: &Robj, target_value: Option<i32>) -> Vec<SCItemRaw> {
    if data == &r!(NULL) {
        return vec![];
    }
    data.as_list().unwrap().iter().filter_map(|(_, item_robj)| {
        let list_items = item_robj.as_list().expect("Failed to unlist R object data");
        let data = list_items[0].as_real_vector().expect("Failed to extract data vector");
        let target = target_value.or_else(|| list_items.get(1)
            .and_then(|item| item.as_real_vector().map(|vec| vec[0] as i32))
        )?;
        Some(SCItemRaw { data, target })
    }).collect()
}

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

impl<B: Backend> Batcher<B, SCItem, SCBatch<B>> for SCBatcher<B>  {

    fn batch(&self, items: Vec<SCItem>, device: &B::Device) -> SCBatch<B> {
        let n: usize = items.first().unwrap().counts.len();
        // let counts = items
        //     .iter()
        //     .map(|item| Data::<f64, 1>::from(&item.counts[0..n]))
        //     .map(|data| Tensor::<B, 1>::from_data(data.convert()).reshape([1, n]), device)
        //     .collect();
        
        let row_tensors: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| {
                // convert Vec<f64> -> Vec<B::FloatElem> in a backend-agnostic way
                let row: Vec<_> = item.counts.iter().map(|&x| x.elem::<B::FloatElem>()).collect();
                Tensor::<B, 1>::from_floats(row.as_slice(), device).reshape([1, n])
            })
            .collect();

        let counts = Tensor::cat(row_tensors, 0).to_device(&self.device);

        // let targets = items
        //     .iter()
        //     .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i32).elem()]), &self.device))
        //     .collect();
        let labels: Vec<i32> = items.iter().map(|item| item.label).collect();
        let targets = Tensor::<B, 1, Int>::from_ints(labels.as_slice(), device).to_device(&self.device);
        // let counts = Tensor::cat(counts, 0).to_device(&self.device);
        // let targets = Tensor::cat(targets, 0).to_device(&self.device);

        SCBatch { counts, targets }
    }
}


pub struct LocalCountstoMatrix;
impl Mapper<SCItemRaw, SCItem> for LocalCountstoMatrix {
    fn map(&self, item: &SCItemRaw) -> SCItem {
        let counts = &item.data;
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

pub struct LocalCountstoMatrixWithIndex;

impl Mapper<(usize, SCItemRaw), SCItem> for LocalCountstoMatrixWithIndex {
    fn map(&self, item: &(usize, SCItemRaw)) -> SCItem {
        let (_index, raw_item) = item;  // We're ignoring the index here
        let counts = &raw_item.data;

        SCItem {
            counts: counts.to_vec(),  // Convert counts from SCItemRaw to SCItem
            label: raw_item.target,   // Copy the label
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
    pub struct ModelMeta {
        pub feature_names: Vec<String>,
        pub class_labels:  Vec<String>,   // or Vec<i32> if you prefer
    }

pub fn save_artifacts(
    artifact_dir: &str,
    feature_names: Vec<String>,
    class_labels:  Vec<String>,
) -> std::io::Result<()> {
    // ---- 1. weights (exactly what you already do) -------------------------
    // model.save_file(format!("{}/model", artifact_dir), &recorder)?;

    // ---- 2. side-car metadata --------------------------------------------
    let meta = ModelMeta { feature_names, class_labels };
    eprintln!("Feature names and labels saved to {}/meta.mpk", artifact_dir);
    let mut file = File::create(format!("{}/meta.mpk", artifact_dir))?;
    write_named(&mut file, &meta).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;   // compact MessagePack
    Ok(())
}

fn load_meta(artifact_dir: &str) -> anyhow::Result<ModelMeta> {
    let file = File::open(format!("{}/meta.mpk", artifact_dir))?;
    let meta: ModelMeta = from_read(file)?;
    Ok(meta)
}
