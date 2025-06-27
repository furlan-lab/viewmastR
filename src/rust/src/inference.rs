
use crate::common::*;
use crate::scrna_mlr::ModelConfig as MlrCfg;
use crate::scrna_ann ::ModelConfig as AnnCfg;
use crate::scrna_ann2l::ModelConfig as Ann2Cfg;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::{InMemDataset, transform::MapperDataset}},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Tensor, DType, backend::Backend as BurnBackend},
};
use num_traits::ToPrimitive;



#[derive(Debug)]
pub enum NetKind {
    Mlr,
    Ann  { hidden: usize },
    Ann2 { hidden1: usize, hidden2: usize },
}


fn run_inference<B, F>(
    mut f      : F,
    device     : <B as BurnBackend>::Device,
    query      : Vec<SCItemRaw>,
    batch_size : usize,
) -> Vec<f32>
where
    B: BurnBackend + 'static,
    B::FloatElem: ToPrimitive,      // ← was Elem, now correct
    F: FnMut(Tensor<B, 2>) -> Tensor<B, 2>,
{
    let dataset = MapperDataset::new(InMemDataset::new(query), LocalCountstoMatrix);
    let loader  = DataLoaderBuilder::new(SCBatcher::<B>::new(device))
        .batch_size(batch_size)
        .build(dataset);


    loader
        .iter()
        .flat_map(|batch| {
            // 1.  Run the model
            let mut data = f(batch.counts).into_data();   // TensorData

            // 2.  Make sure the buffer is f32-typed
            if data.dtype != DType::F32 {
                data = data.convert::<f32>();             // still TensorData
            }

            // 3.  Take the buffer out as a Vec<f32>
            let vec = data.into_vec::<f32>().unwrap();    // Vec<f32>

            vec.into_iter()
        })
        .collect()
}


pub fn infer<B>(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
    device : <B as BurnBackend>::Device,
) -> Vec<f32>
where
    B: BurnBackend + 'static,
    B::FloatElem: ToPrimitive,
{
    let bs = batch_size.unwrap_or(64);

    match net {
        NetKind::Mlr => {
            let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                           .load(model_path.into(), &device)
                           .expect("load MLR weights");
            let model = MlrCfg::new(num_classes)
                           .init(num_features, device.clone())
                           .load_record(rec);

            run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
        }

        NetKind::Ann { hidden } => {
            let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                           .load(model_path.into(), &device)
                           .expect("load ANN-1L weights");
            let model = AnnCfg::new(num_classes, 0, hidden, 0.0)
                           .init(num_features, device.clone())
                           .load_record(rec);

            run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
        }

        NetKind::Ann2 { hidden1, hidden2 } => {
            let rec   = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                           .load(model_path.into(), &device)
                           .expect("load ANN-2L weights");
            let model = Ann2Cfg::new(num_classes, 0, hidden1, hidden2, 0.0)
                           .init(num_features, device.clone())
                           .load_record(rec);

            run_inference::<B, _>(move |x| model.forward(x), device, query, bs)
        }
    }
}


pub fn infer_wgpu(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
) -> Vec<f32> {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    type B = Wgpu<f32, i32>;
    infer::<B>(model_path,
        net,
        num_classes,
        num_features,
        query,
        batch_size, WgpuDevice::default())
}

pub fn infer_nd(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
) -> Vec<f32> {
    use burn::backend::ndarray::{NdArray, NdArrayDevice}; 
    type B = NdArray<f32, i32>;
    infer::<B>(model_path,
        net,
        num_classes,
        num_features,
        query,
        batch_size, NdArrayDevice::default())
}

pub fn infer_candle(
    model_path   : &str,
    net          : NetKind,
    num_classes  : usize,
    num_features : usize,
    query        : Vec<SCItemRaw>,
    batch_size   : Option<usize>,
) -> Vec<f32> {
    use burn::backend::candle::{Candle, CandleDevice}; 
    type B = Candle<f32, i64>;
    infer::<B>(model_path,
        net,
        num_classes,
        num_features,
        query,
        batch_size, CandleDevice::default())
}

