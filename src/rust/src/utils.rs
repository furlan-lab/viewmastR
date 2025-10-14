
#![allow(dead_code)]
use std::ops::Div;
// use extendr_api::*;
// use burn::prelude::Device;
// use burn::tensor::Tensor;
// use burn::backend::{ndarray::NdArray, candle::Candle};
// use burn::backend::ndarray::NdArray;

// use burn::backend::Autodiff;
// use serde::de;
// use statrs::function::gamma::ln_gamma;
// use extendr_api::prelude::Scalar as RobjScalar;

// pub type B = Autodiff<NdArray<f32>>;
// pub type B = Autodiff<Candle<f32>>;

struct Scalar { value: f64 }

#[derive(Debug, PartialEq)]
struct Vector { value: Vec<f64> }

impl Div<Scalar> for Vector {
    type Output = Vector;

    fn div(self, rhs: Scalar) -> Vector {
        Vector { value: self.value.iter().map(|v| v / rhs.value).collect() }
    }
}


// pub fn rmat_to_tensor(mat: Robj, device: &Device<B>) -> Result<Tensor<B, 2>> {
//     // 1) get dims
//     let dims = mat.dim().ok_or("dim attribute missing")?;
//     if dims.len() != 2 {
//         return Err("matrix must be 2-D".into());
//     }
//     let (r, c) = (dims[0].inner() as usize, dims[1].inner() as usize);
//     // eprintln!("▶ rmat_to_tensor: target shape = [{}, {}]", r, c);

//     // 2) pull data
//     let col_major: Vec<f32> = if let Some(rv) = mat.as_real_vector() {
//         rv.iter().map(|&x| x as f32).collect()
//     } else if let Some(iv) = mat.as_integer_vector() {
//         iv.iter().map(|&x| x as f32).collect()
//     } else {
//         return Err("matrix must be numeric or integer".into());
//     };
//     // eprintln!("▶ rmat_to_tensor: col_major.len() = {}", col_major.len());

//     // 3) build row_major
//     let mut row_major = vec![0f32; r * c];
//     for i in 0..r {
//         for j in 0..c {
//             row_major[i * c + j] = col_major[j * r + i];
//         }
//     }

//     // 4) from_floats → rank-1
//     let t1 = Tensor::<B, 1>::from_floats(row_major.as_slice(), device);
//     // eprintln!("▶ after from_floats: t1.dims() = {:?}", t1.dims());

//     // 5) bump to rank-2 via unsqueeze
//     let t2 = t1.unsqueeze::<2>();
//     // eprintln!("▶ after unsqueeze: t2.dims() = {:?}", t2.dims());

//     // 6) reshape into [r, c]
//     let t3 = t2.reshape([r, c]);
//     // eprintln!("▶ after reshape: t3.dims() = {:?}", t3.dims());

//     Ok(t3)
// }


// pub fn lgamma_plus_one(
//     k: &Tensor<B, 2>,
//     device: &Device<B>,
// ) -> Tensor<B, 2> {
//     // ── 1. pull raw values ------------------------------------------------
//     let vals: Vec<f32> = k.to_data().convert::<f32>().to_vec().expect("failed to convert tensor data");

//     // ── 2. lgamma(k + 1)
//     let out: Vec<f32> = vals
//         .iter()
//         .map(|&x| ln_gamma(x as f64 + 1.0) as f32)
//         .collect();

//     // ── 3. wrap back into a tensor ---------------------------------------
//     // 3a) build a 1D tensor of length r*c
//     let t1 = Tensor::<B, 1>::from_floats(out.as_slice(), device);
//     // 3b) bump it to rank=2: [r*c] -> [1, r*c]
//     let t2 = t1.unsqueeze::<2>();
//     // 3c) reshape into [r, c]
//     t2.reshape(k.dims())
// }

// deprecated v. 0.2.1
// pub fn sparse_row_variances(j: Vec<usize>, val: Vec<f64>, rm: Vec<f64>, n: usize) -> Vec<f64> {
//     let nv = j.len();
//     let nm = rm.len();
//     let mut rv:Vec<f64>  = vec![0.0; nm];
//     let mut rit:Vec<f64> = vec![0.0; nm];

//     // Calculate RowVars Initial
//     for i in 0..nv {
//         let current = j[i] - 1;
//         rv[current] = rv[current] + (val[i] - rm[current]).powi(2);
//         rit[current] += 1.0;
//     }

//     // Calculate Remainder Variance
//     for i in 0..nm {
//         rv[i] += (n as f64 - rit[i]) * rm[i].powi(2);
//     }
    
//     let return_v =  Vector{value: rv} / Scalar{ value: (n as f64 - 1.0)};
//     return_v.value
// }

pub fn sparse_row_variances(j: Vec<usize>, val: Vec<f64>, rm: Vec<f64>, n: usize) -> Vec<f64> {
    // let nv = j.len();
    let nm = rm.len();
    let mut rv = vec![0.0; nm];
    let mut rit = vec![0.0; nm];

    // Calculate variances in one pass
    for (index, &row_index) in j.iter().enumerate() {
        let current = row_index - 1;
        rv[current] += (val[index] - rm[current]).powi(2);
        rit[current] += 1.0;
    }

    // Adjust for remaining variance and normalize
    rv.iter_mut().enumerate().for_each(|(i, r)| {
        *r += (n as f64 - rit[i]) * rm[i].powi(2);
        *r /= n as f64 - 1.0;
    });

    rv
}
