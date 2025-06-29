
use std::ops::Div;
use extendr_api::*;
use burn::tensor::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
// use serde::de;
use statrs::function::gamma::ln_gamma;
use extendr_api::prelude::Scalar as RobjScalar;

pub type B = Autodiff<NdArray<f32>>;

struct Scalar { value: f64 }

#[derive(Debug, PartialEq)]
struct Vector { value: Vec<f64> }

impl Div<Scalar> for Vector {
    type Output = Vector;

    fn div(self, rhs: Scalar) -> Vector {
        Vector { value: self.value.iter().map(|v| v / rhs.value).collect() }
    }
}


pub fn rmat_to_tensor(mat: Robj, device: &NdArrayDevice) -> Result<Tensor<B, 2>> {
    let dims = mat.dim().ok_or("dim attribute missing")?;
    if dims.len() != 2 { return Err("object must be 2-D".into()); }

    let r = dims[0].inner() as usize;  // Rint → &i32 → usize
    let c = dims[1].inner() as usize;

    let col_major: Vec<f32> = mat
        .as_real_vector()
        .ok_or("matrix must be numeric")?
        .iter()
        .map(|x| *x as f32)
        .collect();

    let mut row_major = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            row_major[i * c + j] = col_major[j * r + i];
        }
    }

    Ok(Tensor::<B, 2>::from_floats(row_major.as_slice(), device))
}

pub fn lgamma_plus_one(
    k: &Tensor<B, 2>,
    device: &NdArrayDevice,
) -> Tensor<B, 2> {
    // ── 1. pull raw values ------------------------------------------------
    let vals: Vec<f32> = k.to_data().convert::<f32>().to_vec().expect("failed to convert tensor data");

    // ── 2. lgamma(k + 1)
    let out: Vec<f32> = vals
        .iter()
        .map(|&x| ln_gamma(x as f64 + 1.0) as f32)
        .collect();

    // ── 3. wrap back into a tensor ---------------------------------------
    let shape = k.dims();                                // [rows, cols]
    Tensor::<B, 2>::from_floats(out.as_slice(), device)
        .reshape(shape)
}

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
