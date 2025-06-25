
use std::ops::Div;
// use extendr_api::*;
// use extendr_api::prelude::{Rint, Scalar as RScalar};
// use burn::tensor::{Data, Tensor};
// use statrs::function::gamma::ln_gamma;
// use burn::backend::ndarray::NdArray;
/// NdArray CPU backend is the simplest to wire up from R
// pub type B = NdArray<f32>;


struct Scalar { value: f64 }

#[derive(Debug, PartialEq)]
struct Vector { value: Vec<f64> }

impl Div<Scalar> for Vector {
    type Output = Vector;

    fn div(self, rhs: Scalar) -> Vector {
        Vector { value: self.value.iter().map(|v| v / rhs.value).collect() }
    }
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


// /// R matrix (column-major) → Burn tensor (row-major)
// pub fn rmat_to_tensor(mat: Robj) -> Result<Tensor<B, 2>> {
//     // 1. dim attribute
//     let dims = mat
//         .dim()
//         .ok_or("object is not a matrix (dim attribute missing)")?;
//     if dims.len() != 2 {
//         return Err("matrix must be 2-D".into());
//     }
//     // let r = i32::from(dims[0]) as usize;
//     // let c = i32::from(dims[1]) as usize;
//     let r: Rint = dims[0];
//     let c: Rint = dims[1];
//     let r = r.inner() as usize;   // number of rows
//     let c = c.inner() as usize;   // number of cols

//     // 2. numeric payload
//     let col_major: Vec<f32> = mat
//         .as_real_vector()
//         .ok_or("matrix must be numeric")?
//         .iter()
//         .map(|x| *x as f32)
//         .collect();
//     if col_major.len() != r * c {
//         return Err("length(data) != prod(dim)".into());
//     }

//     // 3. transpose to row-major
//     let mut row_major = vec![0f32; r * c];
//     for i in 0..r {
//         for j in 0..c {
//             row_major[i * c + j] = col_major[j * r + i];
//         }
//     }

//     // 4. tensor
//     let data = Data::<f32, 2>::new(row_major, burn::tensor::Shape { dims: [r, c] });
//     Ok(Tensor::<B, 2>::from_data(data))
// }

// /// lgamma(k + 1) pre-computed on CPU
// pub fn lgamma_plus_one(k: &Tensor<B, 2>) -> Tensor<B, 2> {
//     let v   = k.to_data().convert::<f32>().value;
//     let out: Vec<f32> = v.iter()
//         .map(|&x| ln_gamma(x as f64 + 1.0) as f32)
//         .collect();
//     let shape = k.dims();
//     let data  = Data::<f32, 2>::new(out, burn::tensor::Shape { dims: shape });
//     Tensor::<B, 2>::from_data(data)
// }
