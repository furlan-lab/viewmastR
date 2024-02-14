
use std::ops::Div;

struct Scalar { value: f64 }

#[derive(Debug, PartialEq)]
struct Vector { value: Vec<f64> }

impl Div<Scalar> for Vector {
    type Output = Vector;

    fn div(self, rhs: Scalar) -> Vector {
        Vector { value: self.value.iter().map(|v| v / rhs.value).collect() }
    }
}


pub fn sparse_row_variances(j: Vec<usize>, val: Vec<f64>, rm: Vec<f64>, n: usize) -> Vec<f64> {
    let nv = j.len();
    let nm = rm.len();
    let mut rv:Vec<f64>  = vec![0.0; nm];
    let mut rit:Vec<f64> = vec![0.0; nm];

    // Calculate RowVars Initial
    for i in 0..nv {
        let current = j[i] - 1;
        rv[current] = rv[current] + (val[i] - rm[current]).powi(2);
        rit[current] += 1.0;
    }

    // Calculate Remainder Variance
    for i in 0..nm {
        rv[i] += (n as f64 - rit[i]) * rm[i].powi(2);
    }
    
    let return_v =  Vector{value: rv} / Scalar{ value: (n as f64 - 1.0)};
    return_v.value
}
