use std::cmp::min;
use std::default::Default;
use std::ops::{Add, AddAssign, Mul};

use crate::Matrix;

const BLOCK_SIZE: usize = 64;

pub fn matmul<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + Default,
{
    if lhs.n_cols != rhs.n_rows {
        panic!(
            "Matrices have incompatible dimensions: {:?} and {:?}",
            (lhs.n_rows, lhs.n_cols),
            (rhs.n_rows, rhs.n_cols)
        );
    }

    let lhs_t = lhs.transpose();

    let mut result = Matrix::new(lhs.n_rows, rhs.n_cols);
    for bi in (0..lhs.n_rows).step_by(BLOCK_SIZE) {
        for bj in (0..rhs.n_cols).step_by(BLOCK_SIZE) {
            for k in 0..rhs.n_rows {
                for i in bi..min(lhs.n_rows, bi + BLOCK_SIZE) {
                    for j in bj..min(rhs.n_cols, bj + BLOCK_SIZE) {
                        result[(i, j)] += lhs_t[(k, i)] * rhs[(k, j)];
                    }
                }
            }
        }
    }

    result
}
