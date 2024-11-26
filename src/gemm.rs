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

    // Transpose rhs for cache locality
    let rhs_t = rhs.transpose();

    let mut result = Matrix::new(lhs.n_rows, rhs.n_cols);
    for bi in (0..lhs.n_rows).step_by(BLOCK_SIZE) {
        for bj in (0..rhs.n_cols).step_by(BLOCK_SIZE) {
            for bk in (0..rhs.n_rows).step_by(BLOCK_SIZE) {
                for i in bi..bi + BLOCK_SIZE {
                    for j in bj..bj + BLOCK_SIZE {
                        let mut acc = T::default();
                        for k in bk..bk + BLOCK_SIZE {
                            acc += lhs[(i, k)] * rhs_t[(j, k)];
                        }
                        result[(i, j)] += acc;
                    }
                }
            }
        }
    }

    result
}
