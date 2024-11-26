use std::default::Default;
use std::ops::{Add, Mul};

use crate::Matrix;

pub fn matmul<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default,
{
    if lhs.n_cols != rhs.n_rows {
        panic!(
            "Matrices have incompatible dimensions: {:?} and {:?}",
            (lhs.n_rows, lhs.n_cols),
            (rhs.n_rows, rhs.n_cols)
        );
    }

    let mut result = Matrix::new(lhs.n_rows, rhs.n_cols);
    for i in 0..lhs.n_rows {
        for j in 0..rhs.n_cols {
            for k in 0..rhs.n_rows {
                result[(i, j)] = lhs[(i, k)] * rhs[(k, j)];
            }
        }
    }

    result
}
