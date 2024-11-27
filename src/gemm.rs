use std::default::Default;
use std::ops::{Add, AddAssign, Mul};

use crate::Matrix;

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

    let mut data = vec![T::default(); lhs.n_rows * rhs.n_cols];
    data.chunks_mut(lhs.n_rows)
        .enumerate()
        .for_each(|(i, c_row)| {
            for k in 0..lhs.n_cols {
                for j in 0..rhs.n_cols {
                    c_row[j] += lhs[(i, k)] * rhs[(k, j)];
                }
            }
        });

    Matrix {
        n_rows: lhs.n_rows,
        n_cols: rhs.n_cols,
        data,
    }
}
