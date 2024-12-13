use std::default::Default;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul};

use crate::core::matrix::Matrix;
use crate::core::vector::Vector;

pub fn dot<T>(lhs: &Vector<T>, rhs: &Vector<T>) -> T
where
    T: Mul<Output = T> + Copy + Sum,
{
    if lhs.n != rhs.n {
        panic!(
            "Vectors must have the same dimensions. Got {} and {}.",
            lhs.n, rhs.n
        );
    }

    lhs.data
        .iter()
        .zip(rhs.data.iter())
        .map(|(x, y)| *x * *y)
        .sum()
}

pub fn gemv<T>(lhs: &Matrix<T>, rhs: &Vector<T>) -> Vector<T>
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + Default,
{
    if lhs.n_cols != rhs.n {
        panic!(
            "Matrix-vector product with incompatible dimensions: {:?} and {:?}",
            (lhs.n_rows, lhs.n_cols),
            rhs.n
        );
    }

    let mut result = Vector::new(lhs.n_rows);
    for i in 0..lhs.n_rows {
        result[i] = T::default();
        for j in 0..lhs.n_cols {
            result[i] += lhs[(i, j)] * rhs[j];
        }
    }
    result
}

pub fn gemm<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T>
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small() {
        let a = Matrix::from_gen(2, 2, |i, j| i + j);
        let b = Matrix::from_gen(2, 2, |i, j| i + j);

        let c = gemm(&a, &b);
        assert_eq!(c[(0, 0)], 1);
        assert_eq!(c[(0, 1)], 2);
        assert_eq!(c[(1, 0)], 2);
        assert_eq!(c[(1, 1)], 5);
    }

    #[test]
    fn matmul_dims() {
        let a = Matrix::from_gen(2, 3, |i, j| i + j);
        let b = Matrix::from_gen(3, 2, |i, j| i + j);

        let c = gemm(&a, &b);
        assert_eq!(c.n_rows, 2);
        assert_eq!(c.n_cols, 2);

        assert_eq!(c[(0, 0)], 5);
        assert_eq!(c[(0, 1)], 8);
        assert_eq!(c[(1, 0)], 8);
        assert_eq!(c[(1, 1)], 14);
    }
}
