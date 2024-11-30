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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small() {
        let a = Matrix::from_gen(2, 2, |i, j| i + j);
        let b = Matrix::from_gen(2, 2, |i, j| i + j);

        let c = matmul(&a, &b);
        assert_eq!(c[(0, 0)], 1);
        assert_eq!(c[(0, 1)], 2);
        assert_eq!(c[(1, 0)], 2);
        assert_eq!(c[(1, 1)], 5);
    }

    #[test]
    fn matmul_dims() {
        let a = Matrix::from_gen(2, 3, |i, j| i + j);
        let b = Matrix::from_gen(3, 2, |i, j| i + j);

        let c = matmul(&a, &b);
        assert_eq!(c.n_rows, 2);
        assert_eq!(c.n_cols, 2);

        assert_eq!(c[(0, 0)], 5);
        assert_eq!(c[(0, 1)], 8);
        assert_eq!(c[(1, 0)], 8);
        assert_eq!(c[(1, 1)], 14);
    }
}
