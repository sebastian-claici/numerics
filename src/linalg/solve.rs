use std::ops::{Div, DivAssign, Mul, SubAssign};

use crate::core::error::LUDecompositionError;
use crate::core::matrix::Matrix;
use crate::core::vector::Vector;
use crate::linalg::chol::Cholesky;

use super::lu::LU;

pub trait Solve<T> {
    fn solve(&self, rhs: &Vector<T>) -> Result<Vector<T>, LUDecompositionError>;
}

fn bsub<T>(lhs: &Matrix<T>, rhs: &Vector<T>) -> Vector<T>
where
    T: Copy
        + Clone
        + Default
        + Mul<Output = T>
        + Div<Output = T>
        + DivAssign
        + SubAssign
        + PartialEq,
{
    // Check that lhs is upper triangular
    for i in 0..lhs.n_rows {
        for j in 0..i {
            if lhs[(i, j)] != T::default() {
                panic!("Back-substitution matrix is not upper triangular");
            }
        }
    }

    let mut result = Vector::from_vec(&rhs.data);
    for i in (0..rhs.n).rev() {
        result[i] /= lhs[(i, i)];

        let ri = result[i];
        for j in (0..i).rev() {
            result[j] -= ri * lhs[(j, i)] / lhs[(i, i)];
        }
    }

    result
}

fn fsub<T>(lhs: &Matrix<T>, rhs: &Vector<T>) -> Vector<T>
where
    T: Copy
        + Clone
        + Default
        + Mul<Output = T>
        + Div<Output = T>
        + DivAssign
        + SubAssign
        + PartialEq,
{
    // Check that lhs is lower triangular
    for i in 0..lhs.n_rows {
        for j in i + 1..lhs.n_rows {
            if lhs[(i, j)] != T::default() {
                panic!("Forward-substitution matrix is not lower triangular");
            }
        }
    }

    let mut result = Vector::from_vec(&rhs.data);
    for i in 0..rhs.n {
        result[i] /= lhs[(i, i)];

        let ri = result[i];
        for j in i + 1..rhs.n {
            result[j] -= ri * lhs[(j, i)] / lhs[(i, i)];
        }
    }

    result
}

impl Solve<f32> for Matrix<f32> {
    fn solve(&self, rhs: &Vector<f32>) -> Result<Vector<f32>, LUDecompositionError> {
        let (l, u) = if let Ok(chol) = self.chol() {
            (chol.transpose(), chol)
        } else {
            self.lu()?
        };

        let y = bsub(&u, rhs);
        Ok(fsub(&l, &y))
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use crate::matrix;

    use super::*;

    #[test]
    #[should_panic]
    fn test_bsub_panic() {
        let lhs = matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0];
        let b = Vector::from_vec(&vec![1.0, 1.0, 1.0]);

        bsub(&lhs, &b);
    }

    #[test]
    fn test_bsub() {
        let lhs = matrix![1.0, 1.0, 1.0; 0.0, 1.0, 2.0; 0.0, 0.0, 1.0];
        let b = Vector::from_vec(&vec![1.0, 1.0, 1.0]);

        let y = bsub(&lhs, &b);

        assert_relative_eq!(y[2], 1.0);
        assert_relative_eq!(y[1], -1.0);
        assert_relative_eq!(y[0], 1.0);
    }

    #[test]
    #[should_panic]
    fn test_fsub_panic() {
        let lhs = matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0];
        let b = Vector::from_vec(&vec![1.0, 1.0, 1.0]);

        fsub(&lhs, &b);
    }

    #[test]
    fn test_fsub() {
        let lhs = matrix![1.0, 0.0, 0.0; 2.0, 1.0, 0.0; 1.0, 1.0, 1.0];
        let b = Vector::from_vec(&vec![1.0, 1.0, 1.0]);

        let y = fsub(&lhs, &b);

        assert_relative_eq!(y[0], 1.0);
        assert_relative_eq!(y[1], -1.0);
        assert_relative_eq!(y[2], 1.0);
    }
}
