use crate::core::error::LUDecompositionError;
use crate::core::error::LUDecompositionError::NotSquareError;
use crate::core::matrix::*;

pub(crate) trait LU<T> {
    fn lu(&self) -> Result<(Matrix<T>, Matrix<T>), LUDecompositionError>;
}

#[macro_export]
macro_rules! impl_lu {
    ($type:ty) => {
        impl LU<$type> for Matrix<$type> {
            fn lu(&self) -> Result<(Matrix<$type>, Matrix<$type>), LUDecompositionError> {
                if (self.n_rows != self.n_cols) {
                    return Err(NotSquareError);
                }
                let mut lu_l = Matrix::new(self.n_rows, self.n_cols);
                let mut lu_u = Matrix::new(self.n_rows, self.n_cols);

                for i in (0..self.n_rows) {
                    lu_l[(i, i)] = 1.0;
                }

                for j in (0..self.n_rows) {
                    // Forward pass
                    for i in (0..=j) {
                        let mut s_term = 0.0;
                        for k in (0..i) {
                            s_term += lu_l[(i, k)] * lu_u[(k, j)];
                        }
                        lu_u[(i, j)] = self[(i, j)] - s_term;
                    }

                    // Backward pass
                    for i in (j + 1..self.n_rows) {
                        let mut s_term = 0.0;
                        for k in (0..i) {
                            s_term += lu_l[(i, k)] * lu_u[(k, j)];
                        }
                        lu_l[(i, j)] = 1.0 / lu_u[(j, j)] * (self[(i, j)] - s_term);
                    }
                }

                Ok((lu_l, lu_u))
            }
        }
    };
}

impl_lu!(f32);
impl_lu!(f64);

#[cfg(test)]
mod test {
    use crate::core::gemm::gemm;

    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lu() {
        let m: Matrix<f32> = matrix![4.0, 3.0; 6.0, 3.0];

        let lu = m.lu();
        assert!(lu.is_ok());

        let (lu_l, lu_u) = lu.unwrap();
        assert_relative_eq!(lu_l[(0, 0)], 1.0);
        assert_relative_eq!(lu_l[(1, 0)], 1.5);
        assert_relative_eq!(lu_l[(1, 1)], 1.0);
        assert_relative_eq!(lu_u[(0, 0)], 4.0);
        assert_relative_eq!(lu_u[(0, 1)], 3.0);
        assert_relative_eq!(lu_u[(1, 1)], -1.5);

        let m_rec = gemm(&lu_l, &lu_u);
        assert_eq!(m_rec.n_rows, m.n_rows);
        assert_eq!(m_rec.n_cols, m.n_cols);
        for i in 0..m_rec.n_rows {
            for j in 0..m_rec.n_cols {
                assert_relative_eq!(m_rec[(i, j)], m[(i, j)]);
            }
        }
    }

    #[test]
    fn test_lu_not_square() {
        let m: Matrix<f32> = matrix![1.0, 1.0, 0.0; 0.0, 1.0, 1.0];

        let lu = m.lu();
        assert!(lu.is_err());
        assert_eq!(lu.unwrap_err(), NotSquareError);
    }
}
