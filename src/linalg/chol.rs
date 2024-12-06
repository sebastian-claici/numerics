use crate::core::error::NotPositiveDefiniteError;
use crate::core::matrix::*;

#[derive(Debug)]
pub struct Cholesky<T> {
    pub chol_l: Matrix<T>,
}

impl<T> Cholesky<T> {
    pub fn new(chol_l: Matrix<T>) -> Self {
        Self { chol_l }
    }
}

macro_rules! impl_cholesky {
    ($type:ty) => {
        impl Matrix<$type> {
            pub fn cholesky(&self) -> Result<Cholesky<$type>, NotPositiveDefiniteError> {
                if !self.is_symmetric() {
                    return Err(NotPositiveDefiniteError);
                }

                let n = self.n_rows;
                let mut chol_l = Matrix::new(n, n);
                // assumes A is square and positive semi-definite
                for i in 0..n {
                    let mut diag = self[(i, i)];
                    for k in 0..i {
                        diag -= chol_l[(i, k)] * chol_l[(i, k)];
                    }
                    if diag < 0.0 {
                        return Err(NotPositiveDefiniteError);
                    } else {
                        chol_l[(i, i)] = diag.sqrt();
                        for j in i + 1..n {
                            let mut off_diag = self[(i, j)];
                            for k in 0..i {
                                off_diag -= chol_l[(i, k)] * chol_l[(j, k)];
                            }
                            chol_l[(j, i)] = (1 as $type) / chol_l[(i, i)] * off_diag;
                        }
                    }
                }

                Ok(Cholesky::new(chol_l))
            }
        }
    };
}

impl_cholesky!(f32);
impl_cholesky!(f64);

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cholesky() {
        let m: Matrix<f32> = matrix![4.0, 12.0, -16.0; 12.0, 37.0, -43.0; -16.0, -43.0, 98.0];

        let chol = m.cholesky();
        assert!(chol.is_ok());

        let chol = chol.unwrap();
        assert_relative_eq!(chol.chol_l[(0, 0)], 2.0);
        assert_relative_eq!(chol.chol_l[(1, 0)], 6.0);
        assert_relative_eq!(chol.chol_l[(2, 0)], -8.0);
        assert_relative_eq!(chol.chol_l[(0, 1)], 0.0);
        assert_relative_eq!(chol.chol_l[(1, 1)], 1.0);
        assert_relative_eq!(chol.chol_l[(2, 1)], 5.0);
        assert_relative_eq!(chol.chol_l[(0, 2)], 0.0);
        assert_relative_eq!(chol.chol_l[(1, 2)], 0.0);
        assert_relative_eq!(chol.chol_l[(2, 2)], 3.0);
    }

    #[test]
    fn test_cholesky_non_symmetric() {
        let m: Matrix<f32> = matrix![1.0, 1.0; 0.0, 1.0];

        let chol = m.cholesky();
        assert!(chol.is_err());
    }

    #[test]
    fn test_cholesky_non_pd() {
        let m: Matrix<f32> = matrix![-1.0, 0.0; 0.0, 1.0];

        let chol = m.cholesky();
        assert!(chol.is_err());
    }
}
