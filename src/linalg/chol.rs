use crate::core::error::NotPositiveDefiniteError;
use crate::core::matrix::Matrix;

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
                for i in 1..n {
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

    #[test]
    fn test_cholesky() {}
}
