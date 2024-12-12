use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum CholDecompositionError {
    #[error("matrix is not symmetric")]
    NotSymmetricError,
    #[error("matrix is not positive definite")]
    NotPositiveDefiniteError,
}

#[derive(Error, Debug, PartialEq)]
pub enum LUDecompositionError {
    #[error("LU decomposition is only implemented for square matrices")]
    NotSquareError,
}
