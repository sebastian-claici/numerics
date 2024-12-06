use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct NotPositiveDefiniteError;

impl Display for NotPositiveDefiniteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix is not positive definite.")
    }
}
