pub mod gemm;

use std::default::Default;
use std::ops::{Add, Index, IndexMut, Sub};

#[derive(Debug)]
pub struct Matrix<T> {
    n_rows: usize,
    n_cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Copy + Clone + Default,
{
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        let data = vec![Default::default(); n_rows * n_cols];
        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    pub fn from_gen(n_rows: usize, n_cols: usize, gen: fn() -> T) -> Self {
        let mut data = Vec::with_capacity(n_rows * n_cols);
        (0..n_rows * n_cols).for_each(|_| {
            data.push(gen());
        });

        Self {
            n_rows,
            n_cols,
            data,
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        if i >= self.n_rows || j >= self.n_cols {
            panic!(
                "Index {:?} out of bounds for matrix of size {:?}",
                index,
                (self.n_rows, self.n_cols)
            );
        }

        &self.data[i * self.n_cols + j]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        if i >= self.n_rows || j >= self.n_cols {
            panic!(
                "Index {:?} out of bounds for matrix of size {:?}",
                index,
                (self.n_rows, self.n_cols)
            );
        }

        &mut self.data[i * self.n_cols + j]
    }
}

impl<T> Add for Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        if self.n_rows != other.n_rows || self.n_cols != other.n_cols {
            panic!(
                "Cannot add matrices of different shapes: ({}, {}) ({}, {})",
                self.n_rows, self.n_cols, other.n_rows, other.n_cols
            );
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.clone() + y.clone())
            .collect();

        Self {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            data,
        }
    }
}

impl<T> Sub for Matrix<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        if self.n_rows != other.n_rows || self.n_cols != other.n_cols {
            panic!(
                "Cannot add matrices of different shapes: ({}, {}) ({}, {})",
                self.n_rows, self.n_cols, other.n_rows, other.n_cols
            );
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.clone() - y.clone())
            .collect();

        Self {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            data,
        }
    }
}
