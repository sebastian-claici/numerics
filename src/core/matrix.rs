use std::default::Default;
use std::ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign};

#[derive(Debug)]
pub struct Matrix<T> {
    pub n_rows: usize,
    pub n_cols: usize,
    pub(crate) data: Vec<T>,
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

    pub fn from_gen(n_rows: usize, n_cols: usize, gen: fn(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(n_rows * n_cols);
        (0..n_rows * n_cols)
            .map(|i| (i / n_cols, i % n_cols))
            .for_each(|(i, j)| {
                data.push(gen(i, j));
            });

        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.n_cols * self.n_rows);
        for j in 0..self.n_cols {
            for i in 0..self.n_rows {
                data.push(self.data[i * self.n_cols + j]);
            }
        }

        Self {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            data,
        }
    }
}

impl Matrix<f32> {
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        let data = (0..n_cols * n_rows).map(|_| 0.0).collect();

        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    pub fn ones(n_rows: usize, n_cols: usize) -> Self {
        let data = (0..n_cols * n_rows).map(|_| 1.0).collect();

        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    pub fn eye(n: usize) -> Self {
        let mut data = Vec::with_capacity(n * n);
        (0..n * n).map(|i| (i / n, i % n)).for_each(|(i, j)| {
            data.push(if i == j { 1.0 } else { 0.0 });
        });

        Self {
            n_rows: n,
            n_cols: n,
            data,
        }
    }
}

impl Matrix<f64> {
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        let data = (0..n_cols * n_rows).map(|_| 0.0).collect();

        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    pub fn ones(n_rows: usize, n_cols: usize) -> Self {
        let data = (0..n_cols * n_rows).map(|_| 1.0).collect();

        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    pub fn eye(n: usize) -> Self {
        let mut data = Vec::with_capacity(n * n);
        (0..n * n).map(|i| (i / n, i % n)).for_each(|(i, j)| {
            data.push(if i == j { 1.0 } else { 0.0 });
        });

        Self {
            n_rows: n,
            n_cols: n,
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

impl<T> Add for &Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Matrix<T>;

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

        Self::Output {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            data,
        }
    }
}

impl<T> AddAssign<&Matrix<T>> for Matrix<T>
where
    T: AddAssign + Clone,
{
    fn add_assign(&mut self, other: &Self) {
        if self.n_rows != other.n_rows || self.n_cols != other.n_cols {
            panic!(
                "Cannot add matrices of different shapes: ({}, {}) ({}, {})",
                self.n_rows, self.n_cols, other.n_rows, other.n_cols
            );
        }

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x += y.clone());
    }
}

impl<T> Sub for &Matrix<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Matrix<T>;

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

        Self::Output {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            data,
        }
    }
}

impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where
    T: SubAssign + Clone,
{
    fn sub_assign(&mut self, other: &Self) {
        if self.n_rows != other.n_rows || self.n_cols != other.n_cols {
            panic!(
                "Cannot add matrices of different shapes: ({}, {}) ({}, {})",
                self.n_rows, self.n_cols, other.n_rows, other.n_cols
            );
        }

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x -= y.clone());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gen() {
        let a = Matrix::from_gen(2, 2, |i, j| i + j);
        assert_eq!(a[(0, 0)], 0);
        assert_eq!(a[(0, 1)], 1);
        assert_eq!(a[(1, 0)], 1);
        assert_eq!(a[(1, 1)], 2);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_gen(2, 2, |i, j| (i + j) as i32);
        let b = Matrix::from_gen(2, 2, |i, j| (i as i32 - j as i32));
        let c = &a + &b;
        assert_eq!(c[(0, 0)], 0);
        assert_eq!(c[(0, 1)], 0);
        assert_eq!(c[(1, 0)], 2);
        assert_eq!(c[(1, 1)], 2);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_gen(2, 2, |i, j| (i + j) as i32);
        let b = Matrix::from_gen(2, 2, |i, j| (i as i32 - j as i32));
        let c = &a - &b;
        assert_eq!(c[(0, 0)], 0);
        assert_eq!(c[(0, 1)], 2);
        assert_eq!(c[(1, 0)], 0);
        assert_eq!(c[(1, 1)], 2);
    }

    #[test]
    fn test_transpose() {
        let mut a = Matrix::new(2, 2);
        a[(0, 0)] = 1;
        a[(0, 1)] = 2;
        a[(1, 0)] = 3;
        a[(1, 1)] = 4;

        let at = a.transpose();
        assert_eq!(at[(0, 0)], 1);
        assert_eq!(at[(0, 1)], 3);
        assert_eq!(at[(1, 0)], 2);
        assert_eq!(at[(1, 1)], 4);
    }
}
