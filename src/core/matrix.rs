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

impl<T> Matrix<T>
where
    T: PartialEq,
{
    pub fn is_symmetric(&self) -> bool {
        if self.n_rows != self.n_cols {
            return false;
        }
        for i in 0..self.n_rows {
            for j in (i + 1)..self.n_rows {
                if self[(i, j)] != self[(j, i)] {
                    return false;
                }
            }
        }

        true
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

macro_rules! matrix_zeros {
    ($type:ty) => {
        impl Matrix<$type> {
            pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
                let data = (0..n_cols * n_rows).map(|_| 0 as $type).collect();

                Self {
                    n_rows,
                    n_cols,
                    data,
                }
            }
        }
    };
}

macro_rules! matrix_ones {
    ($type:ty) => {
        impl Matrix<$type> {
            pub fn ones(n_rows: usize, n_cols: usize) -> Self {
                let data = (0..n_cols * n_rows).map(|_| 1 as $type).collect();

                Self {
                    n_rows,
                    n_cols,
                    data,
                }
            }
        }
    };
}

macro_rules! matrix_eye {
    ($type:ty) => {
        impl Matrix<$type> {
            pub fn eye(n: usize) -> Self {
                let mut data = Vec::with_capacity(n * n);
                (0..n * n).map(|i| (i / n, i % n)).for_each(|(i, j)| {
                    data.push(if i == j { 1 as $type } else { 0 as $type });
                });

                Self {
                    n_rows: n,
                    n_cols: n,
                    data,
                }
            }
        }
    };
}

matrix_zeros!(f32);
matrix_zeros!(f64);
matrix_zeros!(i32);
matrix_zeros!(i64);

matrix_ones!(f32);
matrix_ones!(f64);
matrix_ones!(i32);
matrix_ones!(i64);

matrix_eye!(f32);
matrix_eye!(f64);
matrix_eye!(i32);
matrix_eye!(i64);

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

    #[test]
    fn test_is_symmetric() {
        let mut a = Matrix::new(2, 2);
        a[(0, 0)] = 1;
        a[(0, 1)] = 2;
        a[(1, 0)] = 3;
        a[(1, 1)] = 4;
        assert!(!a.is_symmetric());

        a[(0, 0)] = 1;
        a[(0, 1)] = 2;
        a[(1, 0)] = 2;
        a[(1, 1)] = 4;
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_zeros() {
        let z_i32 = Matrix::<i32>::zeros(3, 3);
        let i_f32 = Matrix::<i64>::eye(3);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(z_i32[(i, j)], 0);
            }
        }
    }

    #[test]
    fn test_ones() {
        let o_i64 = Matrix::<i64>::ones(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(o_i64[(i, j)], 1);
            }
        }
    }

    #[test]
    fn test_eye() {
        let i_f64 = Matrix::<f64>::eye(3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(i_f64[(i, j)], 1.0);
                } else {
                    assert_eq!(i_f64[(i, j)], 0.0);
                }
            }
        }
    }
}
