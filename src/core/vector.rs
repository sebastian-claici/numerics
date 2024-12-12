use std::default::Default;
use std::ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign};

#[derive(Debug)]
pub struct Vector<T> {
    pub n: usize,
    pub(crate) data: Vec<T>,
}

impl<T> Vector<T>
where
    T: Copy + Clone + Default,
{
    pub fn new(n: usize) -> Self {
        let data = vec![Default::default(); n];
        Self { n, data }
    }

    pub fn from_gen(n: usize, gen: fn(usize) -> T) -> Self {
        let data = (0..n).map(gen).collect();

        Self { n, data }
    }

    pub fn from_vec(data: &Vec<T>) -> Self {
        Self {
            n: data.len(),
            data: data.clone(),
        }
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.n {
            panic!(
                "Index {:?} out of bounds for vector of size {:?}",
                index, self.n
            );
        }

        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.n {
            panic!(
                "Index {:?} out of bounds for vector of size {:?}",
                index, self.n
            );
        }

        &mut self.data[index]
    }
}

impl<T> Add for &Vector<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Vector<T>;

    fn add(self, other: Self) -> Self::Output {
        if self.n != other.n {
            panic!(
                "Cannot add vectors of different shapes: {} and {}",
                self.n, other.n
            );
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.clone() + y.clone())
            .collect();

        Self::Output { n: self.n, data }
    }
}

impl<T> AddAssign<&Vector<T>> for Vector<T>
where
    T: AddAssign + Clone,
{
    fn add_assign(&mut self, other: &Self) {
        if self.n != other.n {
            panic!(
                "Cannot add vectors of different shapes: {} and {}",
                self.n, other.n
            );
        }

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, y)| *x += y.clone());
    }
}

impl<T> Sub for &Vector<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Vector<T>;

    fn sub(self, other: Self) -> Self::Output {
        if self.n != other.n {
            panic!(
                "Cannot add vectors of different shapes: {} and {}",
                self.n, other.n
            );
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x.clone() - y.clone())
            .collect();

        Self::Output { n: self.n, data }
    }
}

impl<T> SubAssign<&Vector<T>> for Vector<T>
where
    T: SubAssign + Clone,
{
    fn sub_assign(&mut self, other: &Self) {
        if self.n != other.n {
            panic!(
                "Cannot add vectors of different shapes: {} and {}",
                self.n, other.n
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

    // TODO
}
