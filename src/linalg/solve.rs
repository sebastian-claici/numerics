use std::ops::{Div, DivAssign, Mul, SubAssign};

use crate::core::matrix::Matrix;
use crate::core::vector::Vector;

pub trait Solve<T> {
    fn solve(&self, rhs: &Vector<T>) -> Vector<T>;
}

fn bsub<T>(lhs: &Matrix<T>, rhs: &Vector<T>) -> Vector<T>
where
    T: Copy + Clone + Default + Mul<Output = T> + Div<Output = T> + DivAssign + SubAssign,
{
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

impl<T> Solve<T> for Matrix<T> {
    fn solve(&self, rhs: &Vector<T>) -> Vector<T> {
        todo!()
    }
}
