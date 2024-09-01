use num_traits::Float;
use ndarray::{ Array1, ScalarOperand };

pub trait LineSearchTrait<T> {
    fn new(tolerance: T, max_iter: usize, args: Option<&[T]>) -> Self;
    fn search(&self, point: &Array1<T>, direction: &Array1<T>, target_function: fn(&Array1<T>) -> T, target_gradient: fn(&Array1<T>) -> Array1<T>) -> Option<T>;
}

