use super::{line_search, set::FirstOrderInitializeSet};
use ndarray::{ Array1, ScalarOperand };
use num_traits::Float;
use line_search::LineSearchEnum;

pub trait FirstOrderNumericAlgorithm<T> where T: Float + ScalarOperand {
    fn optimize(&self) -> Option<Array1<T>>;
    fn new(initial_setter: FirstOrderInitializeSet<T>, tolerance: T, max_iter: usize) -> Self;
    fn initialize(&self) -> (Array1<T>, Array1<T>, Array1<T>);
    fn set_line_search(&mut self, line_search: LineSearchEnum<T>) -> ();
}

