use ndarray::{ArrayD, IxDyn, Dim};
mod algorithm;
use algorithm::{ConjugateGradient, set::initialize_setter::InitializeSetter::FirstOrderInitializeSet, FirstOrderNumericAlgorithm};
use algorithm::{LineSearchEnum, ArmijoLineSearch};
use crate::algorithm::line_search::search_traits::LineSearchTrait;
fn main(){
    let initialize_setter = FirstOrderInitializeSet::<f32>::new(
        ndarray::arr1(&[10.0, -10.0]),
        1e-2,
        |array: &ndarray::Array1<f32>| -> f32 {
            array[0] * array[0] + array[1] * array[1] + 10.0 - 2.0 * array[0] - 3.0 * array[1]
        },
        |array: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
            -ndarray::arr1(&[2.0 * array[0] - 2.0, 2.0 * array[1] - 3.0])
        }
    );
    let mut cg = ConjugateGradient::<f32>::new(initialize_setter, 1e-6, 1000);
    cg.set_line_search(LineSearchEnum::ArmijoLineSearch(ArmijoLineSearch::<f32>::new(1e-2, 1000, None)));
    let result = cg.optimize();
    match result {
        Some(array) => {
            println!("Optimized point: {:?}", array);
        },
        None => {
            println!("Optimization failed");
        }
    }
}
