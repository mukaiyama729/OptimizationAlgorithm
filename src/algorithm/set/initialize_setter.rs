
pub mod InitializeSetter {
    use ndarray::{ Array1 };
    pub struct FirstOrderInitializeSet<T>  {
        pub initial_point: Array1<T>,
        pub converge_threshold: f64,
        pub target_gradient: fn(&Array1<T>) -> Array1<T>,
        pub target_function: fn(&Array1<T>) -> T,
    }

    impl<T> FirstOrderInitializeSet<T> {
        pub fn new(initial_point: Array1<T>, converge_threshold: f64, target_function: fn(array: &Array1<T>) -> T, target_gradient: fn(array: &Array1<T>) -> Array1<T>) -> Self {
            Self {
                initial_point,
                converge_threshold,
                target_function,
                target_gradient,
            }
        }

        pub fn get_initial_point(&self) -> &Array1<T> {
            &self.initial_point
        }

        pub fn apply_target_function(&self, point: &Array1<T>) -> T {
            (self.target_function)(point)
        }

        pub fn apply_target_gradient(&self, point: &Array1<T>) -> Array1<T> {
            (self.target_gradient)(point)
        }

        pub fn get_converge_threshold(&self) -> &f64 {
            &self.converge_threshold
        }

        pub fn get_target_function(&self) -> fn(&Array1<T>) -> T {
            self.target_function
        }

        pub fn get_target_gradient(&self) -> fn(&Array1<T>) -> Array1<T> {
            self.target_gradient
        }
    }
}
