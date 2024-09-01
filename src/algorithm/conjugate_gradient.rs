pub mod GradientMethod {

    use ndarray::{ ArrayD, Array1, ScalarOperand };
    use std::option::Option;
    use crate::algorithm::set::FirstOrderInitializeSet;
    use crate::algorithm::traits::{ FirstOrderNumericAlgorithm };
    use crate::algorithm::line_search::{ LineSearchTrait, DefaultLineSearch, LineSearchEnum };
    use num_traits::Float;

    pub struct ConjugateGradient<T> where T: Float + ScalarOperand {
        pub initial_setter: FirstOrderInitializeSet<T>,
        pub tolerance: T,
        pub max_iter: usize,
        pub line_search: Option<LineSearchEnum<T>>,
    }

    impl<T> FirstOrderNumericAlgorithm<T> for ConjugateGradient<T>
    where T: Float + ScalarOperand + std::fmt::Debug {
        fn new(initial_setter: FirstOrderInitializeSet<T>, tolerance: T, max_iter: usize) -> Self {
            Self {
                initial_setter,
                tolerance,
                max_iter,
                line_search: Some(LineSearchEnum::<T>::DefaultLineSearch(DefaultLineSearch::new(tolerance, max_iter, None))),
            }
        }

        fn initialize(&self) -> (Array1<T>, Array1<T>, Array1<T>) {
            let initial_point = self.initial_setter.get_initial_point().clone();
            let initial_gradient = self.initial_setter.apply_target_gradient(&initial_point);
            let initial_direction = initial_gradient.clone();
            (initial_point, initial_gradient, initial_direction)
        }

        fn set_line_search(&mut self, line_search: LineSearchEnum<T>) -> () {
            self.line_search = Some(line_search);
        }

        fn optimize(&self) -> Option<Array1<T>> {
            let (mut current_point, mut current_gradient, mut current_direction) = self.initialize();
            let mut prev_gradient = current_gradient.clone();
            let mut prev_direction = current_direction.clone();
            let mut beta: T;
            let mut iter = 0;
            let mut step_size: T;
            let search_algorithm = self.line_search.as_ref().unwrap();

            while !self._has_converged(&current_gradient) && iter < self.max_iter {
                step_size = search_algorithm.search(
                    &current_point,
                    &current_direction,
                    self.initial_setter.get_target_function(),
                    self.initial_setter.get_target_gradient()
                ).unwrap();
                println!("Step size: {:?}", step_size);

                current_point = current_point + &current_direction * step_size;
                println!("Current point: {:?}", current_point);

                current_gradient = self.initial_setter.apply_target_gradient(&current_point);
                beta = current_gradient.dot(&current_gradient) / prev_gradient.dot(&prev_gradient);
                current_direction = &current_gradient + prev_direction * beta;

                prev_gradient = current_gradient.clone();
                prev_direction = current_direction.clone();
                iter += 1;
            }

            if self._has_converged(&current_gradient) {
                Some(current_point)
            } else {
                println!("Failed to converge");
                Some(current_point)
            }
        }
    }

    impl<T> ConjugateGradient<T>
    where T: Float + ScalarOperand {
        fn _next_direction(&self, gradient: &Array1<T>, previous_direction: &Array1<T>, prev_gradient: &Array1<T>) -> Array1<T> {
            let beta = gradient.dot(gradient) / prev_gradient.dot(prev_gradient);
            gradient + previous_direction * beta
        }

        fn _has_converged(&self, gradient: &Array1<T>) -> bool {
            gradient.dot(gradient).sqrt() < self.tolerance
        }
    }
}
