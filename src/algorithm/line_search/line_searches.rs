
pub mod LineSearches {
    use ndarray::{ Array1, ScalarOperand };
    use std::option::Option;
    use num_traits::Float;
    use crate::algorithm::line_search::LineSearchTrait;

    pub enum LineSearchEnum<T> where T: Float + ScalarOperand {
        ArmijoLineSearch(ArmijoLineSearch<T>),
        DefaultLineSearch(DefaultLineSearch<T>),
    }

    impl<T> LineSearchEnum<T>
    where T: Float + ScalarOperand {
        pub fn search(&self, point: &Array1<T>, direction: & Array1<T>, target_function: fn(&Array1<T>) -> T, target_gradient: fn(&Array1<T>) -> Array1<T>) -> Option<T> {
            match self {
                LineSearchEnum::ArmijoLineSearch(armijo) => armijo.search(point, direction, target_function, target_gradient),
                LineSearchEnum::DefaultLineSearch(default) => default.search(point, direction, target_function, target_gradient),
            }
        }
    }

    pub struct ArmijoLineSearch<T>
    where T: Float + ScalarOperand {
        pub tolerance: T,
        pub max_iter: usize,
    }

    impl<T> LineSearchTrait<T> for ArmijoLineSearch<T>
    where T: Float + ScalarOperand {
        fn new(tolerance: T, max_iter: usize, args: Option<&[T]>) -> Self {
            Self {
                tolerance,
                max_iter,
            }
        }

        fn search(&self, point: &Array1<T>, direction: & Array1<T>, target_function: fn(&Array1<T>) -> T, target_gradient: fn(&Array1<T>) -> Array1<T>) -> Option<T> {
            let mut step_size = T::from(0.5).unwrap();
            let mut iter = 0;
            let current_point = point;
            let current_gradient = target_gradient(point);
            let mut next_point;
            let alpha = T::from(0.5).unwrap();
            let c = T::from(0.5).unwrap();

            while iter < self.max_iter {
                next_point = current_point + direction * step_size;
                if target_function(&next_point) <= target_function(current_point) + c * step_size * current_gradient.dot(direction) {
                    return Some(step_size);
                } else {
                    step_size = step_size * alpha;
                }
                iter += 1;
            }

            None
        }
    }

    pub struct DefaultLineSearch<T> where T: Float + ScalarOperand {
        pub tolerance: T,
        pub max_iter: usize,
    }

    impl<T> LineSearchTrait<T> for DefaultLineSearch<T>
    where T: Float + ScalarOperand {
        fn new(tolerance: T, max_iter: usize, _args: Option<&[T]>) -> Self {
            Self {
                tolerance,
                max_iter,
            }
        }

        fn search(&self, _point: &Array1<T>, _direction: & Array1<T>, _target_function: fn(&Array1<T>) -> T, _target_gradient: fn(&Array1<T>) -> Array1<T>) -> Option<T> {
            let step_size = T::from(0.5).unwrap();
            return Some(step_size);
        }
    }
}
