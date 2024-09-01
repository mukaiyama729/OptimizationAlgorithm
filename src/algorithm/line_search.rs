pub mod line_searches;
pub mod search_traits;
pub use line_searches::LineSearches::{ ArmijoLineSearch, DefaultLineSearch, LineSearchEnum };
pub use search_traits::LineSearchTrait;
