mod distributions;
mod utils;

#[cfg(target_os = "linux")]
use pyo3_polars::PolarsAllocator;

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

use pyo3::types::PyModule;
use pyo3::{pymodule, Bound, PyResult};

#[pymodule]
fn _polars_sdist(_py: pyo3::Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
