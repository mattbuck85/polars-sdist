mod distributions;
mod utils;

#[cfg(target_os = "linux")]
use pyo3_polars::PolarsAllocator;

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_polars::PySeries;

use crate::distributions::sampling::sample_vec;

/// Direct sampling function bypassing polars plugin dispatch overhead.
#[pyfunction]
#[pyo3(signature = (dist, n, param1, param2=None, param3=None, seed=None))]
fn sample_direct(
    dist: &str,
    n: usize,
    param1: f64,
    param2: Option<f64>,
    param3: Option<f64>,
    seed: Option<u64>,
) -> PyResult<PySeries> {
    let values = sample_vec(dist, n, param1, param2, param3, seed)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let ca = Float64Chunked::from_vec("sample".into(), values);
    Ok(PySeries(ca.into_series()))
}

#[pymodule]
fn _polars_sdist(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_direct, m)?)?;
    Ok(())
}
