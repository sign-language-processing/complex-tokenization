use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub static MAX_MERGE_SIZE: AtomicUsize = AtomicUsize::new(2);
pub static ONLY_MINIMAL_MERGES: AtomicBool = AtomicBool::new(true);

#[pyclass]
pub struct GraphSettings;

#[pymethods]
impl GraphSettings {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        match name {
            "MAX_MERGE_SIZE" => {
                let val = MAX_MERGE_SIZE.load(Ordering::Relaxed);
                Ok(val.into_pyobject(py)?.into_any().unbind())
            }
            "ONLY_MINIMAL_MERGES" => {
                let val = ONLY_MINIMAL_MERGES.load(Ordering::Relaxed);
                Ok(val.to_object(py))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                format!("GraphSettings has no attribute '{name}'"),
            )),
        }
    }

    fn __setattr__(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match name {
            "MAX_MERGE_SIZE" => {
                MAX_MERGE_SIZE.store(value.extract::<usize>()?, Ordering::Relaxed);
                Ok(())
            }
            "ONLY_MINIMAL_MERGES" => {
                ONLY_MINIMAL_MERGES.store(value.extract::<bool>()?, Ordering::Relaxed);
                Ok(())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                format!("GraphSettings has no attribute '{name}'"),
            )),
        }
    }
}
