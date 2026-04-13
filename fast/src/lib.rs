mod graph;
mod settings;
mod trainer;
mod units;

use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<graph::Node>()?;
    m.add_class::<graph::NodesSequence>()?;
    m.add_class::<graph::Tree>()?;
    m.add_class::<graph::FullyConnectedGraph>()?;
    m.add_class::<graph::UnconnectedGraphs>()?;
    m.add_class::<trainer::Trainer>()?;
    m.add_function(wrap_pyfunction!(units::utf8, m)?)?;
    m.add_function(wrap_pyfunction!(units::utf8_clusters, m)?)?;
    m.add_function(wrap_pyfunction!(units::characters, m)?)?;
    m.add_function(wrap_pyfunction!(units::register_script, m)?)?;
    m.add_function(wrap_pyfunction!(units::clear_handlers, m)?)?;
    m.add_function(wrap_pyfunction!(units::get_handlers_dict, m)?)?;
    m.add_function(wrap_pyfunction!(sync_settings, m)?)?;
    Ok(())
}

#[pyfunction]
fn sync_settings(max_merge_size: usize, only_minimal_merges: bool) {
    settings::MAX_MERGE_SIZE.store(max_merge_size, std::sync::atomic::Ordering::Relaxed);
    settings::ONLY_MINIMAL_MERGES.store(only_minimal_merges, std::sync::atomic::Ordering::Relaxed);
}
