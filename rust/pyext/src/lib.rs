use augment_core::{
    BorderMode, BufferExplanation, CompiledPipeline, Compiler, CopyExplanation, CoreError,
    DropoutSizeRange, ExecutionMode, ImageContractExplanation, Interpolation, MaskOutput,
    MaskTransformExplanation, PadPosition, PipelineExplanation, PipelineOutput, PipelineSpec,
    PolicyExplanation, TargetBuffer, TargetInput, TargetOutput, TargetRequirements, TargetSpec,
    TransformExplanation, TransformSpec, Workspace, REGISTERED_TRANSFORM_NAMES,
};
use augment_io::{
    CodecError, ColorModel, DecodeMode, DecodeOptions, DecodedImage, EncodeOptions, ImageFormat,
    ImageView, OwnedImage, PixelData, PixelDataRef,
};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyByteArray, PyByteArrayMethods, PyBytes, PyBytesMethods, PyDict, PyList, PyMemoryView,
    PyTuple,
};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const MAX_CACHED_WORKSPACES: usize = 8;
const MAX_RETAINED_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const EXPLANATION_SCHEMA_VERSION: u8 = 4;

#[pyfunction]
fn registered_transform_names() -> Vec<&'static str> {
    REGISTERED_TRANSFORM_NAMES.to_vec()
}

#[pyclass(name = "Pipeline")]
struct PyPipeline {
    core: CompiledPipeline,
    targets: Vec<TargetRoute>,
    seed: u64,
    next_key: Mutex<u64>,
    workspaces: Mutex<Vec<Workspace>>,
}

#[derive(Clone, Copy, Debug)]
enum SourceConfig {
    Array,
    Encoded {
        max_pixels: Option<usize>,
        max_encoded_bytes: Option<usize>,
    },
    Path {
        max_pixels: Option<usize>,
        max_encoded_bytes: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug)]
enum SinkConfig {
    ReturnArray,
    ReturnTensor,
    Encoded {
        format: ImageFormat,
        options: EncodeOptions,
    },
    Path {
        format: Option<ImageFormat>,
        quality: Option<u8>,
        compression: Option<u8>,
    },
}

#[derive(Clone, Debug)]
struct TargetRoute {
    role: TargetSpec,
    source: SourceConfig,
    outputs: Vec<OutputRoute>,
    name: Option<String>,
}

#[derive(Clone, Debug)]
struct OutputRoute {
    name: Option<String>,
    sink: SinkConfig,
}

enum NativeOutput {
    ReturnImage(PipelineOutput),
    ReturnMask(MaskOutput),
    Encoded(Vec<u8>),
    Written,
}

enum PreparedSource {
    Array {
        storage: TargetStorage,
        height: usize,
        width: usize,
    },
    Encoded(Vec<u8>),
    Path(PathBuf),
}

enum PreparedSink {
    Return,
    ReturnTensor,
    Encoded(EncodeOptions),
    Write {
        destination: PathBuf,
        options: EncodeOptions,
    },
}

struct PreparedTarget {
    label: String,
    source: PreparedSource,
    sinks: Vec<PreparedSink>,
}

enum TargetStorage {
    Borrowed(Py<PyAny>),
    Owned(Vec<u8>),
}

struct AcquiredTarget {
    label: String,
    storage: TargetStorage,
    height: usize,
    width: usize,
}

enum PreparedDelivery {
    ReturnImage(PipelineOutput),
    ReturnMask(MaskOutput),
    Encoded(Vec<u8>),
    Write {
        destination: PathBuf,
        encoded: Vec<u8>,
    },
}

enum PipelineRunError {
    Core(CoreError),
    Codec(CodecError),
    Workspace,
    DecodedContract(&'static str),
    InvalidTarget(String),
    Read {
        path: PathBuf,
        error: std::io::Error,
    },
    Write {
        path: PathBuf,
        error: std::io::Error,
    },
    EncodedLimit {
        actual: usize,
        limit: usize,
    },
    SinkContract,
    Target {
        label: String,
        error: Box<PipelineRunError>,
    },
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (specs, seed, mode = "reference", targets = None))]
    fn new(
        specs: &Bound<'_, PyAny>,
        seed: u64,
        mode: &str,
        targets: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let targets = parse_target_routes(targets)?;
        let target_specs = targets
            .iter()
            .map(|target| (target.role, target_requirements(target)))
            .collect();
        let core = Compiler::new(parse_mode(mode)?)
            .compile(PipelineSpec::with_target_requirements(
                parse_specs(specs)?,
                target_specs,
            ))
            .map_err(map_core_error)?;
        let explanation = core.explain();
        if targets.iter().any(|target| {
            target.role == TargetSpec::Image
                && target.outputs.iter().any(|output| {
                    matches!(
                        output.sink,
                        SinkConfig::Encoded { .. } | SinkConfig::Path { .. }
                    )
                })
        }) && explanation.output_dtype != "uint8"
        {
            return Err(PyValueError::new_err(
                "encoded pipeline output requires an always-HWC RGB uint8 result",
            ));
        }
        Ok(Self {
            core,
            targets,
            seed,
            next_key: Mutex::new(0),
            workspaces: Mutex::new(Vec::new()),
        })
    }

    #[pyo3(signature = (bindings, key = None))]
    fn apply_targets(
        &self,
        py: Python<'_>,
        bindings: &Bound<'_, PyAny>,
        key: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let (prepared, has_array) = self.preflight(py, bindings)?;
        let output = if has_array {
            self.execute(Some(py), prepared, key)
        } else {
            py.allow_threads(move || self.execute(None, prepared, key))
        }
        .map_err(map_pipeline_run_error)?;
        native_outputs_to_python(py, output)
    }

    fn explain<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        explanation_to_python(py, self.core.explain(), &self.targets)
    }
}

impl PyPipeline {
    fn take_workspace(&self) -> Result<Workspace, PipelineRunError> {
        self.workspaces
            .lock()
            .map_err(|_| PipelineRunError::Workspace)
            .map(|mut pool| pool.pop().unwrap_or_default())
    }

    fn return_workspace(&self, workspace: Workspace) -> Result<(), PipelineRunError> {
        if workspace.retained_bytes() > MAX_RETAINED_WORKSPACE_BYTES {
            return Ok(());
        }
        let mut pool = self
            .workspaces
            .lock()
            .map_err(|_| PipelineRunError::Workspace)?;
        if pool.len() < MAX_CACHED_WORKSPACES {
            pool.push(workspace);
        }
        Ok(())
    }

    fn preflight(
        &self,
        py: Python<'_>,
        bindings: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<PreparedTarget>, bool)> {
        let bindings = bindings.downcast::<PyTuple>()?;
        if bindings.len() != self.targets.len() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected {} target bindings, received {}",
                self.targets.len(),
                bindings.len()
            )));
        }
        let mut destinations = HashSet::new();
        let mut prepared = Vec::with_capacity(bindings.len());
        let mut has_array = false;
        for (index, (binding, route)) in bindings.iter().zip(&self.targets).enumerate() {
            let bound = binding.hasattr("_source")?;
            let source = if bound {
                binding.getattr("_source")?
            } else {
                binding.clone()
            };
            let write_destinations = if bound {
                let values = binding.getattr("_write_bindings")?;
                let values = values.downcast::<PyTuple>()?;
                values
                    .iter()
                    .map(|value| value.getattr("_destination")?.extract::<PathBuf>())
                    .collect::<PyResult<Vec<_>>>()?
            } else {
                Vec::new()
            };
            let label = target_label(index, route);
            let source = match route.source {
                SourceConfig::Array => {
                    has_array = true;
                    array_source(&source, route.role, &label)?
                }
                SourceConfig::Encoded {
                    max_encoded_bytes, ..
                } => PreparedSource::Encoded(snapshot_encoded(
                    py,
                    &source,
                    max_encoded_bytes,
                    &label,
                )?),
                SourceConfig::Path { .. } => {
                    PreparedSource::Path(source.extract::<PathBuf>().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(format!(
                            "{label} source must be a local path"
                        ))
                    })?)
                }
            };
            let sinks = prepare_sinks(route, write_destinations, &label)?;
            for sink in &sinks {
                if let PreparedSink::Write { destination, .. } = sink {
                    validate_write_destination(destination, &label)?;
                    if !destinations.insert(destination.clone()) {
                        return Err(PyValueError::new_err(format!(
                            "duplicate target destination '{}'",
                            destination.display()
                        )));
                    }
                }
            }
            prepared.push(PreparedTarget {
                label,
                source,
                sinks,
            });
        }
        Ok((prepared, has_array))
    }

    fn execute(
        &self,
        py: Option<Python<'_>>,
        prepared: Vec<PreparedTarget>,
        key: Option<u64>,
    ) -> Result<Vec<Vec<NativeOutput>>, PipelineRunError> {
        let mut acquired = Vec::with_capacity(prepared.len());
        let mut sinks = Vec::with_capacity(prepared.len());
        for (target, route) in prepared.into_iter().zip(&self.targets) {
            acquired.push(acquire_target(target.source, route, target.label)?);
            sinks.push(target.sinks);
        }
        let (height, width) = (acquired[0].height, acquired[0].width);
        for target in acquired.iter().skip(1) {
            if (target.height, target.width) != (height, width) {
                return Err(PipelineRunError::InvalidTarget(format!(
                    "{} dimensions do not match the initial coordinate frame",
                    target.label
                )));
            }
        }
        if acquired
            .iter()
            .any(|target| matches!(target.storage, TargetStorage::Borrowed(_)))
        {
            return self.run_borrowed_targets(
                py.ok_or_else(|| {
                    PipelineRunError::InvalidTarget(
                        "array-backed target execution requires the GIL".into(),
                    )
                })?,
                acquired,
                sinks,
                key,
            );
        }
        let inputs = acquired
            .into_iter()
            .zip(&self.targets)
            .map(|(target, route)| TargetInput {
                role: route.role,
                data: match target.storage {
                    TargetStorage::Owned(data) => TargetBuffer::Owned(data),
                    TargetStorage::Borrowed(_) => unreachable!("borrowed targets handled above"),
                },
                height: target.height,
                width: target.width,
            })
            .collect();
        self.run_inputs(inputs, sinks, key)
    }

    fn run_borrowed_targets(
        &self,
        py: Python<'_>,
        acquired: Vec<AcquiredTarget>,
        sinks: Vec<Vec<PreparedSink>>,
        key: Option<u64>,
    ) -> Result<Vec<Vec<NativeOutput>>, PipelineRunError> {
        let arrays = acquired
            .iter()
            .filter_map(|target| match &target.storage {
                TargetStorage::Borrowed(array) => Some((array.clone_ref(py), target.label.clone())),
                TargetStorage::Owned(_) => None,
            })
            .collect::<Vec<_>>();
        let guards = arrays
            .iter()
            .map(|(array, label)| {
                array
                    .bind(py)
                    .extract::<PyReadonlyArrayDyn<'_, u8>>()
                    .map_err(|_| {
                        PipelineRunError::InvalidTarget(format!(
                            "{label} validated NumPy source is no longer available"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut guard_index = 0;
        let mut inputs = Vec::with_capacity(acquired.len());
        for (target, route) in acquired.into_iter().zip(&self.targets) {
            let data = match target.storage {
                TargetStorage::Borrowed(_) => {
                    let guard = &guards[guard_index];
                    guard_index += 1;
                    TargetBuffer::Borrowed(guard.as_slice().map_err(|_| {
                        PipelineRunError::InvalidTarget(format!(
                            "{} validated NumPy source is not contiguous",
                            target.label
                        ))
                    })?)
                }
                TargetStorage::Owned(data) => TargetBuffer::Owned(data),
            };
            inputs.push(TargetInput {
                role: route.role,
                data,
                height: target.height,
                width: target.width,
            });
        }
        self.run_inputs(inputs, sinks, key)
    }

    fn run_inputs(
        &self,
        inputs: Vec<TargetInput<'_>>,
        sinks: Vec<Vec<PreparedSink>>,
        key: Option<u64>,
    ) -> Result<Vec<Vec<NativeOutput>>, PipelineRunError> {
        let mut implicit_key = if key.is_none() {
            Some(
                self.next_key
                    .lock()
                    .map_err(|_| PipelineRunError::Workspace)?,
            )
        } else {
            None
        };
        let key = key.unwrap_or_else(|| **implicit_key.as_ref().expect("implicit key is locked"));
        let mut workspace = self.take_workspace()?;
        let outputs = self
            .core
            .apply_targets(inputs, self.seed, key, &mut workspace)
            .map_err(PipelineRunError::Core);
        self.return_workspace(workspace)?;
        let result = self.deliver(outputs?, sinks)?;
        if let Some(next_key) = implicit_key.as_mut() {
            **next_key = next_key.wrapping_add(1);
        }
        Ok(result)
    }

    fn deliver(
        &self,
        outputs: Vec<TargetOutput>,
        sinks: Vec<Vec<PreparedSink>>,
    ) -> Result<Vec<Vec<NativeOutput>>, PipelineRunError> {
        let mut deliveries = Vec::with_capacity(outputs.len());
        for (output, target_sinks) in outputs.into_iter().zip(sinks) {
            let encoded = target_sinks
                .iter()
                .map(|sink| match sink {
                    PreparedSink::Encoded(options) | PreparedSink::Write { options, .. } => {
                        encode_target_output(&output, *options).map(Some)
                    }
                    PreparedSink::Return | PreparedSink::ReturnTensor => Ok(None),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut hwc_returns = target_sinks
                .iter()
                .filter(|sink| matches!(sink, PreparedSink::Return))
                .count();
            let mut chw_returns = target_sinks
                .iter()
                .filter(|sink| matches!(sink, PreparedSink::ReturnTensor))
                .count();
            let (mut image, mut mask) = match output {
                TargetOutput::Image(image) => (Some(image), None),
                TargetOutput::Mask(mask) => {
                    hwc_returns += chw_returns;
                    chw_returns = 0;
                    (None, Some(mask))
                }
            };
            let mut target_deliveries = Vec::with_capacity(target_sinks.len());
            for (sink, encoded) in target_sinks.into_iter().zip(encoded) {
                match sink {
                    PreparedSink::Return => {
                        if let Some(image) = image.as_mut() {
                            target_deliveries.push(PreparedDelivery::ReturnImage(
                                take_or_clone_image(&mut image.hwc, &mut hwc_returns)?,
                            ));
                        } else {
                            target_deliveries.push(PreparedDelivery::ReturnMask(
                                take_or_clone_mask(&mut mask, &mut hwc_returns)?,
                            ));
                        }
                    }
                    PreparedSink::ReturnTensor => {
                        if let Some(image) = image.as_mut() {
                            target_deliveries.push(PreparedDelivery::ReturnImage(
                                take_or_clone_image(&mut image.chw, &mut chw_returns)?,
                            ));
                        } else {
                            target_deliveries.push(PreparedDelivery::ReturnMask(
                                take_or_clone_mask(&mut mask, &mut hwc_returns)?,
                            ));
                        }
                    }
                    PreparedSink::Encoded(_) => target_deliveries.push(PreparedDelivery::Encoded(
                        encoded.ok_or(PipelineRunError::SinkContract)?,
                    )),
                    PreparedSink::Write { destination, .. } => {
                        target_deliveries.push(PreparedDelivery::Write {
                            destination,
                            encoded: encoded.ok_or(PipelineRunError::SinkContract)?,
                        });
                    }
                }
            }
            deliveries.push(target_deliveries);
        }
        let mut result = Vec::with_capacity(deliveries.len());
        for target_deliveries in deliveries {
            let mut target_result = Vec::with_capacity(target_deliveries.len());
            for delivery in target_deliveries {
                match delivery {
                    PreparedDelivery::ReturnImage(output) => {
                        target_result.push(NativeOutput::ReturnImage(output));
                    }
                    PreparedDelivery::ReturnMask(output) => {
                        target_result.push(NativeOutput::ReturnMask(output));
                    }
                    PreparedDelivery::Encoded(encoded) => {
                        target_result.push(NativeOutput::Encoded(encoded));
                    }
                    PreparedDelivery::Write {
                        destination,
                        encoded,
                    } => {
                        write_atomic(&destination, &encoded).map_err(|error| {
                            PipelineRunError::Write {
                                path: destination,
                                error,
                            }
                        })?;
                        target_result.push(NativeOutput::Written);
                    }
                }
            }
            result.push(target_result);
        }
        Ok(result)
    }
}

fn take_or_clone_image(
    output: &mut Option<PipelineOutput>,
    remaining: &mut usize,
) -> Result<PipelineOutput, PipelineRunError> {
    *remaining = remaining
        .checked_sub(1)
        .ok_or(PipelineRunError::SinkContract)?;
    if *remaining == 0 {
        output.take().ok_or(PipelineRunError::SinkContract)
    } else {
        output.clone().ok_or(PipelineRunError::SinkContract)
    }
}

fn take_or_clone_mask(
    output: &mut Option<MaskOutput>,
    remaining: &mut usize,
) -> Result<MaskOutput, PipelineRunError> {
    *remaining = remaining
        .checked_sub(1)
        .ok_or(PipelineRunError::SinkContract)?;
    if *remaining == 0 {
        output.take().ok_or(PipelineRunError::SinkContract)
    } else {
        output.clone().ok_or(PipelineRunError::SinkContract)
    }
}

fn validate_write_destination(path: &Path, label: &str) -> PyResult<()> {
    if path.file_name().is_none() {
        return Err(PyValueError::new_err(format!(
            "{label} destination must have a file name"
        )));
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(PyValueError::new_err(format!(
            "{label} destination parent '{}' is not a directory",
            parent.display()
        )));
    }
    if path.is_dir() {
        return Err(PyValueError::new_err(format!(
            "{label} destination '{}' is a directory",
            path.display()
        )));
    }
    Ok(())
}

fn parse_source_config(value: &Bound<'_, PyDict>) -> PyResult<SourceConfig> {
    let source: String = required(value, "carrier")?.extract()?;
    let max_pixels = optional(value, "max_pixels")?.map_or(Ok(None), |value| value.extract())?;
    let max_encoded_bytes =
        optional(value, "max_encoded_bytes")?.map_or(Ok(None), |value| value.extract())?;
    if max_pixels == Some(0) {
        return Err(PyValueError::new_err("max_pixels must be positive or None"));
    }
    if max_encoded_bytes == Some(0) {
        return Err(PyValueError::new_err(
            "max_encoded_bytes must be positive or None",
        ));
    }
    match source.as_str() {
        "array" if max_pixels.is_none() && max_encoded_bytes.is_none() => Ok(SourceConfig::Array),
        "encoded" => Ok(SourceConfig::Encoded {
            max_pixels,
            max_encoded_bytes,
        }),
        "path" => Ok(SourceConfig::Path {
            max_pixels,
            max_encoded_bytes,
        }),
        "array" => Err(PyValueError::new_err("Array does not accept decode limits")),
        _ => Err(PyValueError::new_err(
            "source must be 'array', 'encoded', or 'path'",
        )),
    }
}

fn parse_output_route(value: &Bound<'_, PyDict>, role: TargetSpec) -> PyResult<OutputRoute> {
    let sink: String = required(value, "type")?.extract()?;
    let format: Option<String> =
        optional(value, "format")?.map_or(Ok(None), |value| value.extract())?;
    let quality: Option<u8> =
        optional(value, "quality")?.map_or(Ok(None), |value| value.extract())?;
    let compression: Option<u8> =
        optional(value, "compression")?.map_or(Ok(None), |value| value.extract())?;
    let name = optional(value, "name")?.map_or(Ok(None), |value| value.extract())?;
    let sink = if sink == "return_array" || sink == "return_tensor" {
        if format.is_some() || quality.is_some() || compression.is_some() {
            return Err(PyValueError::new_err(
                "return outputs do not accept codec options",
            ));
        }
        if sink == "return_array" {
            SinkConfig::ReturnArray
        } else {
            SinkConfig::ReturnTensor
        }
    } else if sink == "encode" {
        let format = format
            .as_deref()
            .ok_or_else(|| PyValueError::new_err("Encode requires a format"))?;
        let image_format = parse_image_format(format)?;
        if matches!(role, TargetSpec::Mask { .. }) && image_format != ImageFormat::Png {
            return Err(PyValueError::new_err("Mask Encode output must use PNG"));
        }
        SinkConfig::Encoded {
            format: image_format,
            options: encode_options(format, quality, compression)?,
        }
    } else if sink == "write" {
        let image_format = format.as_deref().map(parse_image_format).transpose()?;
        if matches!(role, TargetSpec::Mask { .. })
            && image_format.is_some_and(|format| format != ImageFormat::Png)
        {
            return Err(PyValueError::new_err("Mask Write output must use PNG"));
        }
        SinkConfig::Path {
            format: image_format,
            quality,
            compression,
        }
    } else {
        return Err(PyValueError::new_err(
            "output type must be 'return_array', 'return_tensor', 'encode', or 'write'",
        ));
    };
    Ok(OutputRoute { name, sink })
}

fn parse_target_routes(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<TargetRoute>> {
    let value = value.ok_or_else(|| PyValueError::new_err("targets are required"))?;
    let targets = value.downcast::<PyList>()?;
    if targets.is_empty() {
        return Err(PyValueError::new_err(
            "a pipeline requires at least one target",
        ));
    }
    targets
        .iter()
        .map(|item| {
            let item = item.downcast::<PyDict>()?;
            let role_name: String = required(item, "role")?.extract()?;
            let role = match role_name.as_str() {
                "image" => TargetSpec::Image,
                "mask" => {
                    let fill = required(item, "fill")?.extract::<u8>().map_err(|_| {
                        PyValueError::new_err("Mask fill must be an integer in [0, 255]")
                    })?;
                    TargetSpec::Mask { fill }
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "target role must be 'image' or 'mask'",
                    ));
                }
            };
            let outputs = required(item, "outputs")?.downcast_into::<PyList>()?;
            if outputs.is_empty() {
                return Err(PyValueError::new_err(
                    "a target requires at least one output",
                ));
            }
            let outputs = outputs
                .iter()
                .map(|output| parse_output_route(output.downcast::<PyDict>()?, role))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(TargetRoute {
                role,
                source: parse_source_config(item)?,
                outputs,
                name: optional(item, "name")?.map_or(Ok(None), |value| value.extract())?,
            })
        })
        .collect()
}

fn target_label(index: usize, route: &TargetRoute) -> String {
    match &route.name {
        Some(name) => format!("target {index} ({name:?})"),
        None => format!("target {index}"),
    }
}

fn target_requirements(route: &TargetRoute) -> TargetRequirements {
    if matches!(route.role, TargetSpec::Mask { .. }) {
        return TargetRequirements::HW;
    }
    let hwc = route.outputs.iter().any(|output| {
        matches!(
            output.sink,
            SinkConfig::ReturnArray | SinkConfig::Encoded { .. } | SinkConfig::Path { .. }
        )
    });
    let chw = route
        .outputs
        .iter()
        .any(|output| matches!(output.sink, SinkConfig::ReturnTensor));
    TargetRequirements { hwc, chw }
}

fn array_source(
    value: &Bound<'_, PyAny>,
    role: TargetSpec,
    label: &str,
) -> PyResult<PreparedSource> {
    let array = value.extract::<PyReadonlyArrayDyn<'_, u8>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!("{label} must be a NumPy uint8 array"))
    })?;
    let (height, width) = match (role, array.shape()) {
        (TargetSpec::Image, [height, width, 3]) if *height > 0 && *width > 0 => (*height, *width),
        (TargetSpec::Mask { .. }, [height, width]) if *height > 0 && *width > 0 => {
            (*height, *width)
        }
        (TargetSpec::Image, _) => {
            return Err(PyValueError::new_err(format!(
                "{label} must be a non-empty HWC RGB array"
            )));
        }
        (TargetSpec::Mask { .. }, _) => {
            return Err(PyValueError::new_err(format!(
                "{label} must be a non-empty HW array"
            )));
        }
    };
    let storage = if array.is_c_contiguous() {
        TargetStorage::Borrowed(value.clone().unbind())
    } else {
        TargetStorage::Owned(array.as_array().iter().copied().collect())
    };
    Ok(PreparedSource::Array {
        storage,
        height,
        width,
    })
}

fn prepare_sinks(
    route: &TargetRoute,
    destinations: Vec<PathBuf>,
    label: &str,
) -> PyResult<Vec<PreparedSink>> {
    let expected = route
        .outputs
        .iter()
        .filter(|output| matches!(output.sink, SinkConfig::Path { .. }))
        .count();
    if destinations.len() != expected {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected {expected} Write bindings for {label}, received {}",
            destinations.len()
        )));
    }
    let mut destinations = destinations.into_iter();
    route
        .outputs
        .iter()
        .map(|output| {
            let destination = matches!(output.sink, SinkConfig::Path { .. })
                .then(|| destinations.next())
                .flatten();
            prepare_sink(output.sink, route.role, destination, label)
        })
        .collect()
}

fn prepare_sink(
    sink: SinkConfig,
    role: TargetSpec,
    destination: Option<PathBuf>,
    label: &str,
) -> PyResult<PreparedSink> {
    match (sink, destination) {
        (SinkConfig::ReturnArray, None) => Ok(PreparedSink::Return),
        (SinkConfig::ReturnTensor, None) => Ok(PreparedSink::ReturnTensor),
        (SinkConfig::Encoded { options, .. }, None) => Ok(PreparedSink::Encoded(options)),
        (
            SinkConfig::ReturnArray | SinkConfig::ReturnTensor | SinkConfig::Encoded { .. },
            Some(_),
        ) => Err(pyo3::exceptions::PyTypeError::new_err(
            "destination is only valid for Write output",
        )),
        (SinkConfig::Path { .. }, None) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "destination is required for {label} Write output"
        ))),
        (
            SinkConfig::Path {
                format,
                quality,
                compression,
            },
            Some(destination),
        ) => {
            let inferred = path_format(&destination);
            let format = format.or(inferred).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "format is required for {label} destination without a JPEG or PNG suffix"
                ))
            })?;
            if inferred.is_some_and(|inferred| inferred != format) {
                return Err(PyValueError::new_err(format!(
                    "{label} output format conflicts with the destination suffix"
                )));
            }
            if matches!(role, TargetSpec::Mask { .. }) && format != ImageFormat::Png {
                return Err(PyValueError::new_err("Mask Write output must use PNG"));
            }
            Ok(PreparedSink::Write {
                destination,
                options: encode_options_for_format(format, quality, compression)?,
            })
        }
    }
}

fn acquire_target(
    source: PreparedSource,
    route: &TargetRoute,
    label: String,
) -> Result<AcquiredTarget, PipelineRunError> {
    let acquired = match source {
        PreparedSource::Array {
            storage,
            height,
            width,
        } => Ok(AcquiredTarget {
            label: String::new(),
            storage,
            height,
            width,
        }),
        PreparedSource::Encoded(encoded) => decode_target(encoded, route),
        PreparedSource::Path(path) => {
            let max_encoded_bytes = match route.source {
                SourceConfig::Path {
                    max_encoded_bytes, ..
                } => max_encoded_bytes,
                _ => None,
            };
            read_pipeline_source(&path, max_encoded_bytes)
                .and_then(|encoded| decode_target(encoded, route))
        }
    }
    .map_err(|error| PipelineRunError::Target {
        label: label.clone(),
        error: Box::new(error),
    })?;
    Ok(AcquiredTarget { label, ..acquired })
}

fn decode_target(
    encoded: Vec<u8>,
    route: &TargetRoute,
) -> Result<AcquiredTarget, PipelineRunError> {
    let max_pixels = match route.source {
        SourceConfig::Encoded { max_pixels, .. } | SourceConfig::Path { max_pixels, .. } => {
            max_pixels
        }
        SourceConfig::Array => None,
    };
    match route.role {
        TargetSpec::Image => {
            let decoded = augment_io::decode_image(
                &encoded,
                DecodeOptions {
                    mode: DecodeMode::Rgb,
                    max_pixels,
                },
            )
            .map_err(PipelineRunError::Codec)?;
            let (data, height, width) = decoded_rgb_u8(decoded)?;
            Ok(AcquiredTarget {
                label: String::new(),
                storage: TargetStorage::Owned(data),
                height,
                width,
            })
        }
        TargetSpec::Mask { .. } => {
            if ImageFormat::detect(&encoded).map_err(PipelineRunError::Codec)? != ImageFormat::Png {
                return Err(PipelineRunError::InvalidTarget(
                    "semantic masks must use PNG input".into(),
                ));
            }
            let decoded = augment_io::decode_image(
                &encoded,
                DecodeOptions {
                    mode: DecodeMode::Unchanged,
                    max_pixels,
                },
            )
            .map_err(PipelineRunError::Codec)?;
            let (data, height, width) = decoded_mask_u8(decoded)?;
            Ok(AcquiredTarget {
                label: String::new(),
                storage: TargetStorage::Owned(data),
                height,
                width,
            })
        }
    }
}

fn decoded_mask_u8(image: DecodedImage) -> Result<(Vec<u8>, usize, usize), PipelineRunError> {
    if image.color != ColorModel::Gray || image.source_has_alpha {
        return Err(PipelineRunError::InvalidTarget(
            "semantic masks must use grayscale or indexed PNG without alpha".into(),
        ));
    }
    match image.pixels {
        PixelData::U8(data) => Ok((data, image.height, image.width)),
        PixelData::U16(_) => Err(PipelineRunError::InvalidTarget(
            "semantic masks must use 1, 2, 4, or 8-bit samples".into(),
        )),
    }
}

fn encode_target_output(
    output: &TargetOutput,
    options: EncodeOptions,
) -> Result<Vec<u8>, PipelineRunError> {
    let image = match output {
        TargetOutput::Image(output) => {
            let output = output.hwc.as_ref().ok_or(PipelineRunError::SinkContract)?;
            match output {
                PipelineOutput::U8Hwc {
                    data,
                    height,
                    width,
                } => ImageView {
                    pixels: PixelDataRef::U8(data),
                    height: *height,
                    width: *width,
                    color: ColorModel::Rgb,
                },
                _ => return Err(PipelineRunError::SinkContract),
            }
        }
        TargetOutput::Mask(output) => ImageView {
            pixels: PixelDataRef::U8(&output.data),
            height: output.height,
            width: output.width,
            color: ColorModel::Gray,
        },
    };
    augment_io::encode_image_view(image, options).map_err(PipelineRunError::Codec)
}

fn encode_options_for_format(
    format: ImageFormat,
    quality: Option<u8>,
    compression: Option<u8>,
) -> PyResult<EncodeOptions> {
    match format {
        ImageFormat::Jpeg => {
            if compression.is_some() {
                return Err(PyValueError::new_err("compression is only valid for PNG"));
            }
            Ok(EncodeOptions::Jpeg {
                quality: quality.unwrap_or(95),
            })
        }
        ImageFormat::Png => {
            if quality.is_some() {
                return Err(PyValueError::new_err("quality is only valid for JPEG"));
            }
            Ok(EncodeOptions::Png {
                compression: compression.unwrap_or(6),
            })
        }
    }
}

static NEXT_TEMPORARY_FILE: AtomicU64 = AtomicU64::new(0);

fn write_atomic(path: &Path, encoded: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "destination must have a file name",
        )
    })?;
    for _ in 0..32 {
        let nonce = NEXT_TEMPORARY_FILE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{}.variopinta-{}-{nonce}",
            file_name.to_string_lossy(),
            std::process::id()
        ));
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        let result = file
            .write_all(encoded)
            .and_then(|_| file.sync_all())
            .and_then(|_| std::fs::rename(&temporary, path));
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        return result;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a temporary sibling file",
    ))
}

fn parse_mode(value: &str) -> PyResult<ExecutionMode> {
    match value {
        "reference" => Ok(ExecutionMode::Reference),
        "compiled" => Ok(ExecutionMode::Compiled),
        "staged-fresh" => Ok(ExecutionMode::StagedFresh),
        "staged-reuse" => Ok(ExecutionMode::StagedReuse),
        _ => Err(PyValueError::new_err(
            "mode must be 'reference', 'compiled', 'staged-fresh', or 'staged-reuse'",
        )),
    }
}

fn parse_specs(value: &Bound<'_, PyAny>) -> PyResult<Vec<TransformSpec>> {
    value
        .downcast::<PyList>()?
        .iter()
        .map(|item| parse_spec(item.downcast::<PyDict>()?))
        .collect()
}

fn parse_spec(value: &Bound<'_, PyDict>) -> PyResult<TransformSpec> {
    let kind: String = required(value, "type")?.extract()?;
    let default_p = if kind == "HorizontalFlip" { 0.5 } else { 1.0 };
    let p = optional(value, "p")?.map_or(Ok(default_p), |value| value.extract())?;
    match kind.as_str() {
        "Resize" => Ok(TransformSpec::Resize {
            height: required(value, "height")?.extract()?,
            width: required(value, "width")?.extract()?,
            interpolation: parse_interpolation(
                &required(value, "interpolation")?.extract::<String>()?,
            )?,
            antialias: required(value, "antialias")?.extract()?,
            p,
        }),
        "RandomCrop" => Ok(TransformSpec::RandomCrop {
            height: required(value, "height")?.extract()?,
            width: required(value, "width")?.extract()?,
            p,
        }),
        "RandomResizedCrop" => Ok(TransformSpec::RandomResizedCrop {
            height: required(value, "height")?.extract()?,
            width: required(value, "width")?.extract()?,
            scale: required(value, "scale")?.extract()?,
            ratio: required(value, "ratio")?.extract()?,
            interpolation: parse_interpolation(
                &required(value, "interpolation")?.extract::<String>()?,
            )?,
            antialias: required(value, "antialias")?.extract()?,
            p,
        }),
        "HorizontalFlip" => Ok(TransformSpec::HorizontalFlip { p }),
        "VerticalFlip" => Ok(TransformSpec::VerticalFlip { p }),
        "CenterCrop" => Ok(TransformSpec::CenterCrop {
            height: required(value, "height")?.extract()?,
            width: required(value, "width")?.extract()?,
            p,
        }),
        "PadIfNeeded" => Ok(TransformSpec::PadIfNeeded {
            min_height: required(value, "min_height")?.extract()?,
            min_width: required(value, "min_width")?.extract()?,
            pad_height_divisor: required(value, "pad_height_divisor")?.extract()?,
            pad_width_divisor: required(value, "pad_width_divisor")?.extract()?,
            position: parse_pad_position(&required(value, "position")?.extract::<String>()?)?,
            border_mode: parse_border_mode(&required(value, "border_mode")?.extract::<String>()?)?,
            fill: required(value, "fill")?.extract()?,
            p,
        }),
        "CoarseDropout" => Ok(TransformSpec::CoarseDropout {
            num_holes_range: required(value, "num_holes_range")?.extract()?,
            hole_height_range: parse_dropout_size_range(
                value,
                "hole_height_range",
                "hole_height_unit",
            )?,
            hole_width_range: parse_dropout_size_range(
                value,
                "hole_width_range",
                "hole_width_unit",
            )?,
            fill: required(value, "fill")?.extract()?,
            p,
        }),
        "ColorJitter" => Ok(TransformSpec::ColorJitter {
            brightness: required(value, "brightness")?.extract()?,
            contrast: required(value, "contrast")?.extract()?,
            saturation: required(value, "saturation")?.extract()?,
            hue: required(value, "hue")?.extract()?,
            p,
        }),
        "Affine" => Ok(TransformSpec::Affine {
            degrees: required(value, "degrees")?.extract()?,
            translate: required(value, "translate")?.extract()?,
            scale: required(value, "scale")?.extract()?,
            shear: required(value, "shear")?.extract()?,
            interpolation: parse_interpolation(
                &required(value, "interpolation")?.extract::<String>()?,
            )?,
            border_mode: parse_border_mode(&required(value, "border_mode")?.extract::<String>()?)?,
            fill: required(value, "fill")?.extract()?,
            p,
        }),
        "RandomRotation" => Ok(TransformSpec::RandomRotation {
            degrees: required(value, "degrees")?.extract()?,
            interpolation: parse_interpolation(
                &required(value, "interpolation")?.extract::<String>()?,
            )?,
            border_mode: parse_border_mode(&required(value, "border_mode")?.extract::<String>()?)?,
            fill: required(value, "fill")?.extract()?,
            p,
        }),
        "GaussianNoise" => Ok(TransformSpec::GaussianNoise {
            mean: required(value, "mean")?.extract()?,
            std: required(value, "std")?.extract()?,
            per_channel: required(value, "per_channel")?.extract()?,
            p,
        }),
        "Sharpen" => Ok(TransformSpec::Sharpen {
            alpha: required(value, "alpha")?.extract()?,
            lightness: required(value, "lightness")?.extract()?,
            p,
        }),
        "Perspective" => Ok(TransformSpec::Perspective {
            scale: required(value, "scale")?.extract()?,
            interpolation: parse_interpolation(
                &required(value, "interpolation")?.extract::<String>()?,
            )?,
            border_mode: parse_border_mode(&required(value, "border_mode")?.extract::<String>()?)?,
            fill: required(value, "fill")?.extract()?,
            p,
        }),
        "GridDistortion" => Ok(TransformSpec::GridDistortion {
            num_steps: required(value, "num_steps")?.extract()?,
            distort_limit: required(value, "distort_limit")?.extract()?,
            interpolation: parse_interpolation(
                &required(value, "interpolation")?.extract::<String>()?,
            )?,
            border_mode: parse_border_mode(&required(value, "border_mode")?.extract::<String>()?)?,
            fill: required(value, "fill")?.extract()?,
            p,
        }),
        "GaussianBlur" => Ok(TransformSpec::GaussianBlur {
            kernel_size: required(value, "kernel_size")?.extract()?,
            sigma: required(value, "sigma")?.extract()?,
            p,
        }),
        "Grayscale" => Ok(TransformSpec::Grayscale { p }),
        "Invert" => Ok(TransformSpec::Invert { p }),
        "Solarize" => Ok(TransformSpec::Solarize {
            threshold: required(value, "threshold")?.extract()?,
            p,
        }),
        "Posterize" => Ok(TransformSpec::Posterize {
            bits: required(value, "bits")?.extract()?,
            p,
        }),
        "Normalize" => Ok(TransformSpec::Normalize {
            mean: required(value, "mean")?.extract()?,
            std: required(value, "std")?.extract()?,
            max_pixel_value: optional(value, "max_pixel_value")?
                .map_or(Ok(255.0), |value| value.extract())?,
            p,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unknown transform type: {kind}"
        ))),
    }
}

fn parse_interpolation(value: &str) -> PyResult<Interpolation> {
    match value {
        "nearest" => Ok(Interpolation::Nearest),
        "bilinear" => Ok(Interpolation::Bilinear),
        _ => Err(PyValueError::new_err(
            "interpolation must be 'nearest' or 'bilinear'",
        )),
    }
}

fn parse_border_mode(value: &str) -> PyResult<BorderMode> {
    match value {
        "constant" => Ok(BorderMode::Constant),
        "reflect101" => Ok(BorderMode::Reflect101),
        _ => Err(PyValueError::new_err(
            "border_mode must be 'constant' or 'reflect101'",
        )),
    }
}

fn parse_pad_position(value: &str) -> PyResult<PadPosition> {
    match value {
        "center" => Ok(PadPosition::Center),
        "top_left" => Ok(PadPosition::TopLeft),
        "top_right" => Ok(PadPosition::TopRight),
        "bottom_left" => Ok(PadPosition::BottomLeft),
        "bottom_right" => Ok(PadPosition::BottomRight),
        "random" => Ok(PadPosition::Random),
        _ => Err(PyValueError::new_err(
            "position must be 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', or 'random'",
        )),
    }
}

fn parse_dropout_size_range(
    value: &Bound<'_, PyDict>,
    range_key: &str,
    unit_key: &str,
) -> PyResult<DropoutSizeRange> {
    match required(value, unit_key)?.extract::<String>()?.as_str() {
        "fraction" => Ok(DropoutSizeRange::Fraction(
            required(value, range_key)?.extract()?,
        )),
        "pixels" => Ok(DropoutSizeRange::Pixels(
            required(value, range_key)?.extract()?,
        )),
        _ => Err(PyValueError::new_err(format!(
            "{unit_key} must be 'fraction' or 'pixels'"
        ))),
    }
}

fn required<'py>(value: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    value
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing transform field: {key}")))
}

fn optional<'py>(value: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    value.get_item(key)
}

fn output_to_python(py: Python<'_>, output: PipelineOutput) -> PyResult<Py<PyAny>> {
    match output {
        PipelineOutput::U8Hwc {
            data,
            height,
            width,
        } => Ok(data
            .into_pyarray(py)
            .reshape([height, width, 3])?
            .into_any()
            .unbind()),
        PipelineOutput::F32Hwc {
            data,
            height,
            width,
        } => Ok(data
            .into_pyarray(py)
            .reshape([height, width, 3])?
            .into_any()
            .unbind()),
        PipelineOutput::U8Chw {
            data,
            height,
            width,
        } => Ok(data
            .into_pyarray(py)
            .reshape([3, height, width])?
            .into_any()
            .unbind()),
        PipelineOutput::F32Chw {
            data,
            height,
            width,
        } => Ok(data
            .into_pyarray(py)
            .reshape([3, height, width])?
            .into_any()
            .unbind()),
    }
}

fn mask_output_to_python(py: Python<'_>, output: MaskOutput) -> PyResult<Py<PyAny>> {
    let array = PyArray2::<u8>::zeros(py, [output.height, output.width], false);
    array
        .readwrite()
        .as_slice_mut()
        .map_err(|_| PyRuntimeError::new_err("allocated mask output is not contiguous"))?
        .copy_from_slice(&output.data);
    Ok(array.into_any().unbind())
}

fn native_output_to_python(py: Python<'_>, output: NativeOutput) -> PyResult<Py<PyAny>> {
    match output {
        NativeOutput::ReturnImage(output) => output_to_python(py, output),
        NativeOutput::ReturnMask(output) => mask_output_to_python(py, output),
        NativeOutput::Encoded(encoded) => Ok(PyBytes::new(py, &encoded).into_any().unbind()),
        NativeOutput::Written => Ok(py.None()),
    }
}

fn native_outputs_to_python(
    py: Python<'_>,
    outputs: Vec<Vec<NativeOutput>>,
) -> PyResult<Py<PyAny>> {
    let outputs = outputs
        .into_iter()
        .map(|target_outputs| {
            let target_outputs = target_outputs
                .into_iter()
                .map(|output| native_output_to_python(py, output))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, target_outputs)?.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, outputs)?.into_any().unbind())
}

fn decoded_rgb_u8(image: DecodedImage) -> Result<(Vec<u8>, usize, usize), PipelineRunError> {
    if image.height == 0 || image.width == 0 || image.color != ColorModel::Rgb {
        return Err(PipelineRunError::DecodedContract(
            "decoded pipeline input must be non-empty RGB",
        ));
    }
    match image.pixels {
        PixelData::U8(pixels) => Ok((pixels, image.height, image.width)),
        PixelData::U16(_) => Err(PipelineRunError::DecodedContract(
            "decoded pipeline input dtype must be uint8",
        )),
    }
}

fn explanation_to_python(
    py: Python<'_>,
    value: PipelineExplanation,
    targets: &[TargetRoute],
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("schema_version", EXPLANATION_SCHEMA_VERSION)?;
    output.set_item("mode", value.mode)?;
    output.set_item("sampling", value.sampling)?;
    output.set_item("transforms", &value.transforms)?;
    let steps = value
        .steps
        .clone()
        .into_iter()
        .map(|step| transform_explanation_to_python(py, step))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("steps", PyList::new(py, steps)?)?;
    let direct_chw = targets
        .iter()
        .any(|target| uses_direct_normalize_chw(target, &value));
    output.set_item(
        "fusions",
        if direct_chw {
            vec!["Normalize+terminal-layout:direct-CHW"]
        } else {
            value.fusions.clone()
        },
    )?;
    output.set_item("unit_specializations", &value.unit_specializations)?;
    output.set_item("optimizations", &value.optimizations)?;
    output.set_item("passes", value.passes)?;
    let image_count = targets
        .iter()
        .filter(|target| target.role == TargetSpec::Image)
        .count();
    let mask_count = targets.len() - image_count;
    let aggregate_pixel_passes = image_count * value.pixel_passes
        + mask_count * value.mask_plan.pixel_passes.saturating_sub(1);
    output.set_item("pixel_passes", aggregate_pixel_passes)?;
    output.set_item(
        "input",
        image_contract_to_python(py, value.input.clone(), None)?,
    )?;
    let holds_gil = targets
        .iter()
        .any(|target| matches!(target.source, SourceConfig::Array));
    let target_values = targets
        .iter()
        .enumerate()
        .map(|(index, target)| target_explanation_to_python(py, index, target, &value, holds_gil))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("targets", PyList::new(py, target_values)?)?;
    output.set_item("fallbacks", value.fallbacks)?;
    let boundary = PyDict::new(py);
    boundary.set_item("crossings_per_call", 1)?;
    boundary.set_item(
        "gil",
        if holds_gil {
            "held-during-aggregate-array-call"
        } else {
            "released-during-acquisition-augmentation-and-delivery"
        },
    )?;
    boundary.set_item("binding_validation", "before-acquisition-and-sampling")?;
    boundary.set_item("augmentation", if holds_gil { "held" } else { "released" })?;
    output.set_item("python_boundary", boundary)?;
    Ok(output.into_any().unbind())
}

fn target_explanation_to_python(
    py: Python<'_>,
    index: usize,
    target: &TargetRoute,
    plan: &PipelineExplanation,
    aggregate_holds_gil: bool,
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    let is_image = target.role == TargetSpec::Image;
    output.set_item("index", index)?;
    output.set_item("name", target.name.as_deref())?;
    output.set_item("role", if is_image { "image" } else { "mask" })?;
    output.set_item(
        "fill",
        match target.role {
            TargetSpec::Image => None,
            TargetSpec::Mask { fill } => Some(fill),
        },
    )?;
    output.set_item(
        "carrier",
        source_explanation_to_python(py, target.source, target.role)?,
    )?;
    let outputs = target
        .outputs
        .iter()
        .enumerate()
        .map(|(output_index, route)| {
            output_explanation_to_python(py, output_index, route, target, plan)
        })
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("outputs", PyList::new(py, outputs)?)?;
    output.set_item(
        "input_dtype",
        if is_image { plan.input.dtype } else { "uint8" },
    )?;
    output.set_item(
        "input_layout",
        if is_image { plan.input.layout } else { "HW" },
    )?;
    output.set_item("ownership", "owned-contiguous-result")?;
    output.set_item(
        "gil",
        if aggregate_holds_gil {
            "held-during-aggregate-array-call"
        } else {
            "released-during-native-work"
        },
    )?;
    output.set_item(
        "pixel_passes",
        if is_image {
            plan.pixel_passes
        } else {
            plan.mask_plan.pixel_passes.saturating_sub(1)
        },
    )?;
    if !is_image {
        let steps = plan
            .mask_plan
            .steps
            .iter()
            .cloned()
            .map(|mut step| {
                step.fill = step.fill.map(|_| match target.role {
                    TargetSpec::Mask { fill } => fill,
                    TargetSpec::Image => 0,
                });
                mask_transform_explanation_to_python(py, step)
            })
            .collect::<PyResult<Vec<_>>>()?;
        output.set_item("steps", PyList::new(py, steps)?)?;
    }
    let mut copies = vec![match target.source {
        SourceConfig::Array => CopyExplanation {
            stage: "target-acquisition",
            count: "0-or-1",
            condition: "non-contiguous-input",
            reason: "normalize-array-into-call-owned-storage",
        },
        SourceConfig::Encoded { .. } => CopyExplanation {
            stage: "target-acquisition",
            count: "1",
            condition: "always",
            reason: "snapshot-compressed-input-into-call-owned-storage",
        },
        SourceConfig::Path { .. } => CopyExplanation {
            stage: "target-acquisition",
            count: "0",
            condition: "always",
            reason: "read-file-directly-into-call-owned-storage",
        },
    }];
    if is_image {
        copies.extend(plan.copies.iter().cloned());
        if let Some(copy) = terminal_chw_copy_explanation(target, plan) {
            copies.push(copy);
        }
    } else {
        copies.push(match target.source {
            SourceConfig::Array => CopyExplanation {
                stage: "target-native-entry",
                count: "0-or-1",
                condition: "contiguous-array-input",
                reason: "establish-owned-mask-result",
            },
            SourceConfig::Encoded { .. } | SourceConfig::Path { .. } => CopyExplanation {
                stage: "target-native-entry",
                count: "0",
                condition: "always",
                reason: "adopt-call-owned-mask-storage",
            },
        });
    }
    for route in &target.outputs {
        copies.push(match (target.role, route.sink) {
            (TargetSpec::Mask { .. }, SinkConfig::ReturnArray | SinkConfig::ReturnTensor) => {
                CopyExplanation {
                    stage: "target-python-output",
                    count: "1",
                    condition: "per-return-output",
                    reason: "isolate-returned-mask-storage",
                }
            }
            (_, SinkConfig::ReturnArray | SinkConfig::ReturnTensor) => CopyExplanation {
                stage: "target-python-output",
                count: "0-or-1",
                condition: "sibling-return-isolation",
                reason: "transfer-or-isolate-returned-storage",
            },
            (_, SinkConfig::Encoded { .. }) => CopyExplanation {
                stage: "target-python-output",
                count: "1",
                condition: "per-encoded-output",
                reason: "copy-compressed-buffer-into-Python-bytes",
            },
            (_, SinkConfig::Path { .. }) => CopyExplanation {
                stage: "target-python-output",
                count: "0",
                condition: "per-written-output",
                reason: "no-Python-raster-transfer",
            },
        });
    }
    let copies = copies
        .into_iter()
        .map(|copy| copy_explanation_to_python(py, copy))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("copies", PyList::new(py, copies)?)?;

    let input_dtype = "uint8";
    let input_layout = if is_image { "HWC" } else { "HW" };
    let output_dtype = if is_image { plan.output_dtype } else { "uint8" };
    let output_layout = if is_image { "per-output" } else { "HW" };
    let mut buffers = Vec::new();
    match target.source {
        SourceConfig::Array => buffers.push(BufferExplanation {
            name: "target-array-input",
            dtype: input_dtype,
            layout: input_layout,
            lifecycle: "borrowed-or-normalized-for-call",
            condition: "always",
        }),
        SourceConfig::Encoded { .. } => {
            buffers.push(BufferExplanation {
                name: "target-encoded-input",
                dtype: "uint8",
                layout: "encoded",
                lifecycle: "call-owned-snapshot",
                condition: "always",
            });
            buffers.push(BufferExplanation {
                name: "target-decoded-input",
                dtype: input_dtype,
                layout: input_layout,
                lifecycle: "call-owned",
                condition: "always",
            });
        }
        SourceConfig::Path { .. } => {
            buffers.push(BufferExplanation {
                name: "target-encoded-input",
                dtype: "uint8",
                layout: "encoded",
                lifecycle: "call-owned-file-read",
                condition: "always",
            });
            buffers.push(BufferExplanation {
                name: "target-decoded-input",
                dtype: input_dtype,
                layout: input_layout,
                lifecycle: "call-owned",
                condition: "always",
            });
        }
    }
    if is_image {
        let all_steps_never = plan.steps.iter().all(|step| step.status == "never");
        let direct_normalize_chw = uses_direct_normalize_chw(target, plan);
        buffers.extend(plan.buffers.iter().cloned().map(|mut buffer| {
            if plan.mode == "compiled" && all_steps_never && buffer.name == "working-u8" {
                buffer.condition = "not-required";
            }
            if direct_normalize_chw && buffer.name == "output-f32" {
                buffer.layout = "CHW";
            }
            buffer
        }));
    }
    buffers.push(BufferExplanation {
        name: "target-result",
        dtype: output_dtype,
        layout: output_layout,
        lifecycle: "owned-per-run",
        condition: "always",
    });
    if target.outputs.iter().any(|route| {
        matches!(
            route.sink,
            SinkConfig::Encoded { .. } | SinkConfig::Path { .. }
        )
    }) {
        buffers.push(BufferExplanation {
            name: "target-encoded-output",
            dtype: "uint8",
            layout: "encoded",
            lifecycle: "call-owned",
            condition: "always",
        });
    }
    let buffers = buffers
        .into_iter()
        .map(|buffer| buffer_explanation_to_python(py, buffer))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("buffers", PyList::new(py, buffers)?)?;
    Ok(output.into_any().unbind())
}

fn mask_transform_explanation_to_python(
    py: Python<'_>,
    value: MaskTransformExplanation,
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("name", value.name)?;
    output.set_item("classification", value.classification)?;
    output.set_item("raster_policy", value.raster_policy)?;
    output.set_item("pixel_passes", value.pixel_passes)?;
    output.set_item("fill", value.fill)?;
    Ok(output.into_any().unbind())
}

fn source_explanation_to_python(
    py: Python<'_>,
    source: SourceConfig,
    role: TargetSpec,
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    match source {
        SourceConfig::Array => output.set_item("type", "array")?,
        SourceConfig::Encoded {
            max_pixels,
            max_encoded_bytes,
        }
        | SourceConfig::Path {
            max_pixels,
            max_encoded_bytes,
        } => {
            output.set_item(
                "type",
                if matches!(source, SourceConfig::Encoded { .. }) {
                    "encoded"
                } else {
                    "path"
                },
            )?;
            match role {
                TargetSpec::Image => {
                    output.set_item("mode", "rgb")?;
                    output.set_item("formats", ["jpeg", "png"])?;
                }
                TargetSpec::Mask { .. } => {
                    output.set_item("mode", "unchanged")?;
                    output.set_item("formats", ["png"])?;
                    output.set_item(
                        "sample_policy",
                        "preserve-grayscale-samples-or-palette-indices",
                    )?;
                }
            }
            output.set_item("max_pixels", max_pixels)?;
            output.set_item("max_encoded_bytes", max_encoded_bytes)?;
        }
    }
    Ok(output.into_any().unbind())
}

fn output_explanation_to_python(
    py: Python<'_>,
    output_index: usize,
    route: &OutputRoute,
    target: &TargetRoute,
    plan: &PipelineExplanation,
) -> PyResult<Py<PyAny>> {
    let output = sink_explanation_to_python(py, route.sink, target.role)?;
    let binding = output.bind(py);
    let dictionary = binding.downcast::<PyDict>()?;
    let is_mask = matches!(target.role, TargetSpec::Mask { .. });
    let tensor = matches!(route.sink, SinkConfig::ReturnTensor);
    let layout = if is_mask {
        "HW"
    } else if tensor {
        "CHW"
    } else {
        "HWC"
    };
    let container = match route.sink {
        SinkConfig::ReturnArray => "NumPy ndarray",
        SinkConfig::ReturnTensor => "Torch CPU Tensor",
        SinkConfig::Encoded { .. } => "bytes",
        SinkConfig::Path { .. } => "pathlib.Path",
    };
    dictionary.set_item("name", route.name.as_deref())?;
    dictionary.set_item("container", container)?;
    dictionary.set_item("layout", layout)?;
    dictionary.set_item("dtype", if is_mask { "uint8" } else { plan.output_dtype })?;
    dictionary.set_item("contiguous", true)?;
    dictionary.set_item("ownership", "independent-result")?;
    let terminal_layout = if is_mask {
        "semantic-HW-raster"
    } else if !tensor {
        "semantic-HWC-raster"
    } else if target_requirements(target).hwc {
        "from-shared-HWC-raster"
    } else if plan.mode == "compiled" && effective_terminal_normalize_status(plan) == Some("always")
    {
        "direct-CHW"
    } else if plan.mode == "compiled"
        && (effective_terminal_normalize_status(plan) == Some("conditional")
            || (effective_terminal_normalize_status(plan).is_none()
                && plan.steps.iter().any(|step| step.status == "conditional")))
    {
        "sample-dependent-direct-CHW-or-terminal-copy"
    } else if plan.mode == "compiled" && plan.steps.iter().all(|step| step.status == "never") {
        "direct-CHW"
    } else {
        "terminal-HWC-to-CHW-copy"
    };
    dictionary.set_item("terminal_layout", terminal_layout)?;
    dictionary.set_item(
        "delivery",
        match route.sink {
            SinkConfig::ReturnArray => "numpy",
            SinkConfig::ReturnTensor => "torch-from-numpy",
            SinkConfig::Encoded { .. } => "python-bytes",
            SinkConfig::Path { .. } => "atomic-file-replace",
        },
    )?;
    let copies = match route.sink {
        SinkConfig::ReturnArray | SinkConfig::ReturnTensor => {
            let copied = target.outputs[output_index + 1..]
                .iter()
                .any(|other| outputs_share_return_artifact(target.role, route.sink, other.sink));
            vec![CopyExplanation {
                stage: "output-delivery",
                count: if copied { "1" } else { "0" },
                condition: "always",
                reason: if copied {
                    "isolate-returned-sibling-storage"
                } else {
                    "transfer-owned-terminal-artifact"
                },
            }]
        }
        SinkConfig::Encoded { .. } => vec![CopyExplanation {
            stage: "output-delivery",
            count: "1",
            condition: "always",
            reason: "copy-compressed-buffer-into-Python-bytes",
        }],
        SinkConfig::Path { .. } => Vec::new(),
    };
    let copies = copies
        .into_iter()
        .map(|copy| copy_explanation_to_python(py, copy))
        .collect::<PyResult<Vec<_>>>()?;
    dictionary.set_item("copies", PyList::new(py, copies)?)?;
    let dtype = if is_mask { "uint8" } else { plan.output_dtype };
    let mut buffers = vec![BufferExplanation {
        name: "terminal-raster",
        dtype,
        layout,
        lifecycle: match route.sink {
            SinkConfig::ReturnArray | SinkConfig::ReturnTensor => "transferred-or-isolated",
            SinkConfig::Encoded { .. } | SinkConfig::Path { .. } => "borrowed-during-encoding",
        },
        condition: "always",
    }];
    if matches!(
        route.sink,
        SinkConfig::Encoded { .. } | SinkConfig::Path { .. }
    ) {
        buffers.push(BufferExplanation {
            name: "encoded-output",
            dtype: "uint8",
            layout: "encoded",
            lifecycle: "owned-per-output",
            condition: "always",
        });
    }
    let buffers = buffers
        .into_iter()
        .map(|buffer| buffer_explanation_to_python(py, buffer))
        .collect::<PyResult<Vec<_>>>()?;
    dictionary.set_item("buffers", PyList::new(py, buffers)?)?;
    Ok(output)
}

fn effective_terminal_normalize_status(plan: &PipelineExplanation) -> Option<&'static str> {
    plan.steps
        .iter()
        .rev()
        .find(|step| step.status != "never")
        .filter(|step| step.name == "Normalize")
        .map(|step| step.status)
}

fn uses_direct_normalize_chw(target: &TargetRoute, plan: &PipelineExplanation) -> bool {
    let requirements = target_requirements(target);
    target.role == TargetSpec::Image
        && plan.mode == "compiled"
        && requirements.chw
        && !requirements.hwc
        && effective_terminal_normalize_status(plan).is_some()
}

fn terminal_chw_copy_explanation(
    target: &TargetRoute,
    plan: &PipelineExplanation,
) -> Option<CopyExplanation> {
    let requirements = target_requirements(target);
    if target.role != TargetSpec::Image || !requirements.chw {
        return None;
    }
    if uses_direct_normalize_chw(target, plan) {
        if effective_terminal_normalize_status(plan) == Some("always") {
            return None;
        }
        return Some(CopyExplanation {
            stage: "terminal-layout",
            count: "0-or-1",
            condition: "sample-dependent",
            reason: "materialize-contiguous-CHW-when-Normalize-is-skipped",
        });
    }
    Some(CopyExplanation {
        stage: "terminal-layout",
        count: "1",
        condition: "always",
        reason: "materialize-contiguous-CHW-raster",
    })
}

fn outputs_share_return_artifact(role: TargetSpec, left: SinkConfig, right: SinkConfig) -> bool {
    match role {
        TargetSpec::Mask { .. } => matches!(
            (left, right),
            (
                SinkConfig::ReturnArray | SinkConfig::ReturnTensor,
                SinkConfig::ReturnArray | SinkConfig::ReturnTensor
            )
        ),
        TargetSpec::Image => matches!(
            (left, right),
            (SinkConfig::ReturnArray, SinkConfig::ReturnArray)
                | (SinkConfig::ReturnTensor, SinkConfig::ReturnTensor)
        ),
    }
}

fn sink_explanation_to_python(
    py: Python<'_>,
    sink: SinkConfig,
    role: TargetSpec,
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    match sink {
        SinkConfig::ReturnArray => output.set_item("type", "return_array")?,
        SinkConfig::ReturnTensor => output.set_item("type", "return_tensor")?,
        SinkConfig::Encoded { format, options } => {
            output.set_item("type", "encode")?;
            output.set_item("format", image_format_name(format))?;
            match options {
                EncodeOptions::Jpeg { quality } => output.set_item("quality", quality)?,
                EncodeOptions::Png { compression } => {
                    output.set_item("compression", compression)?
                }
            }
        }
        SinkConfig::Path {
            format,
            quality,
            compression,
        } => {
            let format_inference = format.is_none();
            let format = match role {
                TargetSpec::Image => format,
                TargetSpec::Mask { .. } => Some(ImageFormat::Png),
            };
            output.set_item("type", "write")?;
            output.set_item("format", format.map(image_format_name))?;
            output.set_item("quality", quality)?;
            output.set_item("compression", compression)?;
            output.set_item("format_inference", format_inference)?;
        }
    }
    Ok(output.into_any().unbind())
}

fn transform_explanation_to_python(
    py: Python<'_>,
    value: TransformExplanation,
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("name", value.name)?;
    output.set_item("category", value.category)?;
    output.set_item("probability", value.probability)?;
    output.set_item("status", value.status)?;
    output.set_item("execution", value.execution)?;
    output.set_item("pixel_passes", value.pixel_passes)?;
    output.set_item("allocation", value.allocation)?;
    output.set_item("fallback", value.fallback)?;
    output.set_item("input_materialization", value.input_materialization)?;
    output.set_item("kernel_form", value.kernel_form)?;
    output.set_item("output_slot", value.output_slot)?;
    output.set_item("scratch_slots", value.scratch_slots)?;
    output.set_item("selection_reason", value.selection_reason)?;
    let policies = value
        .policies
        .into_iter()
        .map(|policy| policy_explanation_to_python(py, policy))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("policies", PyList::new(py, policies)?)?;
    Ok(output.into_any().unbind())
}

fn policy_explanation_to_python(py: Python<'_>, value: PolicyExplanation) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("name", value.name)?;
    output.set_item("value", value.value)?;
    Ok(output.into_any().unbind())
}

fn image_contract_to_python(
    py: Python<'_>,
    value: ImageContractExplanation,
    container: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("container", container.unwrap_or(value.container))?;
    output.set_item("dtype", value.dtype)?;
    output.set_item("layout", value.layout)?;
    output.set_item("channels", value.channels)?;
    output.set_item("contiguous", value.contiguous)?;
    output.set_item("ownership", value.ownership)?;
    Ok(output.into_any().unbind())
}

fn buffer_explanation_to_python(py: Python<'_>, value: BufferExplanation) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("name", value.name)?;
    output.set_item("dtype", value.dtype)?;
    output.set_item("layout", value.layout)?;
    output.set_item("lifecycle", value.lifecycle)?;
    output.set_item("condition", value.condition)?;
    Ok(output.into_any().unbind())
}

fn copy_explanation_to_python(py: Python<'_>, value: CopyExplanation) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("stage", value.stage)?;
    output.set_item("count", value.count)?;
    output.set_item("condition", value.condition)?;
    output.set_item("reason", value.reason)?;
    Ok(output.into_any().unbind())
}

fn map_core_error(error: CoreError) -> PyErr {
    match error {
        CoreError::Invalid(message) => PyValueError::new_err(message),
        CoreError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}

#[pyfunction]
fn boundary(value: Py<PyAny>) -> Py<PyAny> {
    value
}

#[pyfunction]
#[pyo3(signature = (
    path,
    mode = "rgb",
    max_pixels = Some(100_000_000),
    max_encoded_bytes = None,
))]
fn read_image(
    py: Python<'_>,
    path: PathBuf,
    mode: &str,
    max_pixels: Option<usize>,
    max_encoded_bytes: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let options = decode_options(mode, max_pixels)?;
    validate_encoded_limit(max_encoded_bytes)?;
    let source_path = path.clone();
    let encoded = py
        .allow_threads(move || read_encoded(&path, max_encoded_bytes))
        .map_err(|error| map_read_error(error, &source_path))?;
    let image = py
        .allow_threads(move || augment_io::decode_image(&encoded, options))
        .map_err(map_codec_error)?;
    decoded_to_python(py, image)
}

#[pyfunction]
#[pyo3(signature = (
    data,
    mode = "rgb",
    max_pixels = Some(100_000_000),
    max_encoded_bytes = None,
))]
fn decode_image(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    mode: &str,
    max_pixels: Option<usize>,
    max_encoded_bytes: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let options = decode_options(mode, max_pixels)?;
    let encoded = snapshot_encoded(py, data, max_encoded_bytes, "image")?;
    let image = py
        .allow_threads(move || augment_io::decode_image(&encoded, options))
        .map_err(map_codec_error)?;
    decoded_to_python(py, image)
}

#[pyfunction]
#[pyo3(signature = (image, format, quality = None, compression = None))]
fn encode_image(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    format: &str,
    quality: Option<u8>,
    compression: Option<u8>,
) -> PyResult<Py<PyBytes>> {
    let image = owned_image(image)?;
    let options = encode_options(format, quality, compression)?;
    let encoded = py
        .allow_threads(move || augment_io::encode_image(&image, options))
        .map_err(map_codec_error)?;
    Ok(PyBytes::new(py, &encoded).unbind())
}

#[pyfunction]
#[pyo3(signature = (path, image, format, quality = None, compression = None))]
fn write_image(
    py: Python<'_>,
    path: PathBuf,
    image: &Bound<'_, PyAny>,
    format: &str,
    quality: Option<u8>,
    compression: Option<u8>,
) -> PyResult<()> {
    let image = owned_image(image)?;
    let options = encode_options(format, quality, compression)?;
    let encoded = py
        .allow_threads(move || augment_io::encode_image(&image, options))
        .map_err(map_codec_error)?;
    let destination = path.clone();
    py.allow_threads(move || std::fs::write(path, encoded))
        .map_err(|error| map_path_io_error("write destination", &destination, error))
}

fn decode_options(mode: &str, max_pixels: Option<usize>) -> PyResult<DecodeOptions> {
    if max_pixels == Some(0) {
        return Err(PyValueError::new_err("max_pixels must be positive or None"));
    }
    let mode = match mode {
        "unchanged" => DecodeMode::Unchanged,
        "gray" => DecodeMode::Gray,
        "rgb" => DecodeMode::Rgb,
        "rgba" => DecodeMode::Rgba,
        _ => {
            return Err(PyValueError::new_err(
                "mode must be 'unchanged', 'gray', 'rgb', or 'rgba'",
            ));
        }
    };
    Ok(DecodeOptions { mode, max_pixels })
}

fn validate_encoded_limit(max_encoded_bytes: Option<usize>) -> PyResult<()> {
    if max_encoded_bytes == Some(0) {
        return Err(PyValueError::new_err(
            "max_encoded_bytes must be positive or None",
        ));
    }
    Ok(())
}

fn snapshot_encoded(
    _py: Python<'_>,
    data: &Bound<'_, PyAny>,
    max_encoded_bytes: Option<usize>,
    role: &str,
) -> PyResult<Vec<u8>> {
    validate_encoded_limit(max_encoded_bytes)?;
    if let Ok(bytes) = data.downcast::<PyBytes>() {
        check_encoded_size(bytes.as_bytes().len(), max_encoded_bytes, role)?;
        return Ok(bytes.as_bytes().to_vec());
    }
    if let Ok(bytearray) = data.downcast::<PyByteArray>() {
        check_encoded_size(bytearray.len(), max_encoded_bytes, role)?;
        return Ok(bytearray.to_vec());
    }
    if data.downcast::<PyMemoryView>().is_ok() {
        let length: usize = data.getattr("nbytes")?.extract()?;
        check_encoded_size(length, max_encoded_bytes, role)?;
        let snapshot = data.call_method0("tobytes")?;
        let bytes = snapshot.downcast::<PyBytes>()?;
        return Ok(bytes.as_bytes().to_vec());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "encoded data must be bytes, bytearray, or memoryview",
    ))
}

fn check_encoded_size(length: usize, limit: Option<usize>, role: &str) -> PyResult<()> {
    if let Some(limit) = limit.filter(|limit| length > *limit) {
        return Err(PyValueError::new_err(format!(
            "encoded {role} has {length} bytes, exceeding the configured limit of {limit}"
        )));
    }
    Ok(())
}

enum ReadEncodedError {
    Io(std::io::Error),
    Limit { actual: usize, limit: usize },
}

fn read_encoded(path: &Path, limit: Option<usize>) -> Result<Vec<u8>, ReadEncodedError> {
    let file = std::fs::File::open(path).map_err(ReadEncodedError::Io)?;
    let mut encoded = Vec::new();
    match limit {
        Some(limit) => {
            let read_limit = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
            file.take(read_limit)
                .read_to_end(&mut encoded)
                .map_err(ReadEncodedError::Io)?;
            if encoded.len() > limit {
                return Err(ReadEncodedError::Limit {
                    actual: encoded.len(),
                    limit,
                });
            }
        }
        None => {
            let mut file = file;
            file.read_to_end(&mut encoded)
                .map_err(ReadEncodedError::Io)?;
        }
    }
    Ok(encoded)
}

fn read_pipeline_source(path: &Path, limit: Option<usize>) -> Result<Vec<u8>, PipelineRunError> {
    read_encoded(path, limit).map_err(|error| match error {
        ReadEncodedError::Io(error) => PipelineRunError::Read {
            path: path.to_path_buf(),
            error,
        },
        ReadEncodedError::Limit { actual, limit } => {
            PipelineRunError::EncodedLimit { actual, limit }
        }
    })
}

fn image_format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
    }
}

fn path_format(path: &Path) -> Option<ImageFormat> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
        "png" => Some(ImageFormat::Png),
        _ => None,
    }
}

fn encode_options(
    format: &str,
    quality: Option<u8>,
    compression: Option<u8>,
) -> PyResult<EncodeOptions> {
    match parse_image_format(format)? {
        ImageFormat::Jpeg => {
            if compression.is_some() {
                return Err(PyValueError::new_err("compression is only valid for PNG"));
            }
            Ok(EncodeOptions::Jpeg {
                quality: quality.unwrap_or(95),
            })
        }
        ImageFormat::Png => {
            if quality.is_some() {
                return Err(PyValueError::new_err("quality is only valid for JPEG"));
            }
            Ok(EncodeOptions::Png {
                compression: compression.unwrap_or(6),
            })
        }
    }
}

fn parse_image_format(format: &str) -> PyResult<ImageFormat> {
    match format {
        "jpeg" => Ok(ImageFormat::Jpeg),
        "png" => Ok(ImageFormat::Png),
        _ => Err(PyValueError::new_err("format must be 'jpeg' or 'png'")),
    }
}

fn owned_image(image: &Bound<'_, PyAny>) -> PyResult<OwnedImage> {
    if let Ok(array) = image.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        let (height, width, color) = image_shape(array.shape())?;
        let pixels = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("image must be C-contiguous"))?
            .to_vec();
        return Ok(OwnedImage {
            pixels: PixelData::U8(pixels),
            height,
            width,
            color,
        });
    }
    if let Ok(array) = image.extract::<PyReadonlyArrayDyn<'_, u16>>() {
        let (height, width, color) = image_shape(array.shape())?;
        let pixels = array
            .as_slice()
            .map_err(|_| PyValueError::new_err("image must be C-contiguous"))?
            .to_vec();
        return Ok(OwnedImage {
            pixels: PixelData::U16(pixels),
            height,
            width,
            color,
        });
    }
    Err(PyValueError::new_err("image dtype must be uint8 or uint16"))
}

fn image_shape(shape: &[usize]) -> PyResult<(usize, usize, ColorModel)> {
    match shape {
        [height, width] if *height > 0 && *width > 0 => Ok((*height, *width, ColorModel::Gray)),
        [height, width, channels] if *height > 0 && *width > 0 => {
            let color = match channels {
                1 => ColorModel::Gray,
                2 => ColorModel::GrayAlpha,
                3 => ColorModel::Rgb,
                4 => ColorModel::Rgba,
                _ => {
                    return Err(PyValueError::new_err(
                        "image must have between one and four channels",
                    ));
                }
            };
            Ok((*height, *width, color))
        }
        _ => Err(PyValueError::new_err(
            "image must have HW or HWC shape with positive dimensions",
        )),
    }
}

fn decoded_to_python(py: Python<'_>, image: DecodedImage) -> PyResult<Py<PyAny>> {
    let channels = image.color.channels();
    match image.pixels {
        PixelData::U8(data) if channels == 1 => Ok(data
            .into_pyarray(py)
            .reshape([image.height, image.width])?
            .into_any()
            .unbind()),
        PixelData::U8(data) => Ok(data
            .into_pyarray(py)
            .reshape([image.height, image.width, channels])?
            .into_any()
            .unbind()),
        PixelData::U16(data) if channels == 1 => Ok(data
            .into_pyarray(py)
            .reshape([image.height, image.width])?
            .into_any()
            .unbind()),
        PixelData::U16(data) => Ok(data
            .into_pyarray(py)
            .reshape([image.height, image.width, channels])?
            .into_any()
            .unbind()),
    }
}

fn map_codec_error(error: CodecError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn map_read_error(error: ReadEncodedError, path: &Path) -> PyErr {
    match error {
        ReadEncodedError::Io(error) => map_path_io_error("read source", path, error),
        ReadEncodedError::Limit { actual, limit } => PyValueError::new_err(format!(
            "encoded image has {actual} bytes, exceeding the configured limit of {limit}"
        )),
    }
}

fn map_path_io_error(operation: &str, path: &Path, error: std::io::Error) -> PyErr {
    PyOSError::new_err(format!(
        "failed to {operation} '{}': {error}",
        path.display()
    ))
}

fn map_target_run_error(label: &str, error: PipelineRunError) -> PyErr {
    match error {
        PipelineRunError::Core(CoreError::Invalid(message)) => {
            PyValueError::new_err(format!("{label}: {message}"))
        }
        PipelineRunError::Core(CoreError::Runtime(message)) => {
            PyRuntimeError::new_err(format!("{label}: {message}"))
        }
        PipelineRunError::Codec(error) => PyValueError::new_err(format!("{label}: {error}")),
        PipelineRunError::Workspace => {
            PyRuntimeError::new_err(format!("{label}: workspace pool poisoned"))
        }
        PipelineRunError::DecodedContract(message) => {
            pyo3::exceptions::PyTypeError::new_err(format!("{label}: {message}"))
        }
        PipelineRunError::InvalidTarget(message) => {
            PyValueError::new_err(format!("{label}: {message}"))
        }
        PipelineRunError::Read { path, error } => PyOSError::new_err(format!(
            "{label}: failed to read source '{}': {error}",
            path.display()
        )),
        PipelineRunError::Write { path, error } => PyOSError::new_err(format!(
            "{label}: failed to write destination '{}': {error}",
            path.display()
        )),
        PipelineRunError::EncodedLimit { actual, limit } => PyValueError::new_err(format!(
            "{label}: encoded image has {actual} bytes, exceeding the configured limit of {limit}"
        )),
        PipelineRunError::SinkContract => PyValueError::new_err(format!(
            "{label}: pipeline output does not satisfy the HWC RGB uint8 encoded-sink contract"
        )),
        PipelineRunError::Target {
            label: nested,
            error,
        } => map_target_run_error(&format!("{label}: {nested}"), *error),
    }
}

fn map_pipeline_run_error(error: PipelineRunError) -> PyErr {
    match error {
        PipelineRunError::Core(error) => map_core_error(error),
        PipelineRunError::Codec(error) => map_codec_error(error),
        PipelineRunError::Workspace => PyRuntimeError::new_err("workspace pool poisoned"),
        PipelineRunError::DecodedContract(message) => {
            pyo3::exceptions::PyTypeError::new_err(message)
        }
        PipelineRunError::InvalidTarget(message) => PyValueError::new_err(message),
        PipelineRunError::Read { path, error } => map_path_io_error("read source", &path, error),
        PipelineRunError::Write { path, error } => {
            map_path_io_error("write destination", &path, error)
        }
        PipelineRunError::EncodedLimit { actual, limit } => PyValueError::new_err(format!(
            "encoded image has {actual} bytes, exceeding the configured limit of {limit}"
        )),
        PipelineRunError::SinkContract => PyValueError::new_err(
            "pipeline output does not satisfy the HWC RGB uint8 encoded-sink contract",
        ),
        PipelineRunError::Target { label, error } => map_target_run_error(&label, *error),
    }
}

#[pymodule]
fn _variopinta(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPipeline>()?;
    module.add_function(wrap_pyfunction!(registered_transform_names, module)?)?;
    module.add_function(wrap_pyfunction!(boundary, module)?)?;
    module.add_function(wrap_pyfunction!(read_image, module)?)?;
    module.add_function(wrap_pyfunction!(decode_image, module)?)?;
    module.add_function(wrap_pyfunction!(encode_image, module)?)?;
    module.add_function(wrap_pyfunction!(write_image, module)?)?;
    Ok(())
}
