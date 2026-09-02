use augment_core::{
    BorderMode, BufferExplanation, CompiledPipeline, Compiler, CopyExplanation, CoreError,
    DropoutSizeRange, ExecutionMode, ImageContractExplanation, Interpolation, PadPosition,
    PipelineExplanation, PipelineOutput, PipelineSpec, PolicyExplanation, TransformExplanation,
    TransformSpec, Workspace, REGISTERED_TRANSFORM_NAMES,
};
use augment_io::{
    CodecError, ColorModel, DecodeMode, DecodeOptions, DecodedImage, EncodeOptions, ImageFormat,
    OwnedImage, PixelData,
};
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyByteArray, PyByteArrayMethods, PyBytes, PyBytesMethods, PyDict, PyList, PyMemoryView,
};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const MAX_CACHED_WORKSPACES: usize = 8;
const MAX_RETAINED_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const EXPLANATION_SCHEMA_VERSION: u8 = 2;

#[pyfunction]
fn registered_transform_names() -> Vec<&'static str> {
    REGISTERED_TRANSFORM_NAMES.to_vec()
}

#[pyclass(name = "Pipeline")]
struct PyPipeline {
    core: CompiledPipeline,
    source: SourceConfig,
    sink: SinkConfig,
    seed: u64,
    next_key: AtomicU64,
    workspaces: Mutex<Vec<Workspace>>,
}

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
enum SinkConfig {
    Return,
    Encoded {
        format: ImageFormat,
        options: EncodeOptions,
    },
    Path {
        format: ImageFormat,
        options: EncodeOptions,
    },
}

enum NativeOutput {
    Return(PipelineOutput),
    Encoded(Vec<u8>),
    Written,
}

enum PipelineRunError {
    Core(CoreError),
    Codec(CodecError),
    Workspace,
    DecodedContract(&'static str),
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
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (
        specs,
        seed,
        mode = "reference",
        source = "array",
        max_pixels = None,
        max_encoded_bytes = None,
        sink = "return",
        format = None,
        quality = None,
        compression = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        specs: &Bound<'_, PyAny>,
        seed: u64,
        mode: &str,
        source: &str,
        max_pixels: Option<usize>,
        max_encoded_bytes: Option<usize>,
        sink: &str,
        format: Option<&str>,
        quality: Option<u8>,
        compression: Option<u8>,
    ) -> PyResult<Self> {
        let source = parse_source_config(source, max_pixels, max_encoded_bytes)?;
        let sink = parse_sink_config(sink, format, quality, compression)?;
        let core = Compiler::new(parse_mode(mode)?)
            .compile(PipelineSpec::new(parse_specs(specs)?))
            .map_err(map_core_error)?;
        let explanation = core.explain();
        if !matches!(sink, SinkConfig::Return)
            && (explanation.output_dtype != "uint8" || explanation.output_layout != "HWC")
        {
            return Err(PyValueError::new_err(
                "encoded pipeline output requires an always-HWC RGB uint8 result",
            ));
        }
        Ok(Self {
            core,
            source,
            sink,
            seed,
            next_key: AtomicU64::new(0),
            workspaces: Mutex::new(Vec::new()),
        })
    }

    #[pyo3(signature = (image, key = None, destination = None))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>,
        key: Option<u64>,
        destination: Option<PathBuf>,
    ) -> PyResult<Py<PyAny>> {
        if !matches!(self.source, SourceConfig::Array) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "configured pipeline input is not ArrayInput",
            ));
        }
        let destination = self.validate_destination(destination)?;
        let shape = image.shape();
        if shape[2] != 3 || shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err("expected a non-empty HWC RGB input"));
        }
        let data = image
            .as_slice()
            .map_err(|_| PyValueError::new_err("input must be C-contiguous"))?;
        let output = self
            .run_core(data, shape[0], shape[1], key)
            .map_err(map_pipeline_run_error)?;
        let output = if matches!(self.sink, SinkConfig::Return) {
            self.deliver(output, destination)
        } else {
            py.allow_threads(move || self.deliver(output, destination))
        }
        .map_err(map_pipeline_run_error)?;
        native_output_to_python(py, output)
    }

    #[pyo3(signature = (data, key = None, destination = None))]
    fn apply_encoded<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        key: Option<u64>,
        destination: Option<PathBuf>,
    ) -> PyResult<Py<PyAny>> {
        let SourceConfig::Encoded {
            max_pixels,
            max_encoded_bytes,
        } = self.source
        else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "configured pipeline input is not EncodedInput",
            ));
        };
        let destination = self.validate_destination(destination)?;
        let encoded = snapshot_encoded(py, data, max_encoded_bytes)?;
        let output = py
            .allow_threads(move || self.run_owned_encoded(encoded, max_pixels, key, destination))
            .map_err(map_pipeline_run_error)?;
        native_output_to_python(py, output)
    }

    #[pyo3(signature = (path, key = None, destination = None))]
    fn apply_path<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
        key: Option<u64>,
        destination: Option<PathBuf>,
    ) -> PyResult<Py<PyAny>> {
        let SourceConfig::Path {
            max_pixels,
            max_encoded_bytes,
        } = self.source
        else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "configured pipeline input is not PathInput",
            ));
        };
        let destination = self.validate_destination(destination)?;
        let output = py
            .allow_threads(move || {
                let encoded =
                    read_encoded(&path, max_encoded_bytes).map_err(|error| match error {
                        ReadEncodedError::Io(error) => PipelineRunError::Read {
                            path: path.clone(),
                            error,
                        },
                        ReadEncodedError::Limit { actual, limit } => {
                            PipelineRunError::EncodedLimit { actual, limit }
                        }
                    })?;
                self.run_owned_encoded(encoded, max_pixels, key, destination)
            })
            .map_err(map_pipeline_run_error)?;
        native_output_to_python(py, output)
    }

    fn explain<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        explanation_to_python(py, self.core.explain(), self.source, self.sink)
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

    fn run_core(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        key: Option<u64>,
    ) -> Result<PipelineOutput, PipelineRunError> {
        let key = key.unwrap_or_else(|| self.next_key.fetch_add(1, Ordering::Relaxed));
        let mut workspace = self.take_workspace()?;
        let output = self
            .core
            .apply(data, height, width, self.seed, key, &mut workspace)
            .map_err(PipelineRunError::Core);
        self.return_workspace(workspace)?;
        output
    }

    fn run_owned_encoded(
        &self,
        encoded: Vec<u8>,
        max_pixels: Option<usize>,
        key: Option<u64>,
        destination: Option<PathBuf>,
    ) -> Result<NativeOutput, PipelineRunError> {
        let decoded = augment_io::decode_image(
            &encoded,
            DecodeOptions {
                mode: DecodeMode::Rgb,
                max_pixels,
            },
        )
        .map_err(PipelineRunError::Codec)?;
        let (pixels, height, width) = decoded_rgb_u8(decoded)?;
        let output = self.run_core(&pixels, height, width, key)?;
        self.deliver(output, destination)
    }

    fn deliver(
        &self,
        output: PipelineOutput,
        destination: Option<PathBuf>,
    ) -> Result<NativeOutput, PipelineRunError> {
        match self.sink {
            SinkConfig::Return => Ok(NativeOutput::Return(output)),
            SinkConfig::Encoded { options, .. } => {
                let image = pipeline_output_to_owned(output)?;
                augment_io::encode_image(&image, options)
                    .map(NativeOutput::Encoded)
                    .map_err(PipelineRunError::Codec)
            }
            SinkConfig::Path { options, .. } => {
                let path = destination.ok_or(PipelineRunError::SinkContract)?;
                let image = pipeline_output_to_owned(output)?;
                let encoded =
                    augment_io::encode_image(&image, options).map_err(PipelineRunError::Codec)?;
                std::fs::write(&path, encoded)
                    .map_err(|error| PipelineRunError::Write { path, error })?;
                Ok(NativeOutput::Written)
            }
        }
    }

    fn validate_destination(&self, destination: Option<PathBuf>) -> PyResult<Option<PathBuf>> {
        match (self.sink, destination) {
            (SinkConfig::Path { format, .. }, Some(path)) => {
                if path_format(&path).is_some_and(|actual| actual != format) {
                    return Err(PyValueError::new_err(
                        "output format conflicts with the destination extension",
                    ));
                }
                Ok(Some(path))
            }
            (SinkConfig::Path { .. }, None) => Err(pyo3::exceptions::PyTypeError::new_err(
                "destination is required for PathOutput",
            )),
            (_, Some(_)) => Err(pyo3::exceptions::PyTypeError::new_err(
                "destination is only valid for PathOutput",
            )),
            (_, None) => Ok(None),
        }
    }
}

fn parse_source_config(
    source: &str,
    max_pixels: Option<usize>,
    max_encoded_bytes: Option<usize>,
) -> PyResult<SourceConfig> {
    if max_pixels == Some(0) {
        return Err(PyValueError::new_err("max_pixels must be positive or None"));
    }
    if max_encoded_bytes == Some(0) {
        return Err(PyValueError::new_err(
            "max_encoded_bytes must be positive or None",
        ));
    }
    match source {
        "array" if max_pixels.is_none() && max_encoded_bytes.is_none() => Ok(SourceConfig::Array),
        "encoded" => Ok(SourceConfig::Encoded {
            max_pixels,
            max_encoded_bytes,
        }),
        "path" => Ok(SourceConfig::Path {
            max_pixels,
            max_encoded_bytes,
        }),
        "array" => Err(PyValueError::new_err(
            "ArrayInput does not accept decode limits",
        )),
        _ => Err(PyValueError::new_err(
            "source must be 'array', 'encoded', or 'path'",
        )),
    }
}

fn parse_sink_config(
    sink: &str,
    format: Option<&str>,
    quality: Option<u8>,
    compression: Option<u8>,
) -> PyResult<SinkConfig> {
    if sink == "return" {
        if format.is_some() || quality.is_some() || compression.is_some() {
            return Err(PyValueError::new_err(
                "ReturnOutput does not accept codec options",
            ));
        }
        return Ok(SinkConfig::Return);
    }
    let format = format.ok_or_else(|| PyValueError::new_err("encoded sinks require a format"))?;
    let image_format = parse_image_format(format)?;
    let options = encode_options(format, quality, compression)?;
    match sink {
        "encoded" => Ok(SinkConfig::Encoded {
            format: image_format,
            options,
        }),
        "path" => Ok(SinkConfig::Path {
            format: image_format,
            options,
        }),
        _ => Err(PyValueError::new_err(
            "sink must be 'return', 'encoded', or 'path'",
        )),
    }
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
        "ToTorch" => Ok(TransformSpec::ToTorch),
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

fn native_output_to_python(py: Python<'_>, output: NativeOutput) -> PyResult<Py<PyAny>> {
    match output {
        NativeOutput::Return(output) => output_to_python(py, output),
        NativeOutput::Encoded(encoded) => Ok(PyBytes::new(py, &encoded).into_any().unbind()),
        NativeOutput::Written => Ok(py.None()),
    }
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

fn pipeline_output_to_owned(output: PipelineOutput) -> Result<OwnedImage, PipelineRunError> {
    match output {
        PipelineOutput::U8Hwc {
            data,
            height,
            width,
        } => Ok(OwnedImage {
            pixels: PixelData::U8(data),
            height,
            width,
            color: ColorModel::Rgb,
        }),
        _ => Err(PipelineRunError::SinkContract),
    }
}

fn explanation_to_python(
    py: Python<'_>,
    value: PipelineExplanation,
    source: SourceConfig,
    sink: SinkConfig,
) -> PyResult<Py<PyAny>> {
    let to_torch = matches!(sink, SinkConfig::Return) && value.output_layout == "CHW";
    let output = PyDict::new(py);
    output.set_item("schema_version", EXPLANATION_SCHEMA_VERSION)?;
    output.set_item("mode", value.mode)?;
    output.set_item("sampling", value.sampling)?;
    output.set_item("transforms", value.transforms)?;
    let steps = value
        .steps
        .into_iter()
        .map(|step| transform_explanation_to_python(py, step))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("steps", PyList::new(py, steps)?)?;
    output.set_item("fusions", value.fusions)?;
    output.set_item("unit_specializations", value.unit_specializations)?;
    output.set_item("optimizations", value.optimizations)?;
    output.set_item("passes", value.passes)?;
    output.set_item("pixel_passes", value.pixel_passes)?;
    output.set_item("output_dtype", value.output_dtype)?;
    output.set_item("output_layout", value.output_layout)?;
    output.set_item("input", image_contract_to_python(py, value.input, None)?)?;
    let output_container = if matches!(sink, SinkConfig::Return) {
        to_torch.then_some("Torch CPU Tensor")
    } else {
        None
    };
    output.set_item(
        "output",
        image_contract_to_python(py, value.output, output_container)?,
    )?;
    output.set_item("source", source_explanation_to_python(py, source)?)?;
    output.set_item("sink", sink_explanation_to_python(py, sink)?)?;
    let mut buffers = Vec::new();
    match source {
        SourceConfig::Array => {}
        SourceConfig::Encoded { .. } => {
            buffers.push(buffer_record(
                py,
                "encoded-input",
                "uint8",
                "encoded",
                "call-owned-snapshot",
                "always",
            )?);
            buffers.push(buffer_record(
                py,
                "decoded-rgb",
                "uint8",
                "HWC",
                "call-owned-source",
                "always",
            )?);
        }
        SourceConfig::Path { .. } => {
            buffers.push(buffer_record(
                py,
                "encoded-input",
                "uint8",
                "encoded",
                "call-owned-file-read",
                "always",
            )?);
            buffers.push(buffer_record(
                py,
                "decoded-rgb",
                "uint8",
                "HWC",
                "call-owned-source",
                "always",
            )?);
        }
    }
    buffers.extend(
        value
            .buffers
            .into_iter()
            .map(|buffer| buffer_explanation_to_python(py, buffer))
            .collect::<PyResult<Vec<_>>>()?,
    );
    if !matches!(sink, SinkConfig::Return) {
        buffers.push(buffer_record(
            py,
            "encoded-output",
            "uint8",
            "encoded",
            "call-owned",
            "always",
        )?);
    }
    output.set_item("buffers", PyList::new(py, buffers)?)?;
    let boundary_copy = match source {
        SourceConfig::Array => CopyExplanation {
            stage: "python-entry",
            count: "0-or-1",
            condition: "non-contiguous-input",
            reason: "normalize-to-contiguous-HWC",
        },
        SourceConfig::Encoded { .. } => CopyExplanation {
            stage: "python-entry",
            count: "1",
            condition: "always",
            reason: "snapshot-compressed-input-into-Rust-owned-storage",
        },
        SourceConfig::Path { .. } => CopyExplanation {
            stage: "python-entry",
            count: "0",
            condition: "always",
            reason: "no-Python-array-entry-normalization",
        },
    };
    let output_transfer = match sink {
        SinkConfig::Return => CopyExplanation {
            stage: "python-output",
            count: "0",
            condition: "always",
            reason: "transfer-Rust-Vec-ownership-to-NumPy-storage",
        },
        SinkConfig::Encoded { .. } => CopyExplanation {
            stage: "python-output",
            count: "1",
            condition: "always",
            reason: "copy-compressed-buffer-into-Python-bytes",
        },
        SinkConfig::Path { .. } => CopyExplanation {
            stage: "python-output",
            count: "0",
            condition: "always",
            reason: "no-Python-output-transfer",
        },
    };
    let copies = std::iter::once(boundary_copy)
        .chain(value.copies)
        .chain(std::iter::once(output_transfer))
        .chain(to_torch.then_some(CopyExplanation {
            stage: "torch-adapter",
            count: "0",
            condition: "always",
            reason: "share-NumPy-storage-with-Torch",
        }))
        .map(|copy| copy_explanation_to_python(py, copy))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("copies", PyList::new(py, copies)?)?;
    output.set_item("fallbacks", value.fallbacks)?;
    let boundary = PyDict::new(py);
    boundary.set_item("crossings_per_call", 1)?;
    boundary.set_item(
        "input",
        match source {
            SourceConfig::Array => "NumPy HWC RGB uint8",
            SourceConfig::Encoded { .. } => "JPEG or static PNG encoded buffer",
            SourceConfig::Path { .. } => "local JPEG or static PNG path",
        },
    )?;
    boundary.set_item(
        "input_access",
        match source {
            SourceConfig::Array => "read-only-borrow",
            SourceConfig::Encoded { .. } => "Rust-owned-snapshot",
            SourceConfig::Path { .. } => "Rust-owned-file-read",
        },
    )?;
    boundary.set_item(
        "output",
        match sink {
            SinkConfig::Return if to_torch => "owned contiguous CPU Torch Tensor",
            SinkConfig::Return => "owned contiguous NumPy array",
            SinkConfig::Encoded { .. } => "Python bytes",
            SinkConfig::Path { .. } => "no Python output",
        },
    )?;
    boundary.set_item(
        "gil",
        match (source, sink) {
            (SourceConfig::Array, SinkConfig::Return) => "held-during-augmentation",
            (SourceConfig::Array, _) => "held-during-augmentation-released-during-delivery",
            (SourceConfig::Encoded { .. } | SourceConfig::Path { .. }, SinkConfig::Return) => {
                "released-during-decode-and-augmentation"
            }
            (SourceConfig::Encoded { .. } | SourceConfig::Path { .. }, _) => {
                "released-during-decode-augmentation-and-delivery"
            }
        },
    )?;
    let gil_stages = PyDict::new(py);
    gil_stages.set_item("arguments", "held")?;
    gil_stages.set_item(
        "source",
        match source {
            SourceConfig::Array | SourceConfig::Encoded { .. } => "held",
            SourceConfig::Path { .. } => "released",
        },
    )?;
    gil_stages.set_item(
        "decode",
        if matches!(source, SourceConfig::Array) {
            "not-applicable"
        } else {
            "released"
        },
    )?;
    gil_stages.set_item(
        "augmentation",
        if matches!(source, SourceConfig::Array) {
            "held"
        } else {
            "released"
        },
    )?;
    gil_stages.set_item(
        "delivery",
        match sink {
            SinkConfig::Return => "held-for-Python-output-transfer",
            SinkConfig::Encoded { .. } => "released-for-encode-held-for-bytes-construction",
            SinkConfig::Path { .. } => "released",
        },
    )?;
    boundary.set_item("source_acquisition", gil_stages.get_item("source")?)?;
    boundary.set_item("decode", gil_stages.get_item("decode")?)?;
    boundary.set_item("augmentation", gil_stages.get_item("augmentation")?)?;
    boundary.set_item("delivery", gil_stages.get_item("delivery")?)?;
    boundary.set_item("gil_stages", gil_stages)?;
    output.set_item("python_boundary", boundary)?;
    Ok(output.into_any().unbind())
}

fn source_explanation_to_python(py: Python<'_>, source: SourceConfig) -> PyResult<Py<PyAny>> {
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
            output.set_item("mode", "rgb")?;
            output.set_item("formats", ["jpeg", "png"])?;
            output.set_item("max_pixels", max_pixels)?;
            output.set_item("max_encoded_bytes", max_encoded_bytes)?;
        }
    }
    Ok(output.into_any().unbind())
}

fn sink_explanation_to_python(py: Python<'_>, sink: SinkConfig) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    match sink {
        SinkConfig::Return => output.set_item("type", "return")?,
        SinkConfig::Encoded { format, options } | SinkConfig::Path { format, options } => {
            output.set_item(
                "type",
                if matches!(sink, SinkConfig::Encoded { .. }) {
                    "encoded"
                } else {
                    "path"
                },
            )?;
            output.set_item("format", image_format_name(format))?;
            match options {
                EncodeOptions::Jpeg { quality } => output.set_item("quality", quality)?,
                EncodeOptions::Png { compression } => {
                    output.set_item("compression", compression)?
                }
            }
        }
    }
    Ok(output.into_any().unbind())
}

fn buffer_record(
    py: Python<'_>,
    name: &str,
    dtype: &str,
    layout: &str,
    lifecycle: &str,
    condition: &str,
) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("name", name)?;
    output.set_item("dtype", dtype)?;
    output.set_item("layout", layout)?;
    output.set_item("lifecycle", lifecycle)?;
    output.set_item("condition", condition)?;
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
    let encoded = snapshot_encoded(py, data, max_encoded_bytes)?;
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
) -> PyResult<Vec<u8>> {
    validate_encoded_limit(max_encoded_bytes)?;
    if let Ok(bytes) = data.downcast::<PyBytes>() {
        check_encoded_size(bytes.as_bytes().len(), max_encoded_bytes)?;
        return Ok(bytes.as_bytes().to_vec());
    }
    if let Ok(bytearray) = data.downcast::<PyByteArray>() {
        check_encoded_size(bytearray.len(), max_encoded_bytes)?;
        return Ok(bytearray.to_vec());
    }
    if data.downcast::<PyMemoryView>().is_ok() {
        let length: usize = data.getattr("nbytes")?.extract()?;
        check_encoded_size(length, max_encoded_bytes)?;
        let snapshot = data.call_method0("tobytes")?;
        let bytes = snapshot.downcast::<PyBytes>()?;
        return Ok(bytes.as_bytes().to_vec());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "encoded data must be bytes, bytearray, or memoryview",
    ))
}

fn check_encoded_size(length: usize, limit: Option<usize>) -> PyResult<()> {
    if let Some(limit) = limit.filter(|limit| length > *limit) {
        return Err(PyValueError::new_err(format!(
            "encoded image has {length} bytes, exceeding the configured limit of {limit}"
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

fn map_pipeline_run_error(error: PipelineRunError) -> PyErr {
    match error {
        PipelineRunError::Core(error) => map_core_error(error),
        PipelineRunError::Codec(error) => map_codec_error(error),
        PipelineRunError::Workspace => PyRuntimeError::new_err("workspace pool poisoned"),
        PipelineRunError::DecodedContract(message) => {
            pyo3::exceptions::PyTypeError::new_err(message)
        }
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
