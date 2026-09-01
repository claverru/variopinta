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
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
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
    seed: u64,
    next_key: AtomicU64,
    workspaces: Mutex<Vec<Workspace>>,
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (specs, seed, mode = "reference"))]
    fn new(specs: &Bound<'_, PyAny>, seed: u64, mode: &str) -> PyResult<Self> {
        Ok(Self {
            core: Compiler::new(parse_mode(mode)?)
                .compile(PipelineSpec::new(parse_specs(specs)?))
                .map_err(map_core_error)?,
            seed,
            next_key: AtomicU64::new(0),
            workspaces: Mutex::new(Vec::new()),
        })
    }

    #[pyo3(signature = (image, key = None))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>,
        key: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let shape = image.shape();
        if shape[2] != 3 || shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err("expected a non-empty HWC RGB input"));
        }
        let data = image
            .as_slice()
            .map_err(|_| PyValueError::new_err("input must be C-contiguous"))?;
        let key = key.unwrap_or_else(|| self.next_key.fetch_add(1, Ordering::Relaxed));
        let mut workspace = self.take_workspace()?;
        let output = self
            .core
            .apply(data, shape[0], shape[1], self.seed, key, &mut workspace)
            .map_err(map_core_error);
        self.return_workspace(workspace)?;
        output_to_python(py, output?)
    }

    fn explain<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        explanation_to_python(py, self.core.explain())
    }
}

impl PyPipeline {
    fn take_workspace(&self) -> PyResult<Workspace> {
        self.workspaces
            .lock()
            .map_err(|_| PyRuntimeError::new_err("workspace pool poisoned"))
            .map(|mut pool| pool.pop().unwrap_or_default())
    }

    fn return_workspace(&self, workspace: Workspace) -> PyResult<()> {
        if workspace.retained_bytes() > MAX_RETAINED_WORKSPACE_BYTES {
            return Ok(());
        }
        let mut pool = self
            .workspaces
            .lock()
            .map_err(|_| PyRuntimeError::new_err("workspace pool poisoned"))?;
        if pool.len() < MAX_CACHED_WORKSPACES {
            pool.push(workspace);
        }
        Ok(())
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

fn explanation_to_python(py: Python<'_>, value: PipelineExplanation) -> PyResult<Py<PyAny>> {
    let to_torch = value.output_layout == "CHW";
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
    output.set_item(
        "output",
        image_contract_to_python(py, value.output, to_torch.then_some("Torch CPU Tensor"))?,
    )?;
    let buffers = value
        .buffers
        .into_iter()
        .map(|buffer| buffer_explanation_to_python(py, buffer))
        .collect::<PyResult<Vec<_>>>()?;
    output.set_item("buffers", PyList::new(py, buffers)?)?;
    let boundary_copy = CopyExplanation {
        stage: "python-entry",
        count: "0-or-1",
        condition: "non-contiguous-input",
        reason: "normalize-to-contiguous-HWC",
    };
    let output_transfer = CopyExplanation {
        stage: "python-output",
        count: "0",
        condition: "always",
        reason: "transfer-Rust-Vec-ownership-to-NumPy-storage",
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
    boundary.set_item("input", "NumPy HWC RGB uint8")?;
    boundary.set_item("input_access", "read-only-borrow")?;
    boundary.set_item(
        "output",
        if to_torch {
            "owned contiguous CPU Torch Tensor"
        } else {
            "owned contiguous NumPy array"
        },
    )?;
    boundary.set_item("gil", "held-during-augmentation")?;
    output.set_item("python_boundary", boundary)?;
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
#[pyo3(signature = (path, mode = "rgb", max_pixels = Some(100_000_000)))]
fn read_image(
    py: Python<'_>,
    path: std::path::PathBuf,
    mode: &str,
    max_pixels: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let options = decode_options(mode, max_pixels)?;
    let encoded = py
        .allow_threads(move || std::fs::read(path))
        .map_err(|error| PyOSError::new_err(error.to_string()))?;
    let image = py
        .allow_threads(move || augment_io::decode_image(&encoded, options))
        .map_err(map_codec_error)?;
    decoded_to_python(py, image)
}

#[pyfunction]
#[pyo3(signature = (data, mode = "rgb", max_pixels = Some(100_000_000)))]
fn decode_image(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    mode: &str,
    max_pixels: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let options = decode_options(mode, max_pixels)?;
    let encoded = data.as_bytes().to_vec();
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
    path: std::path::PathBuf,
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
    py.allow_threads(move || std::fs::write(path, encoded))
        .map_err(|error| PyOSError::new_err(error.to_string()))
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
