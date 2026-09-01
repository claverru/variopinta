use crate::{
    BorderMode, CoreError, CoreResult, DropoutSizeRange, Interpolation, PadPosition,
    PolicyExplanation, TransformExplanation, TransformSpec,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

mod explanation;
mod sampling;
mod validation;

#[derive(Debug, Clone)]
pub(crate) enum TransformPlan {
    Resize {
        height: usize,
        width: usize,
        interpolation: Interpolation,
        antialias: bool,
        p: f32,
    },
    RandomCrop {
        height: usize,
        width: usize,
        p: f32,
    },
    RandomResizedCrop {
        height: usize,
        width: usize,
        scale: [f32; 2],
        ratio: [f32; 2],
        interpolation: Interpolation,
        antialias: bool,
        p: f32,
    },
    HorizontalFlip {
        p: f32,
    },
    VerticalFlip {
        p: f32,
    },
    CenterCrop {
        height: usize,
        width: usize,
        p: f32,
    },
    PadIfNeeded {
        min_height: Option<usize>,
        min_width: Option<usize>,
        pad_height_divisor: Option<usize>,
        pad_width_divisor: Option<usize>,
        position: PadPosition,
        border_mode: BorderMode,
        fill: [u8; 3],
        p: f32,
    },
    CoarseDropout {
        num_holes_range: [usize; 2],
        hole_height_range: DropoutSizeRange,
        hole_width_range: DropoutSizeRange,
        fill: [u8; 3],
        p: f32,
    },
    ColorJitter {
        brightness: [f32; 2],
        contrast: [f32; 2],
        saturation: [f32; 2],
        hue: [f32; 2],
        p: f32,
    },
    Affine {
        degrees: [f32; 2],
        translate: [f32; 2],
        scale: [f32; 2],
        shear: [f32; 4],
        interpolation: Interpolation,
        border_mode: BorderMode,
        fill: [u8; 3],
        p: f32,
    },
    RandomRotation {
        degrees: [f32; 2],
        interpolation: Interpolation,
        border_mode: BorderMode,
        fill: [u8; 3],
        p: f32,
    },
    GaussianNoise {
        mean: [f32; 2],
        std: [f32; 2],
        per_channel: bool,
        p: f32,
    },
    Sharpen {
        alpha: [f32; 2],
        lightness: [f32; 2],
        p: f32,
    },
    Perspective {
        scale: [f32; 2],
        interpolation: Interpolation,
        border_mode: BorderMode,
        fill: [u8; 3],
        p: f32,
    },
    GridDistortion {
        num_steps: usize,
        distort_limit: [f32; 2],
        interpolation: Interpolation,
        border_mode: BorderMode,
        fill: [u8; 3],
        p: f32,
    },
    GaussianBlur {
        kernel_size: usize,
        sigma: [f32; 2],
        fixed_kernel: Option<Vec<u16>>,
        p: f32,
    },
    Grayscale {
        p: f32,
    },
    Invert {
        p: f32,
    },
    Solarize {
        threshold: u8,
        p: f32,
    },
    Posterize {
        bits: u8,
        p: f32,
    },
    Normalize {
        mean: [f32; 3],
        std: [f32; 3],
        max_pixel_value: f32,
        p: f32,
    },
    ToTorch,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CropSample {
    pub top: usize,
    pub left: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PadSample {
    pub top: usize,
    pub left: usize,
    pub height: usize,
    pub width: usize,
    pub border_mode: BorderMode,
    pub fill: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DropoutHole {
    pub top: usize,
    pub left: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ColorJitterSample {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
    pub hue_enabled: bool,
    pub order: [u8; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AffineSample {
    pub degrees: f32,
    pub translate: [f32; 2],
    pub scale: f32,
    pub shear: [f32; 2],
    pub interpolation: Interpolation,
    pub border_mode: BorderMode,
    pub fill: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RotationSample {
    pub degrees: f32,
    pub interpolation: Interpolation,
    pub border_mode: BorderMode,
    pub fill: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GaussianNoiseSample {
    pub mean: f32,
    pub std: f32,
    pub seed: u64,
    pub per_channel: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SharpenSample {
    pub alpha: f32,
    pub lightness: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PerspectiveSample {
    pub inverse: [f32; 9],
    pub interpolation: Interpolation,
    pub border_mode: BorderMode,
    pub fill: [u8; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GridDistortionSample {
    pub x_map: Vec<f32>,
    pub y_map: Vec<f32>,
    pub interpolation: Interpolation,
    pub border_mode: BorderMode,
    pub fill: [u8; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SampledTransform {
    Skip,
    Resize {
        height: usize,
        width: usize,
        interpolation: Interpolation,
        antialias: bool,
    },
    RandomCrop(CropSample),
    RandomResizedCrop {
        crop: CropSample,
        height: usize,
        width: usize,
        interpolation: Interpolation,
        antialias: bool,
    },
    HorizontalFlip,
    VerticalFlip,
    CenterCrop(CropSample),
    PadIfNeeded(PadSample),
    CoarseDropout {
        holes: Vec<DropoutHole>,
        fill: [u8; 3],
    },
    ColorJitter(ColorJitterSample),
    Affine(AffineSample),
    RandomRotation(RotationSample),
    GaussianNoise(GaussianNoiseSample),
    Sharpen(SharpenSample),
    Perspective(PerspectiveSample),
    GridDistortion(GridDistortionSample),
    GaussianBlur {
        sigma: f32,
    },
    Grayscale,
    Invert,
    Solarize {
        threshold: u8,
    },
    Posterize {
        bits: u8,
    },
    Normalize,
    ToTorch,
}

impl TransformPlan {
    pub(crate) fn name(&self) -> &'static str {
        self.tag().name()
    }
}

fn sample_resized_crop(
    height: usize,
    width: usize,
    scale: [f32; 2],
    ratio: [f32; 2],
    rng: &mut SmallRng,
) -> CropSample {
    let area = (height as f64) * (width as f64);
    let log_ratio = [(ratio[0] as f64).ln(), (ratio[1] as f64).ln()];
    for _ in 0..10 {
        let target_area = area * rng.random_range(scale[0] as f64..=scale[1] as f64);
        let aspect_ratio = rng.random_range(log_ratio[0]..=log_ratio[1]).exp();
        let crop_width = (target_area * aspect_ratio).sqrt().round() as usize;
        let crop_height = (target_area / aspect_ratio).sqrt().round() as usize;
        if crop_height > 0 && crop_height <= height && crop_width > 0 && crop_width <= width {
            return CropSample {
                top: rng.random_range(0..=height - crop_height),
                left: rng.random_range(0..=width - crop_width),
                height: crop_height,
                width: crop_width,
            };
        }
    }

    let input_ratio = width as f64 / height as f64;
    let (crop_height, crop_width) = if input_ratio < ratio[0] as f64 {
        (
            ((width as f64 / ratio[0] as f64).round() as usize).clamp(1, height),
            width,
        )
    } else if input_ratio > ratio[1] as f64 {
        (
            height,
            ((height as f64 * ratio[1] as f64).round() as usize).clamp(1, width),
        )
    } else {
        (height, width)
    };
    CropSample {
        top: (height - crop_height) / 2,
        left: (width - crop_width) / 2,
        height: crop_height,
        width: crop_width,
    }
}

fn sample_perspective(
    height: usize,
    width: usize,
    scale: [f32; 2],
    rng: &mut SmallRng,
) -> [f32; 9] {
    if height <= 1 || width <= 1 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let right = (width - 1) as f64;
    let bottom = (height - 1) as f64;
    let source = [[0.0, 0.0], [right, 0.0], [right, bottom], [0.0, bottom]];
    for _ in 0..10 {
        let amount = f64::from(sample_uniform(scale, rng));
        let destination = [
            [
                rng.random_range(0.0..=amount) * right,
                rng.random_range(0.0..=amount) * bottom,
            ],
            [
                right - rng.random_range(0.0..=amount) * right,
                rng.random_range(0.0..=amount) * bottom,
            ],
            [
                right - rng.random_range(0.0..=amount) * right,
                bottom - rng.random_range(0.0..=amount) * bottom,
            ],
            [
                rng.random_range(0.0..=amount) * right,
                bottom - rng.random_range(0.0..=amount) * bottom,
            ],
        ];
        if let Some(matrix) = solve_homography(destination, source) {
            return matrix;
        }
    }
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

fn solve_homography(from: [[f64; 2]; 4], to: [[f64; 2]; 4]) -> Option<[f32; 9]> {
    let mut system = [[0.0f64; 9]; 8];
    for index in 0..4 {
        let [x, y] = from[index];
        let [u, v] = to[index];
        system[index * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        system[index * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for column in 0..8 {
        let pivot = (column..8).max_by(|left, right| {
            system[*left][column]
                .abs()
                .total_cmp(&system[*right][column].abs())
        })?;
        if system[pivot][column].abs() < 1e-10 {
            return None;
        }
        system.swap(column, pivot);
        let divisor = system[column][column];
        for value in column..9 {
            system[column][value] /= divisor;
        }
        for row in 0..8 {
            if row == column {
                continue;
            }
            let factor = system[row][column];
            for value in column..9 {
                system[row][value] -= factor * system[column][value];
            }
        }
    }
    let mut result = [0.0f32; 9];
    for index in 0..8 {
        result[index] = system[index][8] as f32;
    }
    result[8] = 1.0;
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn sample_grid_map(
    length: usize,
    requested_steps: usize,
    limit: [f32; 2],
    rng: &mut SmallRng,
) -> CoreResult<Vec<f32>> {
    let mut map = Vec::new();
    map.try_reserve_exact(length)
        .map_err(|_| CoreError::Runtime("grid map allocation failed".into()))?;
    if length <= 1 {
        map.push(0.0);
        return Ok(map);
    }
    if limit == [0.0, 0.0] {
        map.extend((0..length).map(|coordinate| coordinate as f32));
        return Ok(map);
    }
    let steps = requested_steps.min(length - 1);
    let mut cumulative = Vec::new();
    cumulative
        .try_reserve_exact(steps + 1)
        .map_err(|_| CoreError::Runtime("grid control allocation failed".into()))?;
    cumulative.push(0.0f32);
    for _ in 0..steps {
        let weight = 1.0 + sample_uniform(limit, rng);
        cumulative.push(cumulative.last().copied().unwrap_or(0.0) + weight);
    }
    let total = cumulative[steps];
    for coordinate in 0..length {
        let position = coordinate as f32 * steps as f32 / (length - 1) as f32;
        let segment = (position.floor() as usize).min(steps - 1);
        let fraction = position - segment as f32;
        let mapped =
            cumulative[segment] + fraction * (cumulative[segment + 1] - cumulative[segment]);
        map.push(mapped / total * (length - 1) as f32);
    }
    if let Some(last) = map.last_mut() {
        *last = (length - 1) as f32;
    }
    Ok(map)
}

fn interpolation_name(value: Interpolation) -> &'static str {
    match value {
        Interpolation::Nearest => "nearest",
        Interpolation::Bilinear => "bilinear",
    }
}

fn border_name(value: BorderMode) -> &'static str {
    match value {
        BorderMode::Constant => "constant",
        BorderMode::Reflect101 => "reflect101",
    }
}

fn pad_position_name(value: PadPosition) -> &'static str {
    match value {
        PadPosition::Center => "center",
        PadPosition::TopLeft => "top_left",
        PadPosition::TopRight => "top_right",
        PadPosition::BottomLeft => "bottom_left",
        PadPosition::BottomRight => "bottom_right",
        PadPosition::Random => "random",
    }
}

fn pad_axis_policy(minimum: Option<usize>, divisor: Option<usize>) -> String {
    match (minimum, divisor) {
        (Some(value), None) => format!("minimum-{value}"),
        (None, Some(value)) => format!("multiple-of-{value}"),
        _ => "invalid".into(),
    }
}

fn dropout_size_policy(range: DropoutSizeRange) -> String {
    match range {
        DropoutSizeRange::Fraction(values) => {
            format!("fraction-[{},{}]", values[0], values[1])
        }
        DropoutSizeRange::Pixels(values) => format!("pixels-[{},{}]", values[0], values[1]),
    }
}

fn validate_positive_usize_range(name: &str, values: [usize; 2]) -> CoreResult<()> {
    if values[0] == 0 || values[0] > values[1] {
        return Err(CoreError::Invalid(format!(
            "{name} must be an ordered positive range"
        )));
    }
    Ok(())
}

fn validate_dropout_size_range(name: &str, range: DropoutSizeRange) -> CoreResult<()> {
    match range {
        DropoutSizeRange::Fraction(values) => validate_positive_range(name, values, Some(1.0)),
        DropoutSizeRange::Pixels(values) => validate_positive_usize_range(name, values),
    }
}

fn sample_dropout_dimension(
    range: DropoutSizeRange,
    dimension: usize,
    rng: &mut SmallRng,
) -> usize {
    match range {
        DropoutSizeRange::Fraction(values) => {
            ((dimension as f32 * sample_uniform(values, rng)).floor() as usize).clamp(1, dimension)
        }
        DropoutSizeRange::Pixels(values) => {
            let minimum = values[0].min(dimension);
            let maximum = values[1].min(dimension);
            rng.random_range(minimum..=maximum)
        }
    }
}

fn validate_pad_axis(name: &str, minimum: Option<usize>, divisor: Option<usize>) -> CoreResult<()> {
    match (minimum, divisor) {
        (Some(value), None) | (None, Some(value)) if value > 0 => Ok(()),
        _ => Err(CoreError::Invalid(format!(
            "PadIfNeeded {name} requires exactly one positive minimum or divisor"
        ))),
    }
}

fn padded_dimension(
    current: usize,
    minimum: Option<usize>,
    divisor: Option<usize>,
) -> CoreResult<usize> {
    if let Some(minimum) = minimum {
        return Ok(current.max(minimum));
    }
    let divisor = divisor.ok_or_else(|| CoreError::Invalid("missing pad divisor".into()))?;
    current
        .checked_add(divisor - 1)
        .map(|value| value / divisor * divisor)
        .ok_or_else(|| CoreError::Invalid("padded dimensions overflow".into()))
}

fn sample_pad_origin(
    position: PadPosition,
    extra_height: usize,
    extra_width: usize,
    rng: &mut SmallRng,
) -> (usize, usize) {
    match position {
        PadPosition::Center => (extra_height / 2, extra_width / 2),
        PadPosition::TopLeft => (0, 0),
        PadPosition::TopRight => (0, extra_width),
        PadPosition::BottomLeft => (extra_height, 0),
        PadPosition::BottomRight => (extra_height, extra_width),
        PadPosition::Random => (
            if extra_height == 0 {
                0
            } else {
                rng.random_range(0..=extra_height)
            },
            if extra_width == 0 {
                0
            } else {
                rng.random_range(0..=extra_width)
            },
        ),
    }
}

fn validate_dimensions(height: usize, width: usize) -> CoreResult<()> {
    if height == 0 || width == 0 {
        return Err(CoreError::Invalid("dimensions must be positive".into()));
    }
    u32::try_from(height)
        .and_then(|_| u32::try_from(width))
        .map_err(|_| CoreError::Invalid("dimensions exceed the native backend limit".into()))?;
    height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| CoreError::Invalid("image dimensions overflow".into()))?;
    Ok(())
}

fn validate_positive_range(name: &str, values: [f32; 2], maximum: Option<f32>) -> CoreResult<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || values[0] > values[1]
        || maximum.is_some_and(|maximum| values[1] > maximum)
    {
        return Err(CoreError::Invalid(format!(
            "{name} must be an ordered finite positive range{}",
            maximum.map_or(String::new(), |maximum| format!(" at most {maximum}"))
        )));
    }
    Ok(())
}

fn validate_non_negative_range(name: &str, values: [f32; 2]) -> CoreResult<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || values[0] > values[1]
    {
        return Err(CoreError::Invalid(format!(
            "{name} must be an ordered finite non-negative range"
        )));
    }
    Ok(())
}

fn validate_finite_range(name: &str, values: [f32; 2]) -> CoreResult<()> {
    if values.iter().any(|value| !value.is_finite()) || values[0] > values[1] {
        return Err(CoreError::Invalid(format!(
            "{name} must be an ordered finite range"
        )));
    }
    Ok(())
}

fn validate_probability(p: f32) -> CoreResult<()> {
    if p.is_finite() && (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(CoreError::Invalid("probability must be in [0, 1]".into()))
    }
}

fn sample_uniform(range: [f32; 2], rng: &mut SmallRng) -> f32 {
    if range[0] == range[1] {
        range[0]
    } else {
        rng.random_range(range[0]..=range[1])
    }
}

fn sample_symmetric(maximum: f32, rng: &mut SmallRng) -> f32 {
    sample_uniform([-maximum, maximum], rng)
}

fn should_apply(p: f32, rng: &mut SmallRng) -> bool {
    p == 1.0 || (p != 0.0 && rng.random::<f32>() < p)
}

pub(crate) fn derive_run_seed(pipeline_seed: u64, run_key: u64) -> u64 {
    let mut value = pipeline_seed ^ run_key.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

pub(crate) fn make_gaussian_kernel(kernel_size: usize, sigma: f32) -> CoreResult<Vec<u16>> {
    if kernel_size == 0 || kernel_size % 2 == 0 || !sigma.is_finite() || sigma <= 0.0 {
        return Err(CoreError::Invalid(
            "GaussianBlur requires an odd positive kernel_size and finite sigma > 0".into(),
        ));
    }
    let radius = kernel_size / 2;
    let mut float_kernel = Vec::new();
    float_kernel
        .try_reserve_exact(kernel_size)
        .map_err(|_| CoreError::Invalid("GaussianBlur kernel_size is too large".into()))?;
    let mut sum = 0.0_f64;
    for index in 0..kernel_size {
        let x = index.abs_diff(radius) as f64;
        let scaled_distance = x / f64::from(sigma);
        let value = (-0.5 * scaled_distance * scaled_distance).exp();
        float_kernel.push(value);
        sum += value;
    }
    let mut kernel = Vec::new();
    kernel
        .try_reserve_exact(kernel_size)
        .map_err(|_| CoreError::Invalid("GaussianBlur kernel_size is too large".into()))?;
    let mut quantized_sum = 0_u64;
    for value in &mut float_kernel {
        let scaled = *value / sum * 256.0;
        let rounded = scaled.round() as u16;
        kernel.push(rounded);
        quantized_sum += u64::from(rounded);
        *value = scaled - f64::from(rounded);
    }

    let correction =
        256_i64 - i64::try_from(quantized_sum).map_err(|_| invalid_gaussian_kernel())?;
    if correction.unsigned_abs() <= 1 {
        kernel[radius] = u16::try_from(i64::from(kernel[radius]) + correction)
            .map_err(|_| invalid_gaussian_kernel())?;
    } else {
        distribute_gaussian_residual(&mut kernel, &mut float_kernel, correction)?;
    }
    if kernel.iter().copied().map(u64::from).sum::<u64>() != 256
        || kernel.iter().any(|&weight| weight > 256)
        || !kernel.iter().eq(kernel.iter().rev())
    {
        return Err(invalid_gaussian_kernel());
    }
    Ok(kernel)
}

fn distribute_gaussian_residual(
    kernel: &mut [u16],
    rounding_errors: &mut [f64],
    correction: i64,
) -> CoreResult<()> {
    let radius = kernel.len() / 2;
    let direction = correction.signum() as i32;
    let mut remaining = correction.unsigned_abs();
    if remaining % 2 == 1 {
        kernel[radius] = u16::try_from(i32::from(kernel[radius]) + direction)
            .map_err(|_| invalid_gaussian_kernel())?;
        remaining -= 1;
    }
    while remaining != 0 {
        let best = (0..radius)
            .rev()
            .filter(|&index| direction > 0 || kernel[index] != 0)
            .max_by(|&left, &right| {
                (rounding_errors[left] * f64::from(direction))
                    .total_cmp(&(rounding_errors[right] * f64::from(direction)))
            })
            .ok_or_else(invalid_gaussian_kernel)?;
        let mirrored = kernel.len() - 1 - best;
        kernel[best] = u16::try_from(i32::from(kernel[best]) + direction)
            .map_err(|_| invalid_gaussian_kernel())?;
        kernel[mirrored] = kernel[best];
        rounding_errors[best] = if direction > 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        remaining -= 2;
    }
    Ok(())
}

fn invalid_gaussian_kernel() -> CoreError {
    CoreError::Invalid("GaussianBlur kernel could not be quantized safely".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_seed_derivation_is_stable_and_keyed() {
        assert_eq!(derive_run_seed(137, 29), 14_249_732_241_852_757_271);
        assert_ne!(derive_run_seed(137, 29), derive_run_seed(137, 30));
        assert_ne!(derive_run_seed(137, 29), derive_run_seed(138, 29));
    }

    #[test]
    fn sampled_geometry_is_deterministic_and_tracks_shape() {
        let transforms = TransformPlan::compile(vec![
            TransformSpec::RandomCrop {
                height: 11,
                width: 13,
                p: 1.0,
            },
            TransformSpec::Resize {
                height: 7,
                width: 9,
                interpolation: Interpolation::Bilinear,
                antialias: false,
                p: 1.0,
            },
            TransformSpec::RandomCrop {
                height: 5,
                width: 6,
                p: 1.0,
            },
        ])
        .unwrap();
        let first = TransformPlan::sample(&transforms, 17, 19, 137).unwrap();
        let second = TransformPlan::sample(&transforms, 17, 19, 137).unwrap();
        assert_eq!(first, second);
        assert!(matches!(
            first[2],
            SampledTransform::RandomCrop(CropSample {
                top: 0..=2,
                left: 0..=3,
                height: 5,
                width: 6,
            })
        ));
    }

    #[test]
    fn pad_if_needed_samples_positions_and_tracks_output_shape() {
        let positions = [
            (PadPosition::Center, (1, 2)),
            (PadPosition::TopLeft, (0, 0)),
            (PadPosition::TopRight, (0, 5)),
            (PadPosition::BottomLeft, (3, 0)),
            (PadPosition::BottomRight, (3, 5)),
        ];
        for (position, expected) in positions {
            let transforms = TransformPlan::compile(vec![TransformSpec::PadIfNeeded {
                min_height: Some(5),
                min_width: Some(8),
                pad_height_divisor: None,
                pad_width_divisor: None,
                position,
                border_mode: BorderMode::Constant,
                fill: [3, 5, 7],
                p: 1.0,
            }])
            .unwrap();
            let sampled = TransformPlan::sample(&transforms, 2, 3, 137).unwrap();
            assert!(matches!(
                sampled[0],
                SampledTransform::PadIfNeeded(PadSample {
                    top,
                    left,
                    height: 5,
                    width: 8,
                    ..
                }) if (top, left) == expected
            ));
        }

        let transforms = TransformPlan::compile(vec![
            TransformSpec::PadIfNeeded {
                min_height: None,
                min_width: None,
                pad_height_divisor: Some(4),
                pad_width_divisor: Some(5),
                position: PadPosition::Random,
                border_mode: BorderMode::Reflect101,
                fill: [0; 3],
                p: 1.0,
            },
            TransformSpec::RandomCrop {
                height: 8,
                width: 10,
                p: 1.0,
            },
        ])
        .unwrap();
        let first = TransformPlan::sample(&transforms, 5, 7, 137).unwrap();
        let second = TransformPlan::sample(&transforms, 5, 7, 137).unwrap();
        assert_eq!(first, second);
        assert!(matches!(
            first[0],
            SampledTransform::PadIfNeeded(PadSample {
                top: 0..=3,
                left: 0..=3,
                height: 8,
                width: 10,
                ..
            })
        ));
        assert!(matches!(
            first[1],
            SampledTransform::RandomCrop(CropSample {
                top: 0,
                left: 0,
                height: 8,
                width: 10,
            })
        ));
    }

    #[test]
    fn pad_if_needed_axis_strategies_are_validated() {
        for (minimum, divisor) in [(None, None), (Some(5), Some(4)), (Some(0), None)] {
            assert!(TransformPlan::compile(vec![TransformSpec::PadIfNeeded {
                min_height: minimum,
                min_width: Some(7),
                pad_height_divisor: divisor,
                pad_width_divisor: None,
                position: PadPosition::Center,
                border_mode: BorderMode::Constant,
                fill: [0; 3],
                p: 1.0,
            }])
            .is_err());
        }
    }

    #[test]
    fn coarse_dropout_sampling_is_deterministic_and_bounded() {
        let transforms = TransformPlan::compile(vec![TransformSpec::CoarseDropout {
            num_holes_range: [3, 3],
            hole_height_range: DropoutSizeRange::Fraction([0.25, 0.25]),
            hole_width_range: DropoutSizeRange::Pixels([7, 7]),
            fill: [3, 5, 7],
            p: 1.0,
        }])
        .unwrap();
        let first = TransformPlan::sample(&transforms, 20, 40, 137).unwrap();
        let second = TransformPlan::sample(&transforms, 20, 40, 137).unwrap();
        assert_eq!(first, second);
        let SampledTransform::CoarseDropout { holes, fill } = &first[0] else {
            panic!("expected coarse dropout")
        };
        assert_eq!(*fill, [3, 5, 7]);
        assert_eq!(holes.len(), 3);
        for hole in holes {
            assert_eq!((hole.height, hole.width), (5, 7));
            assert!(hole.top + hole.height <= 20);
            assert!(hole.left + hole.width <= 40);
        }

        let clamped = TransformPlan::compile(vec![TransformSpec::CoarseDropout {
            num_holes_range: [1, 1],
            hole_height_range: DropoutSizeRange::Pixels([100, 200]),
            hole_width_range: DropoutSizeRange::Fraction([0.01, 0.01]),
            fill: [0; 3],
            p: 1.0,
        }])
        .unwrap();
        let SampledTransform::CoarseDropout { holes, .. } =
            &TransformPlan::sample(&clamped, 3, 5, 137).unwrap()[0]
        else {
            panic!("expected coarse dropout")
        };
        assert_eq!((holes[0].height, holes[0].width), (3, 1));
    }

    #[test]
    fn coarse_dropout_ranges_are_validated_in_core() {
        for (count, height, width) in [
            (
                [0, 1],
                DropoutSizeRange::Pixels([1, 2]),
                DropoutSizeRange::Fraction([0.1, 0.2]),
            ),
            (
                [1, 2],
                DropoutSizeRange::Pixels([3, 2]),
                DropoutSizeRange::Fraction([0.1, 0.2]),
            ),
            (
                [1, 2],
                DropoutSizeRange::Pixels([1, 2]),
                DropoutSizeRange::Fraction([0.0, 0.2]),
            ),
            (
                [1, 2],
                DropoutSizeRange::Pixels([1, 2]),
                DropoutSizeRange::Fraction([0.2, 1.1]),
            ),
        ] {
            assert!(TransformPlan::compile(vec![TransformSpec::CoarseDropout {
                num_holes_range: count,
                hole_height_range: height,
                hole_width_range: width,
                fill: [0; 3],
                p: 1.0,
            }])
            .is_err());
        }
    }

    #[test]
    fn affine_sampling_is_deterministic_and_within_configured_ranges() {
        let transforms = TransformPlan::compile(vec![TransformSpec::Affine {
            degrees: [-17.0, 23.0],
            translate: [0.25, 0.4],
            scale: [0.75, 1.4],
            shear: [-13.0, 19.0, -9.0, 11.0],
            interpolation: Interpolation::Bilinear,
            border_mode: BorderMode::Reflect101,
            fill: [11, 13, 17],
            p: 1.0,
        }])
        .unwrap();
        let first = TransformPlan::sample(&transforms, 20, 40, 137).unwrap();
        let second = TransformPlan::sample(&transforms, 20, 40, 137).unwrap();
        assert_eq!(first, second);
        let SampledTransform::Affine(sample) = first[0] else {
            panic!("expected an affine sample")
        };
        assert!((-17.0..=23.0).contains(&sample.degrees));
        assert!((-10.0..=10.0).contains(&sample.translate[0]));
        assert!((-8.0..=8.0).contains(&sample.translate[1]));
        assert!((0.75..=1.4).contains(&sample.scale));
        assert!((-13.0..=19.0).contains(&sample.shear[0]));
        assert!((-9.0..=11.0).contains(&sample.shear[1]));
    }

    #[test]
    fn color_jitter_sampling_covers_ranges_and_hue_order() {
        let transforms = TransformPlan::compile(vec![TransformSpec::ColorJitter {
            brightness: [0.7, 1.4],
            contrast: [0.8, 1.2],
            saturation: [0.5, 1.5],
            hue: [-0.25, 0.3],
            p: 1.0,
        }])
        .unwrap();
        let first = TransformPlan::sample(&transforms, 17, 19, 137).unwrap();
        let second = TransformPlan::sample(&transforms, 17, 19, 137).unwrap();
        assert_eq!(first, second);
        let SampledTransform::ColorJitter(sample) = first[0] else {
            panic!("expected a color jitter sample")
        };
        assert!((0.7..=1.4).contains(&sample.brightness));
        assert!((0.8..=1.2).contains(&sample.contrast));
        assert!((0.5..=1.5).contains(&sample.saturation));
        assert!((-0.25..=0.3).contains(&sample.hue));
        assert!(sample.hue_enabled);
        let mut order = sample.order;
        order.sort_unstable();
        assert_eq!(order, [0, 1, 2, 3]);

        let without_hue = TransformPlan::compile(vec![TransformSpec::ColorJitter {
            brightness: [0.8, 1.2],
            contrast: [0.8, 1.2],
            saturation: [0.8, 1.2],
            hue: [0.0, 0.0],
            p: 1.0,
        }])
        .unwrap();
        let SampledTransform::ColorJitter(sample) =
            TransformPlan::sample(&without_hue, 17, 19, 137).unwrap()[0]
        else {
            panic!("expected a color jitter sample")
        };
        assert!(!sample.hue_enabled);
        assert_eq!(sample.order[3], 3);
    }

    #[test]
    fn color_jitter_ranges_are_validated_in_core() {
        let invalid = [
            TransformSpec::ColorJitter {
                brightness: [-0.1, 1.0],
                contrast: [1.0, 1.0],
                saturation: [1.0, 1.0],
                hue: [0.0, 0.0],
                p: 1.0,
            },
            TransformSpec::ColorJitter {
                brightness: [1.0, 1.0],
                contrast: [1.2, 0.8],
                saturation: [1.0, 1.0],
                hue: [0.0, 0.0],
                p: 1.0,
            },
            TransformSpec::ColorJitter {
                brightness: [1.0, 1.0],
                contrast: [1.0, 1.0],
                saturation: [1.0, 1.0],
                hue: [-0.6, 0.0],
                p: 1.0,
            },
        ];
        for transform in invalid {
            assert!(TransformPlan::compile(vec![transform]).is_err());
        }
    }

    #[test]
    fn gaussian_sigma_sampling_is_deterministic_and_preserves_the_fixed_fast_path() {
        let ranged = TransformPlan::compile(vec![TransformSpec::GaussianBlur {
            kernel_size: 5,
            sigma: [0.6, 2.0],
            p: 1.0,
        }])
        .unwrap();
        assert!(matches!(
            ranged[0],
            TransformPlan::GaussianBlur {
                fixed_kernel: None,
                ..
            }
        ));
        let first = TransformPlan::sample(&ranged, 17, 19, 137).unwrap();
        let second = TransformPlan::sample(&ranged, 17, 19, 137).unwrap();
        assert_eq!(first, second);
        let SampledTransform::GaussianBlur { sigma } = first[0] else {
            panic!("expected a Gaussian blur sample")
        };
        assert!((0.6..=2.0).contains(&sigma));

        let fixed = TransformPlan::compile(vec![TransformSpec::GaussianBlur {
            kernel_size: 5,
            sigma: [1.1, 1.1],
            p: 1.0,
        }])
        .unwrap();
        assert!(matches!(
            &fixed[0],
            TransformPlan::GaussianBlur {
                fixed_kernel: Some(kernel),
                ..
            } if kernel.len() == 5
        ));
    }

    #[test]
    fn gaussian_sigma_ranges_are_validated_in_core() {
        for sigma in [[0.0, 1.0], [1.2, 0.8], [1.0, f32::INFINITY]] {
            assert!(TransformPlan::compile(vec![TransformSpec::GaussianBlur {
                kernel_size: 5,
                sigma,
                p: 1.0,
            }])
            .is_err());
        }
    }

    #[test]
    fn gaussian_kernel_quantization_is_symmetric_and_normalized() {
        for kernel_size in [1, 3, 5, 101, 511] {
            for sigma in [f32::MIN_POSITIVE, 0.000_001, 1.1, 1_000_000.0, f32::MAX] {
                let kernel = make_gaussian_kernel(kernel_size, sigma).unwrap();
                assert_eq!(kernel.len(), kernel_size);
                assert_eq!(kernel.iter().copied().map(u32::from).sum::<u32>(), 256);
                assert!(kernel.iter().all(|&weight| weight <= 256));
                assert!(kernel.iter().eq(kernel.iter().rev()));
            }
        }
        assert_eq!(make_gaussian_kernel(5, 1.1).unwrap(), [18, 63, 94, 63, 18]);
    }

    #[test]
    fn configuration_allocations_fail_without_panicking() {
        assert!(matches!(
            make_gaussian_kernel(usize::MAX, 1.0),
            Err(CoreError::Invalid(message)) if message.contains("kernel_size")
        ));

        let transforms = TransformPlan::compile(vec![TransformSpec::CoarseDropout {
            num_holes_range: [usize::MAX, usize::MAX],
            hole_height_range: DropoutSizeRange::Pixels([1, 1]),
            hole_width_range: DropoutSizeRange::Pixels([1, 1]),
            fill: [0; 3],
            p: 1.0,
        }])
        .unwrap();
        assert!(matches!(
            TransformPlan::sample(&transforms, 1, 1, 137),
            Err(CoreError::Runtime(message)) if message.contains("allocation")
        ));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn dimensions_enforce_backend_and_buffer_limits() {
        assert!(TransformPlan::compile(vec![TransformSpec::Resize {
            height: u32::MAX as usize + 1,
            width: 1,
            interpolation: Interpolation::Nearest,
            antialias: false,
            p: 1.0,
        }])
        .is_err());
        assert!(TransformPlan::compile(vec![TransformSpec::Resize {
            height: u32::MAX as usize,
            width: u32::MAX as usize,
            interpolation: Interpolation::Nearest,
            antialias: false,
            p: 1.0,
        }])
        .is_err());
    }

    #[test]
    fn affine_ranges_are_validated_in_core() {
        let invalid = [
            TransformSpec::Affine {
                degrees: [10.0, -10.0],
                translate: [0.0, 0.0],
                scale: [1.0, 1.0],
                shear: [0.0; 4],
                interpolation: Interpolation::Bilinear,
                border_mode: BorderMode::Constant,
                fill: [0; 3],
                p: 1.0,
            },
            TransformSpec::Affine {
                degrees: [0.0, 0.0],
                translate: [1.1, 0.0],
                scale: [1.0, 1.0],
                shear: [0.0; 4],
                interpolation: Interpolation::Bilinear,
                border_mode: BorderMode::Constant,
                fill: [0; 3],
                p: 1.0,
            },
            TransformSpec::Affine {
                degrees: [0.0, 0.0],
                translate: [0.0, 0.0],
                scale: [0.0, 1.0],
                shear: [0.0; 4],
                interpolation: Interpolation::Bilinear,
                border_mode: BorderMode::Constant,
                fill: [0; 3],
                p: 1.0,
            },
            TransformSpec::Affine {
                degrees: [0.0, 0.0],
                translate: [0.0, 0.0],
                scale: [1.0, 1.0],
                shear: [0.0, 0.0, 90.0, 90.0],
                interpolation: Interpolation::Bilinear,
                border_mode: BorderMode::Constant,
                fill: [0; 3],
                p: 1.0,
            },
        ];
        for transform in invalid {
            assert!(TransformPlan::compile(vec![transform]).is_err());
        }
    }

    #[test]
    fn random_resized_crop_sampling_is_deterministic_and_tracks_output_shape() {
        let transforms = TransformPlan::compile(vec![
            TransformSpec::RandomResizedCrop {
                height: 7,
                width: 9,
                scale: [1.0, 1.0],
                ratio: [4.0, 4.0],
                interpolation: Interpolation::Bilinear,
                antialias: true,
                p: 1.0,
            },
            TransformSpec::RandomCrop {
                height: 6,
                width: 8,
                p: 1.0,
            },
        ])
        .unwrap();
        let first = TransformPlan::sample(&transforms, 10, 20, 137).unwrap();
        let second = TransformPlan::sample(&transforms, 10, 20, 137).unwrap();
        assert_eq!(first, second);
        assert!(matches!(
            first[0],
            SampledTransform::RandomResizedCrop {
                crop: CropSample {
                    top: 2,
                    left: 0,
                    height: 5,
                    width: 20,
                },
                height: 7,
                width: 9,
                interpolation: Interpolation::Bilinear,
                antialias: true,
            }
        ));
        assert!(matches!(
            first[1],
            SampledTransform::RandomCrop(CropSample {
                top: 0..=1,
                left: 0..=1,
                height: 6,
                width: 8,
            })
        ));
    }

    #[test]
    fn random_resized_crop_ranges_are_validated() {
        for (scale, ratio) in [
            ([0.0, 1.0], [0.75, 1.25]),
            ([0.8, 0.2], [0.75, 1.25]),
            ([0.2, 1.1], [0.75, 1.25]),
            ([0.2, 0.8], [1.25, 0.75]),
        ] {
            assert!(
                TransformPlan::compile(vec![TransformSpec::RandomResizedCrop {
                    height: 7,
                    width: 9,
                    scale,
                    ratio,
                    interpolation: Interpolation::Bilinear,
                    antialias: false,
                    p: 1.0,
                }])
                .is_err()
            );
        }
    }

    #[test]
    fn skipped_invalid_crop_does_not_fail() {
        let transforms = TransformPlan::compile(vec![TransformSpec::RandomCrop {
            height: 20,
            width: 20,
            p: 0.0,
        }])
        .unwrap();
        assert_eq!(
            TransformPlan::sample(&transforms, 3, 5, 137).unwrap(),
            vec![SampledTransform::Skip]
        );
    }

    #[test]
    fn terminal_layout_rules_are_validated() {
        TransformPlan::compile(vec![
            TransformSpec::Normalize {
                mean: [0.0; 3],
                std: [1.0; 3],
                max_pixel_value: 255.0,
                p: 1.0,
            },
            TransformSpec::ToTorch,
        ])
        .unwrap();

        let error = TransformPlan::compile(vec![
            TransformSpec::ToTorch,
            TransformSpec::Invert { p: 1.0 },
        ])
        .unwrap_err();
        assert!(
            matches!(error, CoreError::Invalid(message) if message == "ToTorch must be terminal")
        );
    }

    #[test]
    fn grid_maps_are_monotonic_and_endpoint_anchored() {
        let mut rng = SmallRng::seed_from_u64(137);
        for length in [1, 2, 7, 19] {
            let map = sample_grid_map(length, 9, [-0.8, 0.8], &mut rng).unwrap();
            assert_eq!(map.len(), length);
            assert_eq!(map[0], 0.0);
            assert_eq!(*map.last().unwrap(), (length - 1) as f32);
            assert!(map.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn identity_homography_is_exact() {
        let corners = [[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]];
        assert_eq!(
            solve_homography(corners, corners).unwrap(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }
}
