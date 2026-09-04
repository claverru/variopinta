use super::*;

impl TransformPlan {
    pub(crate) fn compile(specs: Vec<TransformSpec>) -> CoreResult<Vec<Self>> {
        for (index, transform) in specs.iter().enumerate() {
            if matches!(transform, TransformSpec::Normalize { .. }) && index + 1 < specs.len() {
                return Err(CoreError::Invalid("Normalize must be terminal".into()));
            }
        }
        specs.into_iter().map(Self::compile_one).collect()
    }

    fn compile_one(spec: TransformSpec) -> CoreResult<Self> {
        Ok(match spec {
            TransformSpec::Resize {
                height,
                width,
                interpolation,
                antialias,
                p,
            } => {
                validate_dimensions(height, width)?;
                validate_probability(p)?;
                Self::Resize {
                    height,
                    width,
                    interpolation,
                    antialias,
                    p,
                }
            }
            TransformSpec::RandomCrop { height, width, p } => {
                validate_dimensions(height, width)?;
                validate_probability(p)?;
                Self::RandomCrop { height, width, p }
            }
            TransformSpec::RandomResizedCrop {
                height,
                width,
                scale,
                ratio,
                interpolation,
                antialias,
                p,
            } => {
                validate_dimensions(height, width)?;
                validate_positive_range("scale", scale, Some(1.0))?;
                validate_positive_range("ratio", ratio, None)?;
                validate_probability(p)?;
                Self::RandomResizedCrop {
                    height,
                    width,
                    scale,
                    ratio,
                    interpolation,
                    antialias,
                    p,
                }
            }
            TransformSpec::HorizontalFlip { p } => {
                validate_probability(p)?;
                Self::HorizontalFlip { p }
            }
            TransformSpec::VerticalFlip { p } => {
                validate_probability(p)?;
                Self::VerticalFlip { p }
            }
            TransformSpec::CenterCrop { height, width, p } => {
                validate_dimensions(height, width)?;
                validate_probability(p)?;
                Self::CenterCrop { height, width, p }
            }
            TransformSpec::PadIfNeeded {
                min_height,
                min_width,
                pad_height_divisor,
                pad_width_divisor,
                position,
                border_mode,
                fill,
                p,
            } => {
                validate_pad_axis("height", min_height, pad_height_divisor)?;
                validate_pad_axis("width", min_width, pad_width_divisor)?;
                validate_probability(p)?;
                Self::PadIfNeeded {
                    min_height,
                    min_width,
                    pad_height_divisor,
                    pad_width_divisor,
                    position,
                    border_mode,
                    fill,
                    p,
                }
            }
            TransformSpec::CoarseDropout {
                num_holes_range,
                hole_height_range,
                hole_width_range,
                fill,
                p,
            } => {
                validate_positive_usize_range("CoarseDropout num_holes_range", num_holes_range)?;
                validate_dropout_size_range("CoarseDropout hole_height_range", hole_height_range)?;
                validate_dropout_size_range("CoarseDropout hole_width_range", hole_width_range)?;
                validate_probability(p)?;
                Self::CoarseDropout {
                    num_holes_range,
                    hole_height_range,
                    hole_width_range,
                    fill,
                    p,
                }
            }
            TransformSpec::ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
                p,
            } => {
                validate_non_negative_range("ColorJitter brightness", brightness)?;
                validate_non_negative_range("ColorJitter contrast", contrast)?;
                validate_non_negative_range("ColorJitter saturation", saturation)?;
                if hue.iter().any(|value| !value.is_finite())
                    || hue[0] > hue[1]
                    || hue[0] < -0.5
                    || hue[1] > 0.5
                {
                    return Err(CoreError::Invalid(
                        "ColorJitter hue must be an ordered finite range in [-0.5, 0.5]".into(),
                    ));
                }
                validate_probability(p)?;
                Self::ColorJitter {
                    brightness,
                    contrast,
                    saturation,
                    hue,
                    p,
                }
            }
            TransformSpec::Affine {
                degrees,
                translate,
                scale,
                shear,
                interpolation,
                border_mode,
                fill,
                p,
            } => {
                if degrees.iter().any(|value| !value.is_finite()) || degrees[0] > degrees[1] {
                    return Err(CoreError::Invalid(
                        "Affine degrees must be a finite ordered range".into(),
                    ));
                }
                if translate
                    .iter()
                    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                {
                    return Err(CoreError::Invalid(
                        "Affine translate values must be finite and in [0, 1]".into(),
                    ));
                }
                if scale
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                    || scale[0] > scale[1]
                {
                    return Err(CoreError::Invalid(
                        "Affine scale must be a finite ordered positive range".into(),
                    ));
                }
                if shear
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() >= 90.0)
                    || shear[0] > shear[1]
                    || shear[2] > shear[3]
                {
                    return Err(CoreError::Invalid(
                        "Affine shear must contain finite ordered ranges within (-90, 90)".into(),
                    ));
                }
                validate_probability(p)?;
                Self::Affine {
                    degrees,
                    translate,
                    scale,
                    shear,
                    interpolation,
                    border_mode,
                    fill,
                    p,
                }
            }
            TransformSpec::RandomRotation {
                degrees,
                interpolation,
                border_mode,
                fill,
                p,
            } => {
                if degrees.iter().any(|value| !value.is_finite()) || degrees[0] > degrees[1] {
                    return Err(CoreError::Invalid(
                        "RandomRotation degrees must be a finite ordered range".into(),
                    ));
                }
                validate_probability(p)?;
                Self::RandomRotation {
                    degrees,
                    interpolation,
                    border_mode,
                    fill,
                    p,
                }
            }
            TransformSpec::GaussianNoise {
                mean,
                std,
                per_channel,
                p,
            } => {
                validate_finite_range("GaussianNoise mean", mean)?;
                validate_non_negative_range("GaussianNoise std", std)?;
                validate_probability(p)?;
                Self::GaussianNoise {
                    mean,
                    std,
                    per_channel,
                    p,
                }
            }
            TransformSpec::Sharpen {
                alpha,
                lightness,
                p,
            } => {
                validate_non_negative_range("Sharpen alpha", alpha)?;
                if alpha[1] > 1.0 {
                    return Err(CoreError::Invalid(
                        "Sharpen alpha values must be in [0, 1]".into(),
                    ));
                }
                validate_non_negative_range("Sharpen lightness", lightness)?;
                validate_probability(p)?;
                Self::Sharpen {
                    alpha,
                    lightness,
                    p,
                }
            }
            TransformSpec::Perspective {
                scale,
                interpolation,
                border_mode,
                fill,
                p,
            } => {
                validate_non_negative_range("Perspective scale", scale)?;
                if scale[1] >= 0.5 {
                    return Err(CoreError::Invalid(
                        "Perspective scale values must be in [0, 0.5)".into(),
                    ));
                }
                validate_probability(p)?;
                Self::Perspective {
                    scale,
                    interpolation,
                    border_mode,
                    fill,
                    p,
                }
            }
            TransformSpec::GridDistortion {
                num_steps,
                distort_limit,
                interpolation,
                border_mode,
                fill,
                p,
            } => {
                if num_steps == 0 {
                    return Err(CoreError::Invalid(
                        "GridDistortion num_steps must be positive".into(),
                    ));
                }
                validate_finite_range("GridDistortion distort_limit", distort_limit)?;
                if distort_limit[0] <= -1.0 || distort_limit[1] >= 1.0 {
                    return Err(CoreError::Invalid(
                        "GridDistortion distort_limit values must be within (-1, 1)".into(),
                    ));
                }
                validate_probability(p)?;
                Self::GridDistortion {
                    num_steps,
                    distort_limit,
                    interpolation,
                    border_mode,
                    fill,
                    p,
                }
            }
            TransformSpec::GaussianBlur {
                kernel_size,
                sigma,
                p,
            } => {
                validate_probability(p)?;
                validate_positive_range("GaussianBlur sigma", sigma, None)?;
                Self::GaussianBlur {
                    kernel_size,
                    sigma,
                    fixed_kernel: if sigma[0] == sigma[1] {
                        Some(make_gaussian_kernel(kernel_size, sigma[0])?)
                    } else {
                        make_gaussian_kernel(kernel_size, sigma[0])?;
                        make_gaussian_kernel(kernel_size, sigma[1])?;
                        None
                    },
                    p,
                }
            }
            TransformSpec::Grayscale { p } => {
                validate_probability(p)?;
                Self::Grayscale { p }
            }
            TransformSpec::Invert { p } => {
                validate_probability(p)?;
                Self::Invert { p }
            }
            TransformSpec::Solarize { threshold, p } => {
                validate_probability(p)?;
                Self::Solarize { threshold, p }
            }
            TransformSpec::Posterize { bits, p } => {
                if !(1..=8).contains(&bits) {
                    return Err(CoreError::Invalid(
                        "Posterize bits must be in [1, 8]".into(),
                    ));
                }
                validate_probability(p)?;
                Self::Posterize { bits, p }
            }
            TransformSpec::Normalize {
                mean,
                std,
                max_pixel_value,
                p,
            } => {
                if mean.iter().any(|value| !value.is_finite())
                    || std.iter().any(|value| !value.is_finite() || *value <= 0.0)
                    || !max_pixel_value.is_finite()
                    || max_pixel_value <= 0.0
                {
                    return Err(CoreError::Invalid(
                        "Normalize mean must be finite; std and max_pixel_value must be finite and positive".into(),
                    ));
                }
                validate_probability(p)?;
                Self::Normalize {
                    mean,
                    std,
                    max_pixel_value,
                    p,
                }
            }
        })
    }
}
