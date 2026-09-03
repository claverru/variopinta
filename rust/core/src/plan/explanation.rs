use super::*;

impl TransformPlan {
    pub(crate) fn explain(&self) -> TransformExplanation {
        let (category, execution, pixel_passes, allocation, fallback) = match self {
            Self::Resize { .. } => ("geometry", "out-of-place", 1, "workspace-u8", "portable"),
            Self::RandomCrop { .. } | Self::CenterCrop { .. } => (
                "geometry",
                "out-of-place",
                1,
                "workspace-u8",
                "portable-scalar",
            ),
            Self::PadIfNeeded { .. } => (
                "geometry",
                "out-of-place",
                1,
                "workspace-u8",
                "portable-scalar",
            ),
            Self::CoarseDropout { .. } => (
                "dropout",
                "in-place",
                1,
                "sampled-rectangles",
                "portable-scalar",
            ),
            Self::RandomResizedCrop { .. } => {
                ("geometry", "out-of-place", 2, "workspace-u8", "portable")
            }
            Self::HorizontalFlip { .. } => {
                ("geometry", "in-place", 1, "none", owned_simd_fallback())
            }
            Self::VerticalFlip { .. } => ("geometry", "in-place", 1, "none", "portable-scalar"),
            Self::ColorJitter { contrast, hue, .. } => (
                "color",
                "in-place",
                if *contrast == [1.0, 1.0] { 1 } else { 2 },
                "none",
                if *hue == [0.0, 0.0] {
                    owned_simd_numeric_fallback()
                } else {
                    hue_simd_fallback()
                },
            ),
            Self::Affine { .. } | Self::RandomRotation { .. } => (
                "geometry",
                "out-of-place",
                1,
                "workspace-u8",
                owned_simd_fallback(),
            ),
            Self::GaussianNoise { .. } => (
                "noise",
                "in-place",
                1,
                "sampled-rng-substream+workspace-f32-block",
                owned_simd_fallback(),
            ),
            Self::Sharpen { .. } => (
                "filter",
                "out-of-place",
                1,
                "workspace-u8",
                owned_simd_fallback(),
            ),
            Self::Perspective { .. } => (
                "geometry",
                "out-of-place",
                1,
                "sampled-homography+workspace-u8",
                owned_simd_fallback(),
            ),
            Self::GridDistortion { .. } => (
                "geometry",
                "out-of-place",
                1,
                "sampled-coordinate-maps+workspace-axis-remap+workspace-u8",
                owned_simd_fallback(),
            ),
            Self::GaussianBlur { sigma, .. } => (
                "filter",
                "in-place",
                2,
                if sigma[0] == sigma[1] {
                    "workspace-u16"
                } else {
                    "workspace-u16+sampled-kernel"
                },
                owned_simd_fallback(),
            ),
            Self::Grayscale { .. } | Self::Solarize { .. } => {
                ("color", "in-place", 1, "none", owned_simd_fallback())
            }
            Self::Invert { .. } | Self::Posterize { .. } => {
                ("color", "in-place", 1, "none", owned_simd_fallback())
            }
            Self::Normalize { .. } => (
                "dtype",
                "terminal",
                1,
                "owned-f32-output",
                owned_simd_fallback(),
            ),
            Self::ToTorch => (
                "layout",
                "terminal",
                1,
                "owned-chw-output",
                layout_simd_fallback(),
            ),
        };
        let probability = self.probability();
        TransformExplanation {
            name: self.name(),
            category,
            probability,
            status: if probability == 0.0 {
                "never"
            } else if probability == 1.0 {
                "always"
            } else {
                "conditional"
            },
            execution,
            pixel_passes,
            allocation,
            fallback,
            input_materialization: "unplanned",
            kernel_form: "unplanned",
            output_slot: "unplanned",
            scratch_slots: Vec::new(),
            selection_reason: "unplanned",
            policies: self.policies(),
        }
    }

    pub(crate) fn probability(&self) -> f32 {
        match self {
            Self::Resize { p, .. }
            | Self::RandomCrop { p, .. }
            | Self::RandomResizedCrop { p, .. }
            | Self::HorizontalFlip { p }
            | Self::VerticalFlip { p }
            | Self::CenterCrop { p, .. }
            | Self::PadIfNeeded { p, .. }
            | Self::CoarseDropout { p, .. }
            | Self::ColorJitter { p, .. }
            | Self::Affine { p, .. }
            | Self::RandomRotation { p, .. }
            | Self::GaussianNoise { p, .. }
            | Self::Sharpen { p, .. }
            | Self::Perspective { p, .. }
            | Self::GridDistortion { p, .. }
            | Self::GaussianBlur { p, .. }
            | Self::Grayscale { p }
            | Self::Invert { p }
            | Self::Solarize { p, .. }
            | Self::Posterize { p, .. }
            | Self::Normalize { p, .. } => *p,
            Self::ToTorch => 1.0,
        }
    }

    fn policies(&self) -> Vec<PolicyExplanation> {
        let policy = |name, value| PolicyExplanation { name, value };
        match self {
            Self::Resize {
                height,
                width,
                interpolation,
                antialias,
                ..
            } => vec![
                policy("size", format!("{height}x{width}")),
                policy("interpolation", interpolation_name(*interpolation).into()),
                policy(
                    "antialias",
                    match interpolation {
                        Interpolation::Nearest => "ignored".into(),
                        Interpolation::Bilinear => antialias.to_string(),
                    },
                ),
            ],
            Self::RandomCrop { height, width, .. } => vec![
                policy("size", format!("{height}x{width}")),
                policy("origin", "uniform-inclusive".into()),
            ],
            Self::RandomResizedCrop {
                height,
                width,
                scale,
                ratio,
                interpolation,
                antialias,
                ..
            } => vec![
                policy("size", format!("{height}x{width}")),
                policy("scale", format!("[{},{}]", scale[0], scale[1])),
                policy("ratio", format!("[{},{}]", ratio[0], ratio[1])),
                policy("sampling-attempts", "10".into()),
                policy("fallback", "centered-ratio-clamp".into()),
                policy("interpolation", interpolation_name(*interpolation).into()),
                policy(
                    "antialias",
                    match interpolation {
                        Interpolation::Nearest => "ignored".into(),
                        Interpolation::Bilinear => antialias.to_string(),
                    },
                ),
            ],
            Self::CenterCrop { height, width, .. } => vec![
                policy("size", format!("{height}x{width}")),
                policy("odd-remainder", "bottom-right".into()),
            ],
            Self::PadIfNeeded {
                min_height,
                min_width,
                pad_height_divisor,
                pad_width_divisor,
                position,
                border_mode,
                fill,
                ..
            } => vec![
                policy("height", pad_axis_policy(*min_height, *pad_height_divisor)),
                policy("width", pad_axis_policy(*min_width, *pad_width_divisor)),
                policy("position", pad_position_name(*position).into()),
                policy("border", border_name(*border_mode).into()),
                policy("fill", format!("[{},{},{}]", fill[0], fill[1], fill[2])),
            ],
            Self::CoarseDropout {
                num_holes_range,
                hole_height_range,
                hole_width_range,
                fill,
                ..
            } => vec![
                policy(
                    "holes",
                    format!("[{},{}]", num_holes_range[0], num_holes_range[1]),
                ),
                policy("hole-height", dropout_size_policy(*hole_height_range)),
                policy("hole-width", dropout_size_policy(*hole_width_range)),
                policy("size-rounding", "floor-clamped-to-at-least-one".into()),
                policy("origin", "uniform-inclusive".into()),
                policy("overlap", "allowed".into()),
                policy("fill", format!("[{},{},{}]", fill[0], fill[1], fill[2])),
            ],
            Self::ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
                ..
            } => vec![
                policy(
                    "brightness",
                    format!("[{},{}]", brightness[0], brightness[1]),
                ),
                policy("contrast", format!("[{},{}]", contrast[0], contrast[1])),
                policy(
                    "saturation",
                    format!("[{},{}]", saturation[0], saturation[1]),
                ),
                policy("hue", format!("[{},{}]", hue[0], hue[1])),
                policy("order", "uniform-random-permutation".into()),
            ],
            Self::Affine {
                degrees,
                translate,
                scale,
                shear,
                interpolation,
                border_mode,
                fill,
                ..
            } => vec![
                policy("degrees", format!("[{},{}]", degrees[0], degrees[1])),
                policy(
                    "translate-fraction",
                    format!(
                        "[-{},{}]x[-{},{}]",
                        translate[0], translate[0], translate[1], translate[1]
                    ),
                ),
                policy("scale", format!("[{},{}]", scale[0], scale[1])),
                policy(
                    "shear-degrees",
                    format!("[{},{}]x[{},{}]", shear[0], shear[1], shear[2], shear[3]),
                ),
                policy("interpolation", interpolation_name(*interpolation).into()),
                policy("border", border_name(*border_mode).into()),
                policy("fill", format!("[{},{},{}]", fill[0], fill[1], fill[2])),
            ],
            Self::RandomRotation {
                degrees,
                interpolation,
                border_mode,
                fill,
                ..
            } => vec![
                policy("degrees", format!("[{},{}]", degrees[0], degrees[1])),
                policy("interpolation", interpolation_name(*interpolation).into()),
                policy("border", border_name(*border_mode).into()),
                policy("fill", format!("[{},{},{}]", fill[0], fill[1], fill[2])),
                policy("kernel", "Affine".into()),
            ],
            Self::GaussianNoise {
                mean,
                std,
                per_channel,
                ..
            } => vec![
                policy("mean-uint8", format!("[{},{}]", mean[0], mean[1])),
                policy("std-uint8", format!("[{},{}]", std[0], std[1])),
                policy("distribution", "rand-distr-0.5.1-zignor-normal".into()),
                policy(
                    "channels",
                    if *per_channel {
                        "independent-rgb"
                    } else {
                        "shared-rgb"
                    }
                    .into(),
                ),
                policy("rounding", "nearest-then-saturate-uint8".into()),
            ],
            Self::Sharpen {
                alpha, lightness, ..
            } => vec![
                policy("alpha", format!("[{},{}]", alpha[0], alpha[1])),
                policy("lightness", format!("[{},{}]", lightness[0], lightness[1])),
                policy("kernel", "cross-3x3-sum-one".into()),
                policy("border", "reflect101".into()),
                policy("rounding", "nearest-then-saturate-uint8".into()),
            ],
            Self::Perspective {
                scale,
                interpolation,
                border_mode,
                fill,
                ..
            } => vec![
                policy("scale", format!("[{},{}]", scale[0], scale[1])),
                policy("corner-displacement", "inward-bounded-fraction".into()),
                policy("degenerate-attempts", "10-then-identity".into()),
                policy("interpolation", interpolation_name(*interpolation).into()),
                policy("border", border_name(*border_mode).into()),
                policy("fill", format!("[{},{},{}]", fill[0], fill[1], fill[2])),
                policy("sampler", "shared-inverse-q8".into()),
            ],
            Self::GridDistortion {
                num_steps,
                distort_limit,
                interpolation,
                border_mode,
                fill,
                ..
            } => vec![
                policy("steps", num_steps.to_string()),
                policy(
                    "distort-limit",
                    format!("[{},{}]", distort_limit[0], distort_limit[1]),
                ),
                policy("maps", "positive-monotonic-anchored".into()),
                policy("small-axis", "reduce-steps-to-axis-minus-one".into()),
                policy("interpolation", interpolation_name(*interpolation).into()),
                policy("border", border_name(*border_mode).into()),
                policy("fill", format!("[{},{},{}]", fill[0], fill[1], fill[2])),
                policy("sampler", "shared-inverse-q8".into()),
            ],
            Self::GaussianBlur {
                kernel_size, sigma, ..
            } => {
                vec![
                    policy("kernel-size", kernel_size.to_string()),
                    policy("sigma", format!("[{},{}]", sigma[0], sigma[1])),
                    policy("border", "reflect101".into()),
                ]
            }
            Self::Grayscale { .. } => vec![
                policy("coefficients", "bt601-q8".into()),
                policy("channels", "replicate-rgb".into()),
            ],
            Self::Solarize { threshold, .. } => {
                vec![
                    policy("threshold", threshold.to_string()),
                    policy("comparison", "at-or-above".into()),
                ]
            }
            Self::Posterize { bits, .. } => vec![policy("bits", bits.to_string())],
            Self::Normalize {
                max_pixel_value, ..
            } => vec![policy("max-pixel-value", max_pixel_value.to_string())],
            Self::ToTorch => vec![
                policy("layout", "CHW".into()),
                policy("dtype", "preserve".into()),
                policy("device", "cpu".into()),
            ],
            Self::HorizontalFlip { .. } | Self::VerticalFlip { .. } | Self::Invert { .. } => {
                Vec::new()
            }
        }
    }
}

fn layout_simd_fallback() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        "runtime-ssse3-or-portable-scalar"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon-or-portable-scalar"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "portable-scalar"
    }
}

fn hue_simd_fallback() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        "hue-runtime-avx2-or-portable-scalar"
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "hue-portable-scalar"
    }
}
