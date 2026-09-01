use crate::plan::TransformPlan;

macro_rules! transform_catalog {
    ($($variant:ident : $pattern:pat => $fixture:expr),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum TransformTag {
            $($variant),+
        }

        #[cfg(test)]
        pub(crate) const TRANSFORM_TAGS: &[TransformTag] = &[
            $(TransformTag::$variant),+
        ];

        pub const REGISTERED_TRANSFORM_NAMES: &[&str] = &[
            $(stringify!($variant)),+
        ];

        impl TransformTag {
            pub(crate) const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($variant)),+
                }
            }

            #[cfg(test)]
            fn fixture(self) -> crate::TransformSpec {
                match self {
                    $(Self::$variant => $fixture),+
                }
            }
        }

        impl TransformPlan {
            pub(crate) fn tag(&self) -> TransformTag {
                match self {
                    $($pattern => TransformTag::$variant),+
                }
            }
        }
    };
}

transform_catalog! {
    Resize: TransformPlan::Resize { .. } => crate::TransformSpec::Resize {
        height: 7,
        width: 11,
        interpolation: crate::Interpolation::Bilinear,
        antialias: false,
        p: 1.0,
    },
    RandomCrop: TransformPlan::RandomCrop { .. } => crate::TransformSpec::RandomCrop {
        height: 7,
        width: 11,
        p: 1.0,
    },
    RandomResizedCrop: TransformPlan::RandomResizedCrop { .. } => crate::TransformSpec::RandomResizedCrop {
        height: 7,
        width: 11,
        scale: [0.08, 1.0],
        ratio: [0.75, 4.0 / 3.0],
        interpolation: crate::Interpolation::Bilinear,
        antialias: false,
        p: 1.0,
    },
    HorizontalFlip: TransformPlan::HorizontalFlip { .. } => crate::TransformSpec::HorizontalFlip { p: 1.0 },
    VerticalFlip: TransformPlan::VerticalFlip { .. } => crate::TransformSpec::VerticalFlip { p: 1.0 },
    CenterCrop: TransformPlan::CenterCrop { .. } => crate::TransformSpec::CenterCrop {
        height: 7,
        width: 11,
        p: 1.0,
    },
    PadIfNeeded: TransformPlan::PadIfNeeded { .. } => crate::TransformSpec::PadIfNeeded {
        min_height: Some(7),
        min_width: Some(11),
        pad_height_divisor: None,
        pad_width_divisor: None,
        position: crate::PadPosition::Center,
        border_mode: crate::BorderMode::Constant,
        fill: [0; 3],
        p: 1.0,
    },
    CoarseDropout: TransformPlan::CoarseDropout { .. } => crate::TransformSpec::CoarseDropout {
        num_holes_range: [1, 1],
        hole_height_range: crate::DropoutSizeRange::Pixels([1, 1]),
        hole_width_range: crate::DropoutSizeRange::Pixels([1, 1]),
        fill: [0; 3],
        p: 1.0,
    },
    ColorJitter: TransformPlan::ColorJitter { .. } => crate::TransformSpec::ColorJitter {
        brightness: [1.0, 1.0],
        contrast: [1.0, 1.0],
        saturation: [1.0, 1.0],
        hue: [0.0, 0.0],
        p: 1.0,
    },
    Affine: TransformPlan::Affine { .. } => crate::TransformSpec::Affine {
        degrees: [0.0, 0.0],
        translate: [0.0, 0.0],
        scale: [1.0, 1.0],
        shear: [0.0; 4],
        interpolation: crate::Interpolation::Bilinear,
        border_mode: crate::BorderMode::Constant,
        fill: [0; 3],
        p: 1.0,
    },
    RandomRotation: TransformPlan::RandomRotation { .. } => crate::TransformSpec::RandomRotation {
        degrees: [-10.0, 10.0],
        interpolation: crate::Interpolation::Bilinear,
        border_mode: crate::BorderMode::Constant,
        fill: [0; 3],
        p: 1.0,
    },
    GaussianNoise: TransformPlan::GaussianNoise { .. } => crate::TransformSpec::GaussianNoise {
        mean: [0.0, 0.0],
        std: [10.0, 10.0],
        per_channel: true,
        p: 1.0,
    },
    Sharpen: TransformPlan::Sharpen { .. } => crate::TransformSpec::Sharpen {
        alpha: [0.5, 0.5],
        lightness: [1.0, 1.0],
        p: 1.0,
    },
    Perspective: TransformPlan::Perspective { .. } => crate::TransformSpec::Perspective {
        scale: [0.05, 0.05],
        interpolation: crate::Interpolation::Bilinear,
        border_mode: crate::BorderMode::Constant,
        fill: [0; 3],
        p: 1.0,
    },
    GridDistortion: TransformPlan::GridDistortion { .. } => crate::TransformSpec::GridDistortion {
        num_steps: 5,
        distort_limit: [-0.3, 0.3],
        interpolation: crate::Interpolation::Bilinear,
        border_mode: crate::BorderMode::Constant,
        fill: [0; 3],
        p: 1.0,
    },
    GaussianBlur: TransformPlan::GaussianBlur { .. } => crate::TransformSpec::GaussianBlur {
        kernel_size: 5,
        sigma: [1.0, 1.0],
        p: 1.0,
    },
    Grayscale: TransformPlan::Grayscale { .. } => crate::TransformSpec::Grayscale { p: 1.0 },
    Invert: TransformPlan::Invert { .. } => crate::TransformSpec::Invert { p: 1.0 },
    Solarize: TransformPlan::Solarize { .. } => crate::TransformSpec::Solarize {
        threshold: 128,
        p: 1.0,
    },
    Posterize: TransformPlan::Posterize { .. } => crate::TransformSpec::Posterize { bits: 4, p: 1.0 },
    Normalize: TransformPlan::Normalize { .. } => crate::TransformSpec::Normalize {
        mean: [0.0; 3],
        std: [1.0; 3],
        max_pixel_value: 255.0,
        p: 1.0,
    },
    ToTorch: TransformPlan::ToTorch => crate::TransformSpec::ToTorch,
}

#[cfg(test)]
pub(crate) fn valid_fixtures() -> Vec<crate::TransformSpec> {
    TRANSFORM_TAGS.iter().map(|tag| tag.fixture()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registration_has_a_valid_fixture_and_matching_tag() {
        for (tag, fixture) in TRANSFORM_TAGS.iter().zip(valid_fixtures()) {
            let plans = TransformPlan::compile(vec![fixture]).unwrap();
            assert_eq!(plans[0].tag(), *tag);
            assert_eq!(plans[0].name(), tag.name());
        }
    }
}
