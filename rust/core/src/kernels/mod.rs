use crate::capability::ExecutionForm;
use crate::plan::TransformPlan;

pub(crate) mod affine;
pub(crate) mod blur;
pub(crate) mod color;
pub(crate) mod layout;
pub(crate) mod noise;
pub(crate) mod pad;
pub(crate) mod point;
pub(crate) mod remap;
pub(crate) mod sharpen;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelImplementation {
    pub form: ExecutionForm,
    pub unit_specialization: Option<&'static str>,
}

const IN_PLACE: &[KernelImplementation] = &[KernelImplementation {
    form: ExecutionForm::OwnedInPlace,
    unit_specialization: None,
}];
const DIRECT_OR_IN_PLACE: &[KernelImplementation] = &[
    KernelImplementation {
        form: ExecutionForm::BorrowedToOwned,
        unit_specialization: None,
    },
    KernelImplementation {
        form: ExecutionForm::OwnedInPlace,
        unit_specialization: None,
    },
];
const OUT_OF_PLACE: &[KernelImplementation] = &[
    KernelImplementation {
        form: ExecutionForm::BorrowedToOwned,
        unit_specialization: None,
    },
    KernelImplementation {
        form: ExecutionForm::OwnedToOwned,
        unit_specialization: None,
    },
];
const AFFINE_OUT_OF_PLACE: &[KernelImplementation] = &[
    KernelImplementation {
        form: ExecutionForm::BorrowedToOwned,
        unit_specialization: None,
    },
    KernelImplementation {
        form: ExecutionForm::OwnedToOwned,
        unit_specialization: None,
    },
];
const COLOR_MATRIX_IN_PLACE: &[KernelImplementation] = &[KernelImplementation {
    form: ExecutionForm::OwnedInPlace,
    unit_specialization: Some("ColorJitter:composed-color-matrix"),
}];

pub(crate) fn implementations(transform: &TransformPlan) -> &'static [KernelImplementation] {
    match transform {
        TransformPlan::Resize { .. }
        | TransformPlan::RandomCrop { .. }
        | TransformPlan::RandomResizedCrop { .. }
        | TransformPlan::CenterCrop { .. }
        | TransformPlan::PadIfNeeded { .. }
        | TransformPlan::Normalize { .. }
        | TransformPlan::Sharpen { .. }
        | TransformPlan::Perspective { .. }
        | TransformPlan::GridDistortion { .. } => OUT_OF_PLACE,
        TransformPlan::HorizontalFlip { .. }
        | TransformPlan::CoarseDropout { .. }
        | TransformPlan::GaussianBlur { .. }
        | TransformPlan::GaussianNoise { .. }
        | TransformPlan::Grayscale { .. } => IN_PLACE,
        TransformPlan::VerticalFlip { .. }
        | TransformPlan::Invert { .. }
        | TransformPlan::Solarize { .. }
        | TransformPlan::Posterize { .. } => DIRECT_OR_IN_PLACE,
        TransformPlan::ColorJitter { hue, .. } if *hue == [0.0, 0.0] => COLOR_MATRIX_IN_PLACE,
        TransformPlan::ColorJitter { .. } => IN_PLACE,
        TransformPlan::Affine { .. } | TransformPlan::RandomRotation { .. } => AFFINE_OUT_OF_PLACE,
    }
}

const GRAY_COLOR_IN_PLACE: &[KernelImplementation] = &[KernelImplementation {
    form: ExecutionForm::OwnedInPlace,
    unit_specialization: Some("ColorJitter:gray-q14-composition"),
}];
const GRAY_NORMALIZE: &[KernelImplementation] = &[
    KernelImplementation {
        form: ExecutionForm::BorrowedToOwned,
        unit_specialization: Some("single-channel-normalize"),
    },
    KernelImplementation {
        form: ExecutionForm::OwnedToOwned,
        unit_specialization: Some("single-channel-normalize"),
    },
];

pub(crate) fn channel_implementations(
    transform: &TransformPlan,
    channels: usize,
) -> &'static [KernelImplementation] {
    if channels == 1 {
        match transform {
            TransformPlan::ColorJitter { hue, .. } if *hue == [0.0, 0.0] => {
                return GRAY_COLOR_IN_PLACE
            }
            TransformPlan::Normalize { .. } => return GRAY_NORMALIZE,
            _ => {}
        }
    }
    implementations(transform)
}
