use crate::plan::TransformPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputContract {
    ShapePreserving,
    StaticallySized,
    SampleSized,
    TerminalType,
    TerminalLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadFootprint {
    Pointwise,
    Neighborhood,
    IrregularResampling,
    GlobalReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteCoverage {
    FullOverwrite,
    PartialUpdate,
    TerminalConversion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionForm {
    BorrowedToOwned,
    OwnedToOwned,
    OwnedInPlace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReusableScratchSlot {
    U8,
    U16,
    NoiseBlock,
    AxisRemap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScratchSizing {
    CurrentImage,
    SampledCrop,
    FixedBlock,
    CurrentAxes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScratchRequirement {
    None,
    Reusable {
        slot: ReusableScratchSlot,
        sizing: ScratchSizing,
    },
    Stack {
        bytes: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumericBarrier {
    Rounding,
    Clipping,
    BorderInterpolation,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransformCapabilities {
    pub output: OutputContract,
    pub read: ReadFootprint,
    pub write: WriteCoverage,
    pub legal_forms: &'static [ExecutionForm],
    pub scratch: ScratchRequirement,
    pub barriers: &'static [NumericBarrier],
}

const OWNED_IN_PLACE: &[ExecutionForm] = &[ExecutionForm::OwnedInPlace];
const DIRECT_OR_IN_PLACE: &[ExecutionForm] =
    &[ExecutionForm::BorrowedToOwned, ExecutionForm::OwnedInPlace];
const OUT_OF_PLACE: &[ExecutionForm] =
    &[ExecutionForm::BorrowedToOwned, ExecutionForm::OwnedToOwned];
const NO_BARRIERS: &[NumericBarrier] = &[];
const ROUNDING: &[NumericBarrier] = &[NumericBarrier::Rounding];
const BORDER_INTERPOLATION: &[NumericBarrier] = &[NumericBarrier::BorderInterpolation];
const ROUNDING_AND_BORDER: &[NumericBarrier] = &[
    NumericBarrier::Rounding,
    NumericBarrier::BorderInterpolation,
];
const ROUNDING_AND_CLIPPING: &[NumericBarrier] =
    &[NumericBarrier::Rounding, NumericBarrier::Clipping];
const ROUNDING_CLIPPING_AND_BORDER: &[NumericBarrier] = &[
    NumericBarrier::Rounding,
    NumericBarrier::Clipping,
    NumericBarrier::BorderInterpolation,
];
const CLIPPING_AND_REDUCTION: &[NumericBarrier] =
    &[NumericBarrier::Clipping, NumericBarrier::Reduction];

impl TransformPlan {
    pub(crate) fn capabilities(&self) -> TransformCapabilities {
        match self {
            Self::Resize { .. } => TransformCapabilities {
                output: OutputContract::StaticallySized,
                read: ReadFootprint::Neighborhood,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::None,
                barriers: BORDER_INTERPOLATION,
            },
            Self::RandomCrop { .. } | Self::CenterCrop { .. } => TransformCapabilities {
                output: OutputContract::StaticallySized,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::None,
                barriers: NO_BARRIERS,
            },
            Self::RandomResizedCrop { .. } => TransformCapabilities {
                output: OutputContract::StaticallySized,
                read: ReadFootprint::Neighborhood,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::Reusable {
                    slot: ReusableScratchSlot::U8,
                    sizing: ScratchSizing::SampledCrop,
                },
                barriers: BORDER_INTERPOLATION,
            },
            Self::HorizontalFlip { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OWNED_IN_PLACE,
                scratch: ScratchRequirement::None,
                barriers: NO_BARRIERS,
            },
            Self::VerticalFlip { .. }
            | Self::Invert { .. }
            | Self::Solarize { .. }
            | Self::Posterize { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::FullOverwrite,
                legal_forms: DIRECT_OR_IN_PLACE,
                scratch: ScratchRequirement::None,
                barriers: NO_BARRIERS,
            },
            Self::PadIfNeeded { .. } => TransformCapabilities {
                output: OutputContract::SampleSized,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::None,
                barriers: BORDER_INTERPOLATION,
            },
            Self::CoarseDropout { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::PartialUpdate,
                legal_forms: OWNED_IN_PLACE,
                scratch: ScratchRequirement::None,
                barriers: NO_BARRIERS,
            },
            Self::ColorJitter { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::GlobalReduction,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OWNED_IN_PLACE,
                scratch: ScratchRequirement::None,
                barriers: CLIPPING_AND_REDUCTION,
            },
            Self::Affine { .. } | Self::RandomRotation { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::IrregularResampling,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::Stack {
                    bytes: std::mem::size_of::<[f32; 6]>(),
                },
                barriers: ROUNDING_AND_BORDER,
            },
            Self::GaussianNoise { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OWNED_IN_PLACE,
                scratch: ScratchRequirement::Reusable {
                    slot: ReusableScratchSlot::NoiseBlock,
                    sizing: ScratchSizing::FixedBlock,
                },
                barriers: ROUNDING_AND_CLIPPING,
            },
            Self::Sharpen { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::Neighborhood,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::None,
                barriers: ROUNDING_CLIPPING_AND_BORDER,
            },
            Self::Perspective { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::IrregularResampling,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::Stack {
                    bytes: std::mem::size_of::<[f32; 9]>(),
                },
                barriers: ROUNDING_AND_BORDER,
            },
            Self::GridDistortion { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::IrregularResampling,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::Reusable {
                    slot: ReusableScratchSlot::AxisRemap,
                    sizing: ScratchSizing::CurrentAxes,
                },
                barriers: ROUNDING_AND_BORDER,
            },
            Self::GaussianBlur { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::Neighborhood,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OWNED_IN_PLACE,
                scratch: ScratchRequirement::Reusable {
                    slot: ReusableScratchSlot::U16,
                    sizing: ScratchSizing::CurrentImage,
                },
                barriers: ROUNDING_AND_BORDER,
            },
            Self::Grayscale { .. } => TransformCapabilities {
                output: OutputContract::ShapePreserving,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::FullOverwrite,
                legal_forms: OWNED_IN_PLACE,
                scratch: ScratchRequirement::None,
                barriers: ROUNDING,
            },
            Self::Normalize { .. } => TransformCapabilities {
                output: OutputContract::TerminalType,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::TerminalConversion,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::None,
                barriers: ROUNDING,
            },
            Self::ToTorch => TransformCapabilities {
                output: OutputContract::TerminalLayout,
                read: ReadFootprint::Pointwise,
                write: WriteCoverage::TerminalConversion,
                legal_forms: OUT_OF_PLACE,
                scratch: ScratchRequirement::None,
                barriers: NO_BARRIERS,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Interpolation, TransformSpec};

    #[test]
    fn classification_distinguishes_semantics_from_execution_forms() {
        let plans = TransformPlan::compile(vec![TransformSpec::Resize {
            height: 7,
            width: 11,
            interpolation: Interpolation::Bilinear,
            antialias: false,
            p: 1.0,
        }])
        .unwrap();
        let capabilities = plans[0].capabilities();
        assert_eq!(capabilities.output, OutputContract::StaticallySized);
        assert_eq!(capabilities.read, ReadFootprint::Neighborhood);
        assert_eq!(capabilities.write, WriteCoverage::FullOverwrite);
        assert_eq!(capabilities.legal_forms, OUT_OF_PLACE);
    }
}
