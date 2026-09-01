#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Bilinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    Constant,
    Reflect101,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadPosition {
    Center,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DropoutSizeRange {
    Fraction([f32; 2]),
    Pixels([usize; 2]),
}

#[derive(Debug, Clone)]
pub enum TransformSpec {
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

pub struct PipelineSpec {
    transforms: Vec<TransformSpec>,
}

impl PipelineSpec {
    pub fn new(transforms: Vec<TransformSpec>) -> Self {
        Self { transforms }
    }

    pub(crate) fn into_transforms(self) -> Vec<TransformSpec> {
        self.transforms
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Reference,
    Compiled,
    StagedFresh,
    StagedReuse,
}

impl ExecutionMode {
    pub fn is_compiled(self) -> bool {
        self == Self::Compiled
    }
}

pub enum PipelineOutput {
    U8Hwc {
        data: Vec<u8>,
        height: usize,
        width: usize,
    },
    F32Hwc {
        data: Vec<f32>,
        height: usize,
        width: usize,
    },
    U8Chw {
        data: Vec<u8>,
        height: usize,
        width: usize,
    },
    F32Chw {
        data: Vec<f32>,
        height: usize,
        width: usize,
    },
}

#[derive(Clone)]
pub struct PipelineExplanation {
    pub mode: &'static str,
    pub sampling: &'static str,
    pub transforms: Vec<&'static str>,
    pub steps: Vec<TransformExplanation>,
    pub fusions: Vec<&'static str>,
    pub unit_specializations: Vec<&'static str>,
    pub optimizations: Vec<&'static str>,
    pub passes: usize,
    pub pixel_passes: usize,
    pub output_dtype: &'static str,
    pub output_layout: &'static str,
    pub input: ImageContractExplanation,
    pub output: ImageContractExplanation,
    pub buffers: Vec<BufferExplanation>,
    pub copies: Vec<CopyExplanation>,
    pub fallbacks: Vec<&'static str>,
}

#[derive(Clone)]
pub struct TransformExplanation {
    pub name: &'static str,
    pub category: &'static str,
    pub probability: f32,
    pub status: &'static str,
    pub execution: &'static str,
    pub pixel_passes: usize,
    pub allocation: &'static str,
    pub fallback: &'static str,
    pub input_materialization: &'static str,
    pub kernel_form: &'static str,
    pub output_slot: &'static str,
    pub scratch_slots: Vec<&'static str>,
    pub selection_reason: &'static str,
    pub policies: Vec<PolicyExplanation>,
}

#[derive(Clone)]
pub struct PolicyExplanation {
    pub name: &'static str,
    pub value: String,
}

#[derive(Clone)]
pub struct ImageContractExplanation {
    pub container: &'static str,
    pub dtype: &'static str,
    pub layout: &'static str,
    pub channels: &'static str,
    pub contiguous: bool,
    pub ownership: &'static str,
}

#[derive(Clone)]
pub struct BufferExplanation {
    pub name: &'static str,
    pub dtype: &'static str,
    pub layout: &'static str,
    pub lifecycle: &'static str,
    pub condition: &'static str,
}

#[derive(Clone)]
pub struct CopyExplanation {
    pub stage: &'static str,
    pub count: &'static str,
    pub condition: &'static str,
    pub reason: &'static str,
}
