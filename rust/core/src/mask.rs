use crate::operations::{copy_u8, inverse_affine_matrix, reflect101_index, source_coordinates};
use crate::plan::{
    AffineSample, CropSample, GridDistortionSample, SampledTransform, TransformPlan,
};
use crate::{
    BorderMode, BufferExplanation, CopyExplanation, CoreError, CoreResult, MaskOutput,
    MaskPlanExplanation, MaskTransformExplanation, Workspace,
};
use fast_image_resize as fir;
use fir::images::{Image as FirImage, ImageRef as FirImageRef};

pub(crate) struct MaskPlan {
    steps: Vec<MaskPlannedStep>,
}

struct MaskPlannedStep {
    name: &'static str,
    probability: f32,
    operation: MaskStep,
}

#[derive(Clone, Copy)]
enum MaskStep {
    Resize,
    RandomCrop,
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    PadIfNeeded { border: BorderMode },
    Affine { border: BorderMode },
    RandomRotation { border: BorderMode },
    Perspective { border: BorderMode },
    GridDistortion { border: BorderMode },
    ImageOnly,
}

struct MaskImage {
    data: Vec<u8>,
    height: usize,
    width: usize,
}

pub(crate) fn mask_len(height: usize, width: usize) -> CoreResult<usize> {
    height
        .checked_mul(width)
        .ok_or_else(|| CoreError::Invalid("mask dimensions overflow".into()))
}

impl MaskPlan {
    pub(crate) fn compile(transforms: &[TransformPlan]) -> CoreResult<Self> {
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(transforms.len())
            .map_err(|_| CoreError::Runtime("mask plan allocation failed".into()))?;
        for transform in transforms {
            let operation = match transform {
                TransformPlan::Resize { .. } => MaskStep::Resize,
                TransformPlan::RandomCrop { .. } => MaskStep::RandomCrop,
                TransformPlan::RandomResizedCrop { .. } => MaskStep::RandomResizedCrop,
                TransformPlan::HorizontalFlip { .. } => MaskStep::HorizontalFlip,
                TransformPlan::VerticalFlip { .. } => MaskStep::VerticalFlip,
                TransformPlan::CenterCrop { .. } => MaskStep::CenterCrop,
                TransformPlan::PadIfNeeded { border_mode, .. } => MaskStep::PadIfNeeded {
                    border: *border_mode,
                },
                TransformPlan::Affine { border_mode, .. } => MaskStep::Affine {
                    border: *border_mode,
                },
                TransformPlan::RandomRotation { border_mode, .. } => MaskStep::RandomRotation {
                    border: *border_mode,
                },
                TransformPlan::Perspective { border_mode, .. } => MaskStep::Perspective {
                    border: *border_mode,
                },
                TransformPlan::GridDistortion { border_mode, .. } => MaskStep::GridDistortion {
                    border: *border_mode,
                },
                TransformPlan::CoarseDropout { .. }
                | TransformPlan::ColorJitter { .. }
                | TransformPlan::GaussianNoise { .. }
                | TransformPlan::Sharpen { .. }
                | TransformPlan::GaussianBlur { .. }
                | TransformPlan::Grayscale { .. }
                | TransformPlan::Invert { .. }
                | TransformPlan::Solarize { .. }
                | TransformPlan::Posterize { .. }
                | TransformPlan::Normalize { .. } => MaskStep::ImageOnly,
            };
            steps.push(MaskPlannedStep {
                name: transform.name(),
                probability: transform.probability(),
                operation,
            });
        }
        Ok(Self { steps })
    }

    pub(crate) fn apply(
        &self,
        data: &[u8],
        dimensions: (usize, usize),
        sampled: &[SampledTransform],
        fill: u8,
        workspace: &mut Workspace,
        reuse: bool,
    ) -> CoreResult<MaskOutput> {
        let (height, width) = dimensions;
        self.apply_image(
            MaskImage::copy_from(data, height, width)?,
            sampled,
            fill,
            workspace,
            reuse,
        )
    }

    pub(crate) fn apply_owned(
        &self,
        data: Vec<u8>,
        dimensions: (usize, usize),
        sampled: &[SampledTransform],
        fill: u8,
        workspace: &mut Workspace,
        reuse: bool,
    ) -> CoreResult<MaskOutput> {
        let (height, width) = dimensions;
        self.apply_image(
            MaskImage::from_owned(data, height, width)?,
            sampled,
            fill,
            workspace,
            reuse,
        )
    }

    fn apply_image(
        &self,
        mut mask: MaskImage,
        sampled: &[SampledTransform],
        fill: u8,
        workspace: &mut Workspace,
        reuse: bool,
    ) -> CoreResult<MaskOutput> {
        if sampled.len() != self.steps.len() {
            return Err(CoreError::Runtime(
                "sampled mask plan does not match pipeline".into(),
            ));
        }
        for (step, sampled) in self.steps.iter().zip(sampled) {
            if matches!(sampled, SampledTransform::Skip) {
                continue;
            }
            match (step.operation, sampled) {
                (MaskStep::Resize, SampledTransform::Resize { height, width, .. }) => {
                    mask = resize_mask(mask, *height, *width, workspace, reuse)?;
                }
                (MaskStep::RandomCrop, SampledTransform::RandomCrop(crop))
                | (MaskStep::CenterCrop, SampledTransform::CenterCrop(crop)) => {
                    mask = crop_mask(mask, *crop, workspace, reuse)?;
                }
                (
                    MaskStep::RandomResizedCrop,
                    SampledTransform::RandomResizedCrop {
                        crop,
                        height,
                        width,
                        ..
                    },
                ) => {
                    mask = crop_mask(mask, *crop, workspace, reuse)?;
                    mask = resize_mask(mask, *height, *width, workspace, reuse)?;
                }
                (MaskStep::HorizontalFlip, SampledTransform::HorizontalFlip) => {
                    horizontal_flip(&mut mask)
                }
                (MaskStep::VerticalFlip, SampledTransform::VerticalFlip) => {
                    vertical_flip(&mut mask)
                }
                (MaskStep::PadIfNeeded { border }, SampledTransform::PadIfNeeded(sample)) => {
                    if sample.height != mask.height || sample.width != mask.width {
                        mask = pad_mask(mask, *sample, border, fill, workspace, reuse)?;
                    }
                }
                (MaskStep::Affine { border }, SampledTransform::Affine(sample)) => {
                    mask = affine_mask(mask, *sample, border, fill, workspace, reuse)?;
                }
                (MaskStep::RandomRotation { border }, SampledTransform::RandomRotation(sample)) => {
                    let affine = AffineSample {
                        degrees: sample.degrees,
                        translate: [0.0, 0.0],
                        scale: 1.0,
                        shear: [0.0, 0.0],
                    };
                    mask = affine_mask(mask, affine, border, fill, workspace, reuse)?;
                }
                (MaskStep::Perspective { border }, SampledTransform::Perspective(sample)) => {
                    mask = perspective_mask(mask, sample.inverse, border, fill, workspace, reuse)?;
                }
                (MaskStep::GridDistortion { border }, SampledTransform::GridDistortion(sample)) => {
                    mask = grid_mask(mask, sample, border, fill, workspace, reuse)?;
                }
                (MaskStep::ImageOnly, sampled) if is_image_only_sample(sampled) => {}
                _ => {
                    return Err(CoreError::Runtime(
                        "sampled mask plan does not match pipeline".into(),
                    ));
                }
            }
        }
        Ok(MaskOutput {
            data: mask.data,
            height: mask.height,
            width: mask.width,
        })
    }

    pub(crate) fn explain(&self, fill: u8) -> MaskPlanExplanation {
        let steps: Vec<_> = self
            .steps
            .iter()
            .map(|step| {
                let active = step.probability > 0.0;
                MaskTransformExplanation {
                    name: step.name,
                    classification: if matches!(step.operation, MaskStep::ImageOnly) {
                        "image-only"
                    } else {
                        "geometric"
                    },
                    raster_policy: step.operation.raster_policy(),
                    pixel_passes: if active {
                        step.operation.pixel_passes()
                    } else {
                        0
                    },
                    fill: step.operation.uses_fill().then_some(fill),
                }
            })
            .collect();
        let has_scratch = self.steps.iter().any(|step| {
            step.probability > 0.0
                && matches!(
                    step.operation,
                    MaskStep::Resize
                        | MaskStep::RandomCrop
                        | MaskStep::RandomResizedCrop
                        | MaskStep::CenterCrop
                        | MaskStep::PadIfNeeded { .. }
                        | MaskStep::Affine { .. }
                        | MaskStep::RandomRotation { .. }
                        | MaskStep::Perspective { .. }
                        | MaskStep::GridDistortion { .. }
                )
        });
        let mut buffers = vec![BufferExplanation {
            name: "mask-input",
            dtype: "uint8",
            layout: "HW",
            lifecycle: "borrowed-for-call",
            condition: "mask-present",
        }];
        buffers.push(BufferExplanation {
            name: "mask-working-output",
            dtype: "uint8",
            layout: "HW",
            lifecycle: "owned-output",
            condition: "mask-present",
        });
        if has_scratch {
            buffers.push(BufferExplanation {
                name: "mask-scratch",
                dtype: "uint8",
                layout: "HW",
                lifecycle: "owned-per-run-workspace-reusable",
                condition: "geometric-step-applied",
            });
        }
        MaskPlanExplanation {
            supported: true,
            input_dtype: "uint8",
            input_layout: "HW",
            output_dtype: "uint8",
            output_layout: "HW",
            contiguous: true,
            ownership: "result",
            gil: "held-during-augmentation",
            pixel_passes: 1 + steps.iter().map(|step| step.pixel_passes).sum::<usize>(),
            steps,
            copies: vec![CopyExplanation {
                stage: "mask-native-entry",
                count: "1",
                condition: "mask-present",
                reason: "establish-owned-mask-output",
            }],
            buffers,
            fallback: "portable-scalar+nearest-resizer",
        }
    }
}

impl MaskStep {
    fn raster_policy(self) -> &'static str {
        match self {
            Self::Resize | Self::RandomResizedCrop => "nearest-no-antialias",
            Self::RandomCrop | Self::CenterCrop => "shared-crop",
            Self::HorizontalFlip | Self::VerticalFlip => "shared-flip",
            Self::PadIfNeeded {
                border: BorderMode::Constant,
            } => "shared-offsets-constant-scalar-fill",
            Self::PadIfNeeded {
                border: BorderMode::Reflect101,
            } => "shared-offsets-reflect101",
            Self::Affine {
                border: BorderMode::Constant,
            }
            | Self::RandomRotation {
                border: BorderMode::Constant,
            }
            | Self::Perspective {
                border: BorderMode::Constant,
            }
            | Self::GridDistortion {
                border: BorderMode::Constant,
            } => "shared-geometry-nearest-constant-scalar-fill",
            Self::Affine {
                border: BorderMode::Reflect101,
            }
            | Self::RandomRotation {
                border: BorderMode::Reflect101,
            }
            | Self::Perspective {
                border: BorderMode::Reflect101,
            }
            | Self::GridDistortion {
                border: BorderMode::Reflect101,
            } => "shared-geometry-nearest-reflect101",
            Self::ImageOnly => "not-applied",
        }
    }

    fn pixel_passes(self) -> usize {
        match self {
            Self::RandomResizedCrop => 2,
            Self::ImageOnly => 0,
            _ => 1,
        }
    }

    fn uses_fill(self) -> bool {
        matches!(
            self,
            Self::PadIfNeeded { .. }
                | Self::Affine { .. }
                | Self::RandomRotation { .. }
                | Self::Perspective { .. }
                | Self::GridDistortion { .. }
        )
    }
}

impl MaskImage {
    fn copy_from(data: &[u8], height: usize, width: usize) -> CoreResult<Self> {
        let expected = mask_len(height, width)?;
        if height == 0 || width == 0 || data.len() != expected {
            return Err(CoreError::Invalid(
                "expected a non-empty HW uint8 mask buffer".into(),
            ));
        }
        Ok(Self {
            data: copy_u8(data)?,
            height,
            width,
        })
    }

    fn from_owned(data: Vec<u8>, height: usize, width: usize) -> CoreResult<Self> {
        let expected = mask_len(height, width)?;
        if height == 0 || width == 0 || data.len() != expected {
            return Err(CoreError::Invalid(
                "expected a non-empty HW uint8 mask buffer".into(),
            ));
        }
        Ok(Self {
            data,
            height,
            width,
        })
    }
}

fn is_image_only_sample(sampled: &SampledTransform) -> bool {
    matches!(
        sampled,
        SampledTransform::CoarseDropout { .. }
            | SampledTransform::ColorJitter(_)
            | SampledTransform::GaussianNoise(_)
            | SampledTransform::Sharpen(_)
            | SampledTransform::GaussianBlur { .. }
            | SampledTransform::Grayscale
            | SampledTransform::Invert
            | SampledTransform::Solarize { .. }
            | SampledTransform::Posterize { .. }
            | SampledTransform::Normalize
    )
}

fn crop_mask(
    mask: MaskImage,
    crop: CropSample,
    workspace: &mut Workspace,
    reuse: bool,
) -> CoreResult<MaskImage> {
    if crop
        .top
        .checked_add(crop.height)
        .is_none_or(|end| end > mask.height)
        || crop
            .left
            .checked_add(crop.width)
            .is_none_or(|end| end > mask.width)
    {
        return Err(CoreError::Runtime("sampled crop exceeds mask input".into()));
    }
    let mut output = workspace.take_staged_u8(mask_len(crop.height, crop.width)?, false, reuse)?;
    for y in 0..crop.height {
        let source = (crop.top + y) * mask.width + crop.left;
        let destination = y * crop.width;
        output[destination..destination + crop.width]
            .copy_from_slice(&mask.data[source..source + crop.width]);
    }
    workspace.recycle_staged_u8(mask.data, reuse);
    Ok(MaskImage {
        data: output,
        height: crop.height,
        width: crop.width,
    })
}

fn resize_mask(
    mask: MaskImage,
    height: usize,
    width: usize,
    workspace: &mut Workspace,
    reuse: bool,
) -> CoreResult<MaskImage> {
    let destination = workspace.take_staged_u8(mask_len(height, width)?, false, reuse)?;
    let source = FirImageRef::new(
        mask.width as u32,
        mask.height as u32,
        &mask.data,
        fir::PixelType::U8,
    )
    .map_err(|error| CoreError::Invalid(format!("invalid mask buffer: {error}")))?;
    let mut output =
        FirImage::from_vec_u8(width as u32, height as u32, destination, fir::PixelType::U8)
            .map_err(|error| {
                CoreError::Runtime(format!("invalid mask resize destination: {error}"))
            })?;
    let options = fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Nearest);
    workspace
        .resizer()
        .resize(&source, &mut output, &options)
        .map_err(|error| CoreError::Runtime(format!("mask resize failed: {error}")))?;
    workspace.recycle_staged_u8(mask.data, reuse);
    Ok(MaskImage {
        data: output.into_vec(),
        height,
        width,
    })
}

fn horizontal_flip(mask: &mut MaskImage) {
    for row in mask.data.chunks_exact_mut(mask.width) {
        row.reverse();
    }
}

fn vertical_flip(mask: &mut MaskImage) {
    for y in 0..mask.height / 2 {
        let opposite = mask.height - 1 - y;
        for x in 0..mask.width {
            mask.data
                .swap(y * mask.width + x, opposite * mask.width + x);
        }
    }
}

fn pad_mask(
    mask: MaskImage,
    sample: crate::plan::PadSample,
    border: BorderMode,
    fill: u8,
    workspace: &mut Workspace,
    reuse: bool,
) -> CoreResult<MaskImage> {
    if sample
        .top
        .checked_add(mask.height)
        .is_none_or(|end| end > sample.height)
        || sample
            .left
            .checked_add(mask.width)
            .is_none_or(|end| end > sample.width)
    {
        return Err(CoreError::Runtime(
            "invalid sampled mask padding geometry".into(),
        ));
    }
    let mut output =
        workspace.take_staged_u8(mask_len(sample.height, sample.width)?, false, reuse)?;
    for y in 0..sample.height {
        for x in 0..sample.width {
            let source_y = isize::try_from(y)
                .ok()
                .and_then(|value| value.checked_sub(isize::try_from(sample.top).ok()?));
            let source_x = isize::try_from(x)
                .ok()
                .and_then(|value| value.checked_sub(isize::try_from(sample.left).ok()?));
            output[y * sample.width + x] = match (source_y, source_x) {
                (Some(source_y), Some(source_x)) => sample_mask(
                    &mask.data,
                    mask.height,
                    mask.width,
                    source_x,
                    source_y,
                    border,
                    fill,
                ),
                _ if border == BorderMode::Constant => fill,
                _ => {
                    return Err(CoreError::Invalid("padding dimensions overflow".into()));
                }
            };
        }
    }
    workspace.recycle_staged_u8(mask.data, reuse);
    Ok(MaskImage {
        data: output,
        height: sample.height,
        width: sample.width,
    })
}

fn affine_mask(
    mask: MaskImage,
    sample: AffineSample,
    border: BorderMode,
    fill: u8,
    workspace: &mut Workspace,
    reuse: bool,
) -> CoreResult<MaskImage> {
    if mask.height > crate::operations::MAX_AFFINE_DIMENSION
        || mask.width > crate::operations::MAX_AFFINE_DIMENSION
    {
        return Err(CoreError::Invalid(format!(
            "Affine mask dimensions must not exceed {} per axis",
            crate::operations::MAX_AFFINE_DIMENSION
        )));
    }
    let matrix = inverse_affine_matrix(mask.width, mask.height, sample);
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(CoreError::Invalid(
            "Affine parameters produce non-finite coordinates".into(),
        ));
    }
    let mut output = workspace.take_staged_u8(mask.data.len(), false, reuse)?;
    for y in 0..mask.height {
        let (source_x, source_y, delta_x, delta_y) = source_coordinates(y, matrix);
        for x in 0..mask.width {
            let sx = (source_x + delta_x * x as f64).round() as isize;
            let sy = (source_y + delta_y * x as f64).round() as isize;
            output[y * mask.width + x] =
                sample_mask(&mask.data, mask.height, mask.width, sx, sy, border, fill);
        }
    }
    workspace.recycle_staged_u8(mask.data, reuse);
    Ok(MaskImage {
        data: output,
        height: mask.height,
        width: mask.width,
    })
}

fn perspective_mask(
    mask: MaskImage,
    inverse: [f32; 9],
    border: BorderMode,
    fill: u8,
    workspace: &mut Workspace,
    reuse: bool,
) -> CoreResult<MaskImage> {
    let mut output = workspace.take_staged_u8(mask.data.len(), false, reuse)?;
    for y in 0..mask.height {
        for x in 0..mask.width {
            let x_value = x as f32;
            let y_value = y as f32;
            let denominator = inverse[6] * x_value + inverse[7] * y_value + inverse[8];
            let value = if !denominator.is_finite() || denominator.abs() < 1e-8 {
                fill
            } else {
                let source_x =
                    (inverse[0] * x_value + inverse[1] * y_value + inverse[2]) / denominator;
                let source_y =
                    (inverse[3] * x_value + inverse[4] * y_value + inverse[5]) / denominator;
                if source_x.is_finite() && source_y.is_finite() {
                    sample_mask(
                        &mask.data,
                        mask.height,
                        mask.width,
                        source_x.round() as isize,
                        source_y.round() as isize,
                        border,
                        fill,
                    )
                } else {
                    fill
                }
            };
            output[y * mask.width + x] = value;
        }
    }
    workspace.recycle_staged_u8(mask.data, reuse);
    Ok(MaskImage {
        data: output,
        height: mask.height,
        width: mask.width,
    })
}

fn grid_mask(
    mask: MaskImage,
    sample: &GridDistortionSample,
    border: BorderMode,
    fill: u8,
    workspace: &mut Workspace,
    reuse: bool,
) -> CoreResult<MaskImage> {
    if sample.x_map.len() != mask.width || sample.y_map.len() != mask.height {
        return Err(CoreError::Runtime(
            "grid maps must match the mask dimensions".into(),
        ));
    }
    let mut output = workspace.take_staged_u8(mask.data.len(), false, reuse)?;
    for (y, source_y) in sample.y_map.iter().enumerate() {
        for (x, source_x) in sample.x_map.iter().enumerate() {
            output[y * mask.width + x] = sample_mask(
                &mask.data,
                mask.height,
                mask.width,
                source_x.round() as isize,
                source_y.round() as isize,
                border,
                fill,
            );
        }
    }
    workspace.recycle_staged_u8(mask.data, reuse);
    Ok(MaskImage {
        data: output,
        height: mask.height,
        width: mask.width,
    })
}

fn sample_mask(
    data: &[u8],
    height: usize,
    width: usize,
    x: isize,
    y: isize,
    border: BorderMode,
    fill: u8,
) -> u8 {
    let (x, y) = match border {
        BorderMode::Constant if x < 0 || y < 0 || x >= width as isize || y >= height as isize => {
            return fill;
        }
        BorderMode::Constant => (x as usize, y as usize),
        BorderMode::Reflect101 => (reflect101_index(x, width), reflect101_index(y, height)),
    };
    data[y * width + x]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_flip_and_pad_preserve_labels() {
        let mask = MaskImage::copy_from(&[0, 1, 2, 3, 254, 255], 2, 3).unwrap();
        let mut workspace = Workspace::default();
        let mut cropped = crop_mask(
            mask,
            CropSample {
                top: 0,
                left: 1,
                height: 2,
                width: 2,
            },
            &mut workspace,
            true,
        )
        .unwrap();
        horizontal_flip(&mut cropped);
        assert_eq!(cropped.data, [2, 1, 255, 254]);
        let padded = pad_mask(
            cropped,
            crate::plan::PadSample {
                top: 1,
                left: 1,
                height: 4,
                width: 4,
            },
            BorderMode::Constant,
            17,
            &mut workspace,
            true,
        )
        .unwrap();
        assert_eq!(
            padded.data,
            [17, 17, 17, 17, 17, 2, 1, 17, 17, 255, 254, 17, 17, 17, 17, 17]
        );
    }

    #[test]
    fn resize_uses_only_existing_labels() {
        let mask = MaskImage::copy_from(&[0, 255, 7, 19], 2, 2).unwrap();
        let resized = resize_mask(mask, 7, 9, &mut Workspace::default(), true).unwrap();
        assert!(resized
            .data
            .iter()
            .all(|value| [0, 7, 19, 255].contains(value)));
    }

    #[test]
    fn nearest_affine_uses_scalar_fill_and_reflection() {
        let sample = AffineSample {
            degrees: 0.0,
            translate: [1.0, 0.0],
            scale: 1.0,
            shear: [0.0, 0.0],
        };
        let constant = affine_mask(
            MaskImage::copy_from(&[1, 2, 3], 1, 3).unwrap(),
            sample,
            BorderMode::Constant,
            17,
            &mut Workspace::default(),
            true,
        )
        .unwrap();
        assert_eq!(constant.data, [17, 1, 2]);
        let reflected = affine_mask(
            MaskImage::copy_from(&[1, 2, 3], 1, 3).unwrap(),
            sample,
            BorderMode::Reflect101,
            17,
            &mut Workspace::default(),
            true,
        )
        .unwrap();
        assert_eq!(reflected.data, [2, 1, 2]);
    }
}
