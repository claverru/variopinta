use crate::capability::{ExecutionForm, OutputContract};
use crate::kernels::layout::{hwc_to_chw, hwc_u8_to_chw, normalize_hwc, normalize_hwc_to_chw};
use crate::kernels::point;
use crate::operations::*;
use crate::optimization::{FusionRule, KernelSelection, LoweringPlan};
use crate::plan::{
    derive_run_seed, make_gaussian_kernel, AffineSample, SampledTransform, TransformPlan,
};
use crate::{
    CoreError, CoreResult, ExecutionMode, PipelineExplanation, PipelineOutput, PipelineSpec,
    Workspace,
};

pub struct CompiledPipeline {
    plan: ExecutionPlan,
}

struct ExecutionPlan {
    transforms: Vec<TransformPlan>,
    mode: ExecutionMode,
    lowering: LoweringPlan,
    explanation: PipelineExplanation,
}

impl ExecutionPlan {
    fn compile(spec: PipelineSpec, mode: ExecutionMode) -> CoreResult<Self> {
        let transforms = TransformPlan::compile(spec.into_transforms())?;
        let lowering = LoweringPlan::compile(&transforms, mode)?;
        let explanation = crate::explanation::build(&transforms, mode, &lowering);
        Ok(Self {
            lowering,
            transforms,
            mode,
            explanation,
        })
    }
}

impl CompiledPipeline {
    pub(crate) fn build(spec: PipelineSpec, mode: ExecutionMode) -> CoreResult<Self> {
        Ok(Self {
            plan: ExecutionPlan::compile(spec, mode)?,
        })
    }

    pub fn apply(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        pipeline_seed: u64,
        run_key: u64,
        workspace: &mut Workspace,
    ) -> CoreResult<PipelineOutput> {
        let expected = rgb_len(height, width)?;
        if height == 0 || width == 0 || data.len() != expected {
            return Err(CoreError::Invalid(
                "expected a non-empty HWC RGB buffer".into(),
            ));
        }
        self.run(
            data,
            height,
            width,
            derive_run_seed(pipeline_seed, run_key),
            workspace,
        )
    }

    pub fn explain(&self) -> PipelineExplanation {
        self.plan.explanation.clone()
    }
}

impl CompiledPipeline {
    fn run(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        run_seed: u64,
        workspace: &mut Workspace,
    ) -> CoreResult<PipelineOutput> {
        let sampled = TransformPlan::sample(&self.plan.transforms, height, width, run_seed)?;
        let reuse = self.plan.mode != ExecutionMode::StagedFresh;
        if self.plan.mode == ExecutionMode::Compiled {
            if let Some(output) = self.compiled_terminal_entry(data, height, width, &sampled)? {
                return Ok(output);
            }
        }
        let (mut image, start) = if self.plan.mode == ExecutionMode::Compiled {
            self.compiled_entry(data, height, width, &sampled, workspace)?
        } else {
            (
                ImageU8 {
                    data: copy_u8(data)?,
                    height,
                    width,
                },
                0,
            )
        };

        for (offset, (transform, sampled)) in self.plan.transforms[start..]
            .iter()
            .zip(&sampled[start..])
            .enumerate()
        {
            let index = start + offset;
            match (transform, sampled) {
                (_, SampledTransform::Skip) => continue,
                (
                    TransformPlan::Resize { .. },
                    SampledTransform::Resize {
                        height,
                        width,
                        interpolation,
                        antialias,
                    },
                ) => {
                    let destination =
                        workspace.take_staged_u8(rgb_len(*height, *width)?, false, reuse)?;
                    let next = resize_raw(
                        &image.data,
                        image.height,
                        image.width,
                        *height,
                        *width,
                        *interpolation,
                        *antialias,
                        workspace.resizer(),
                        destination,
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (TransformPlan::RandomCrop { .. }, SampledTransform::RandomCrop(crop)) => {
                    let destination = workspace.take_staged_u8(
                        rgb_len(crop.height, crop.width)?,
                        false,
                        reuse,
                    )?;
                    let next = random_crop_raw_into(
                        &image.data,
                        image.height,
                        image.width,
                        *crop,
                        destination,
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (
                    TransformPlan::RandomResizedCrop { .. },
                    SampledTransform::RandomResizedCrop {
                        crop,
                        height,
                        width,
                        interpolation,
                        antialias,
                    },
                ) => {
                    let crop_destination = workspace.take_staged_u8(
                        rgb_len(crop.height, crop.width)?,
                        false,
                        reuse,
                    )?;
                    let cropped = random_crop_raw_into(
                        &image.data,
                        image.height,
                        image.width,
                        *crop,
                        crop_destination,
                    )?;
                    let resize_destination =
                        workspace.take_staged_u8(rgb_len(*height, *width)?, false, reuse)?;
                    let next = resize_raw(
                        &cropped.data,
                        cropped.height,
                        cropped.width,
                        *height,
                        *width,
                        *interpolation,
                        *antialias,
                        workspace.resizer(),
                        resize_destination,
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    workspace.recycle_staged_u8(cropped.data, reuse);
                    image = next;
                }
                (TransformPlan::HorizontalFlip { .. }, SampledTransform::HorizontalFlip) => {
                    point::horizontal_flip(&mut image.data, image.height, image.width);
                }
                (TransformPlan::VerticalFlip { .. }, SampledTransform::VerticalFlip) => {
                    point::vertical_flip(&mut image.data, image.height, image.width);
                }
                (TransformPlan::CenterCrop { .. }, SampledTransform::CenterCrop(crop)) => {
                    let destination = workspace.take_staged_u8(
                        rgb_len(crop.height, crop.width)?,
                        false,
                        reuse,
                    )?;
                    let next = random_crop_raw_into(
                        &image.data,
                        image.height,
                        image.width,
                        *crop,
                        destination,
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (TransformPlan::PadIfNeeded { .. }, SampledTransform::PadIfNeeded(sample)) => {
                    if sample.height == image.height && sample.width == image.width {
                        continue;
                    }
                    let destination = workspace.take_staged_u8(
                        rgb_len(sample.height, sample.width)?,
                        false,
                        reuse,
                    )?;
                    let next =
                        pad_raw(&image.data, image.height, image.width, *sample, destination)?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (
                    TransformPlan::CoarseDropout { .. },
                    SampledTransform::CoarseDropout { holes, fill },
                ) => coarse_dropout(&mut image, holes, *fill)?,
                (TransformPlan::ColorJitter { .. }, SampledTransform::ColorJitter(sample)) => {
                    if self.plan.lowering.node(index).unit_specialization.is_some() {
                        color_jitter(&mut image, sample);
                    } else {
                        color_jitter_staged(&mut image, sample);
                    }
                }
                (TransformPlan::Affine { .. }, SampledTransform::Affine(sample)) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let next =
                        rotate_raw(&image.data, image.height, image.width, *sample, destination)?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (
                    TransformPlan::RandomRotation { .. },
                    SampledTransform::RandomRotation(sample),
                ) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let affine = AffineSample {
                        degrees: sample.degrees,
                        translate: [0.0, 0.0],
                        scale: 1.0,
                        shear: [0.0, 0.0],
                        interpolation: sample.interpolation,
                        border_mode: sample.border_mode,
                        fill: sample.fill,
                    };
                    let next =
                        rotate_raw(&image.data, image.height, image.width, affine, destination)?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (TransformPlan::GaussianNoise { .. }, SampledTransform::GaussianNoise(sample)) => {
                    gaussian_noise(&mut image, *sample, workspace.noise_block())?
                }
                (TransformPlan::Sharpen { .. }, SampledTransform::Sharpen(sample)) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let next =
                        sharpen_raw(&image.data, image.height, image.width, *sample, destination)?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (TransformPlan::Perspective { .. }, SampledTransform::Perspective(sample)) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let next = perspective_raw(
                        &image.data,
                        image.height,
                        image.width,
                        *sample,
                        destination,
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (
                    TransformPlan::GridDistortion { .. },
                    SampledTransform::GridDistortion(sample),
                ) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let next = grid_distortion_raw(
                        &image.data,
                        image.height,
                        image.width,
                        sample,
                        destination,
                        workspace.axis_remap(),
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (
                    TransformPlan::GaussianBlur {
                        kernel_size,
                        fixed_kernel,
                        ..
                    },
                    SampledTransform::GaussianBlur { sigma },
                ) => {
                    let sampled_kernel;
                    let kernel = if let Some(kernel) = fixed_kernel {
                        kernel.as_slice()
                    } else {
                        sampled_kernel = make_gaussian_kernel(*kernel_size, *sigma)?;
                        &sampled_kernel
                    };
                    if reuse {
                        gaussian_blur_in_place(&mut image, kernel, workspace.blur_temp())?;
                    } else {
                        gaussian_blur_in_place(&mut image, kernel, &mut Vec::new())?;
                    }
                }
                (TransformPlan::Grayscale { .. }, SampledTransform::Grayscale) => {
                    point::grayscale(&mut image.data);
                }
                (TransformPlan::Invert { .. }, SampledTransform::Invert) => {
                    point::invert(&mut image.data);
                }
                (TransformPlan::Solarize { .. }, SampledTransform::Solarize { threshold }) => {
                    point::solarize(&mut image.data, *threshold);
                }
                (TransformPlan::Posterize { .. }, SampledTransform::Posterize { bits }) => {
                    point::posterize(&mut image.data, *bits);
                }
                (
                    TransformPlan::Normalize {
                        mean,
                        std,
                        max_pixel_value,
                        ..
                    },
                    SampledTransform::Normalize,
                ) => {
                    let height = image.height;
                    let width = image.width;
                    if self
                        .plan
                        .lowering
                        .fusion_at(index)
                        .is_some_and(|selection| selection.is_active())
                    {
                        return Ok(PipelineOutput::F32Chw {
                            data: normalize_hwc_to_chw(
                                &image.data,
                                height,
                                width,
                                *mean,
                                *std,
                                *max_pixel_value,
                            )?,
                            height,
                            width,
                        });
                    }
                    let output = normalize_hwc(&image.data, *mean, *std, *max_pixel_value)?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    if self
                        .plan
                        .lowering
                        .nodes()
                        .get(index + 1)
                        .is_some_and(|node| {
                            node.capabilities.output == OutputContract::TerminalLayout
                        })
                    {
                        return Ok(PipelineOutput::F32Chw {
                            data: hwc_to_chw(&output, height, width)?,
                            height,
                            width,
                        });
                    }
                    return Ok(PipelineOutput::F32Hwc {
                        data: output,
                        height,
                        width,
                    });
                }
                (TransformPlan::ToTorch, SampledTransform::ToTorch) => {
                    return Ok(PipelineOutput::U8Chw {
                        data: hwc_u8_to_chw(&image.data, image.height, image.width)?,
                        height: image.height,
                        width: image.width,
                    });
                }
                _ => {
                    return Err(CoreError::Runtime(
                        "sampled plan does not match pipeline".into(),
                    ))
                }
            }
        }

        Ok(PipelineOutput::U8Hwc {
            data: image.data,
            height: image.height,
            width: image.width,
        })
    }

    fn compiled_terminal_entry(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        sampled: &[SampledTransform],
    ) -> CoreResult<Option<PipelineOutput>> {
        if let Some(fusion) = self
            .plan
            .lowering
            .fusion()
            .filter(|selection| selection.first == 0)
        {
            match (fusion.rule, self.plan.transforms.as_slice(), sampled) {
                (
                    FusionRule::NormalizeToTorch,
                    [TransformPlan::Normalize {
                        mean,
                        std,
                        max_pixel_value,
                        ..
                    }, TransformPlan::ToTorch],
                    [SampledTransform::Normalize, SampledTransform::ToTorch],
                ) => {
                    return Ok(Some(PipelineOutput::F32Chw {
                        data: normalize_hwc_to_chw(
                            data,
                            height,
                            width,
                            *mean,
                            *std,
                            *max_pixel_value,
                        )?,
                        height,
                        width,
                    }));
                }
                (
                    FusionRule::NormalizeToTorch,
                    [TransformPlan::Normalize { .. }, TransformPlan::ToTorch],
                    [SampledTransform::Skip, SampledTransform::ToTorch],
                ) => {
                    return Ok(Some(PipelineOutput::U8Chw {
                        data: hwc_u8_to_chw(data, height, width)?,
                        height,
                        width,
                    }));
                }
                _ => {
                    return Err(CoreError::Runtime(
                        "selected fusion does not match sampled plan".into(),
                    ));
                }
            }
        }
        if !matches!(
            self.plan.lowering.nodes(),
            [node] if node.kernel
                == KernelSelection::Form(ExecutionForm::BorrowedToOwned)
        ) {
            return Ok(None);
        }
        match (self.plan.transforms.as_slice(), sampled) {
            ([TransformPlan::ToTorch], [SampledTransform::ToTorch]) => {
                Ok(Some(PipelineOutput::U8Chw {
                    data: hwc_u8_to_chw(data, height, width)?,
                    height,
                    width,
                }))
            }
            (
                [TransformPlan::Normalize {
                    mean,
                    std,
                    max_pixel_value,
                    ..
                }],
                [SampledTransform::Normalize],
            ) => Ok(Some(PipelineOutput::F32Hwc {
                data: normalize_hwc(data, *mean, *std, *max_pixel_value)?,
                height,
                width,
            })),
            _ => Ok(None),
        }
    }

    fn compiled_entry(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        sampled: &[SampledTransform],
        workspace: &mut Workspace,
    ) -> CoreResult<(ImageU8, usize)> {
        let owned_copy = || {
            Ok((
                ImageU8 {
                    data: copy_u8(data)?,
                    height,
                    width,
                },
                0,
            ))
        };
        if !self.plan.lowering.entry_ready(sampled) {
            return owned_copy();
        }
        if !matches!(
            self.plan.lowering.nodes().first().map(|node| node.kernel),
            Some(KernelSelection::Form(ExecutionForm::BorrowedToOwned))
        ) {
            return owned_copy();
        }
        match (self.plan.transforms.first(), sampled.first()) {
            (
                Some(TransformPlan::RandomResizedCrop { .. }),
                Some(SampledTransform::RandomResizedCrop {
                    crop,
                    height: out_h,
                    width: out_w,
                    interpolation,
                    antialias,
                }),
            ) => {
                let crop_destination =
                    workspace.take_u8(rgb_len(crop.height, crop.width)?, false)?;
                let cropped = random_crop_raw_into(data, height, width, *crop, crop_destination)?;
                let resize_destination = workspace.take_u8(rgb_len(*out_h, *out_w)?, false)?;
                let resized = resize_raw(
                    &cropped.data,
                    cropped.height,
                    cropped.width,
                    *out_h,
                    *out_w,
                    *interpolation,
                    *antialias,
                    workspace.resizer(),
                    resize_destination,
                )?;
                workspace.recycle_u8(cropped.data);
                Ok((resized, 1))
            }
            (
                Some(TransformPlan::Resize { .. }),
                Some(SampledTransform::Resize {
                    height: out_h,
                    width: out_w,
                    interpolation,
                    antialias,
                }),
            ) => {
                let destination = workspace.take_u8(rgb_len(*out_h, *out_w)?, false)?;
                Ok((
                    resize_raw(
                        data,
                        height,
                        width,
                        *out_h,
                        *out_w,
                        *interpolation,
                        *antialias,
                        workspace.resizer(),
                        destination,
                    )?,
                    1,
                ))
            }
            (
                Some(TransformPlan::RandomCrop { .. } | TransformPlan::CenterCrop { .. }),
                Some(SampledTransform::RandomCrop(crop) | SampledTransform::CenterCrop(crop)),
            ) => {
                let destination = workspace.take_u8(rgb_len(crop.height, crop.width)?, false)?;
                Ok((
                    random_crop_raw_into(data, height, width, *crop, destination)?,
                    1,
                ))
            }
            (
                Some(TransformPlan::PadIfNeeded { .. }),
                Some(SampledTransform::PadIfNeeded(sample)),
            ) => {
                let destination =
                    workspace.take_u8(rgb_len(sample.height, sample.width)?, false)?;
                Ok((pad_raw(data, height, width, *sample, destination)?, 1))
            }
            (Some(TransformPlan::VerticalFlip { .. }), Some(SampledTransform::VerticalFlip)) => {
                let mut destination = workspace.take_u8(data.len(), false)?;
                point::vertical_flip_into(data, &mut destination, height, width);
                Ok((
                    ImageU8 {
                        data: destination,
                        height,
                        width,
                    },
                    1,
                ))
            }
            (Some(TransformPlan::Invert { .. }), Some(SampledTransform::Invert)) => {
                let mut destination = workspace.take_u8(data.len(), false)?;
                point::invert_into(data, &mut destination);
                Ok((
                    ImageU8 {
                        data: destination,
                        height,
                        width,
                    },
                    1,
                ))
            }
            (
                Some(TransformPlan::Solarize { .. }),
                Some(SampledTransform::Solarize { threshold }),
            ) => {
                let mut destination = workspace.take_u8(data.len(), false)?;
                point::solarize_into(data, &mut destination, *threshold);
                Ok((
                    ImageU8 {
                        data: destination,
                        height,
                        width,
                    },
                    1,
                ))
            }
            (Some(TransformPlan::Posterize { .. }), Some(SampledTransform::Posterize { bits })) => {
                let mut destination = workspace.take_u8(data.len(), false)?;
                point::posterize_into(data, &mut destination, *bits);
                Ok((
                    ImageU8 {
                        data: destination,
                        height,
                        width,
                    },
                    1,
                ))
            }
            (Some(TransformPlan::Sharpen { .. }), Some(SampledTransform::Sharpen(sample))) => {
                let destination = workspace.take_u8(data.len(), false)?;
                Ok((sharpen_raw(data, height, width, *sample, destination)?, 1))
            }
            (
                Some(TransformPlan::Perspective { .. }),
                Some(SampledTransform::Perspective(sample)),
            ) => {
                let destination = workspace.take_u8(data.len(), false)?;
                Ok((
                    perspective_raw(data, height, width, *sample, destination)?,
                    1,
                ))
            }
            (
                Some(TransformPlan::GridDistortion { .. }),
                Some(SampledTransform::GridDistortion(sample)),
            ) => {
                let destination = workspace.take_u8(data.len(), false)?;
                Ok((
                    grid_distortion_raw(
                        data,
                        height,
                        width,
                        sample,
                        destination,
                        workspace.axis_remap(),
                    )?,
                    1,
                ))
            }
            _ => Err(CoreError::Runtime(
                "selected borrowed-to-owned lowering does not match sampled plan".into(),
            )),
        }
    }
}
