use crate::capability::ExecutionForm;
use crate::kernels::layout::{hwc_to_chw, hwc_u8_to_chw, normalize_hwc, normalize_hwc_to_chw};
use crate::kernels::point;
use crate::mask::{mask_len, MaskPlan};
use crate::operations::*;
use crate::optimization::{KernelSelection, LoweringPlan};
use crate::plan::{
    derive_run_seed, make_gaussian_kernel, AffineSample, SampledTransform, TransformPlan,
};
use crate::{
    CoreError, CoreResult, ExecutionMode, ImageOutput, PipelineExplanation, PipelineOutput,
    PipelineSpec, TargetBuffer, TargetInput, TargetOutput, TargetRequirements, TargetSpec,
    Workspace,
};

pub struct CompiledPipeline {
    plan: ExecutionPlan,
}

struct ExecutionPlan {
    transforms: Vec<TransformPlan>,
    mode: ExecutionMode,
    lowering: LoweringPlan,
    mask: MaskPlan,
    targets: Vec<TargetSpec>,
    requirements: Vec<TargetRequirements>,
    explanation: PipelineExplanation,
}

impl ExecutionPlan {
    fn compile(spec: PipelineSpec, mode: ExecutionMode) -> CoreResult<Self> {
        let (transform_specs, targets, requirements) = spec.into_parts();
        if targets.is_empty() {
            return Err(CoreError::Invalid(
                "a pipeline requires at least one target".into(),
            ));
        }
        let transforms = TransformPlan::compile(transform_specs)?;
        let mask = MaskPlan::compile(&transforms)?;
        let lowering = LoweringPlan::compile(&transforms, mode)?;
        let explanation = crate::explanation::build(&transforms, mode, &lowering, &mask);
        Ok(Self {
            lowering,
            mask,
            targets,
            requirements,
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
        let mut output = self.sample_and_run_image(
            data,
            height,
            width,
            derive_run_seed(pipeline_seed, run_key),
            workspace,
            TargetRequirements::HWC,
        )?;
        output
            .hwc
            .take()
            .ok_or_else(|| CoreError::Runtime("missing HWC image output".into()))
    }

    pub fn apply_targets<'a>(
        &self,
        inputs: Vec<TargetInput<'a>>,
        pipeline_seed: u64,
        run_key: u64,
        workspace: &mut Workspace,
    ) -> CoreResult<Vec<TargetOutput>> {
        self.validate_targets(&inputs)?;
        let height = inputs[0].height;
        let width = inputs[0].width;
        let sampled = TransformPlan::sample(
            &self.plan.transforms,
            height,
            width,
            derive_run_seed(pipeline_seed, run_key),
        )?;
        let reuse = self.plan.mode != ExecutionMode::StagedFresh;
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(inputs.len())
            .map_err(|_| CoreError::Runtime("target output allocation failed".into()))?;
        for (input, requirements) in inputs.into_iter().zip(&self.plan.requirements) {
            let output = match (input.role, input.data) {
                (TargetSpec::Image, TargetBuffer::Borrowed(data)) => {
                    TargetOutput::Image(self.run_image_outputs(
                        data,
                        height,
                        width,
                        &sampled,
                        workspace,
                        *requirements,
                    )?)
                }
                (TargetSpec::Image, TargetBuffer::Owned(data)) => {
                    TargetOutput::Image(self.run_image_outputs(
                        &data,
                        height,
                        width,
                        &sampled,
                        workspace,
                        *requirements,
                    )?)
                }
                (TargetSpec::Mask { fill }, TargetBuffer::Borrowed(data)) => {
                    TargetOutput::Mask(self.plan.mask.apply(
                        data,
                        (height, width),
                        &sampled,
                        fill,
                        workspace,
                        reuse,
                    )?)
                }
                (TargetSpec::Mask { fill }, TargetBuffer::Owned(data)) => {
                    TargetOutput::Mask(self.plan.mask.apply_owned(
                        data,
                        (height, width),
                        &sampled,
                        fill,
                        workspace,
                        reuse,
                    )?)
                }
            };
            outputs.push(output);
        }
        Ok(outputs)
    }

    pub fn explain(&self) -> PipelineExplanation {
        self.plan.explanation.clone()
    }
}

fn materialize_image_outputs(
    output: PipelineOutput,
    requirements: TargetRequirements,
) -> CoreResult<ImageOutput> {
    match output {
        PipelineOutput::U8Hwc {
            data,
            height,
            width,
        } => {
            let chw = requirements
                .chw
                .then(|| hwc_u8_to_chw(&data, height, width))
                .transpose()?
                .map(|data| PipelineOutput::U8Chw {
                    data,
                    height,
                    width,
                });
            Ok(ImageOutput {
                hwc: requirements.hwc.then_some(PipelineOutput::U8Hwc {
                    data,
                    height,
                    width,
                }),
                chw,
            })
        }
        PipelineOutput::F32Hwc {
            data,
            height,
            width,
        } => {
            let chw = requirements
                .chw
                .then(|| hwc_to_chw(&data, height, width))
                .transpose()?
                .map(|data| PipelineOutput::F32Chw {
                    data,
                    height,
                    width,
                });
            Ok(ImageOutput {
                hwc: requirements.hwc.then_some(PipelineOutput::F32Hwc {
                    data,
                    height,
                    width,
                }),
                chw,
            })
        }
        output @ (PipelineOutput::U8Chw { .. } | PipelineOutput::F32Chw { .. }) => {
            if requirements.hwc {
                return Err(CoreError::Runtime(
                    "direct CHW output cannot satisfy an HWC requirement".into(),
                ));
            }
            Ok(ImageOutput {
                hwc: None,
                chw: requirements.chw.then_some(output),
            })
        }
    }
}

impl CompiledPipeline {
    fn validate_targets(&self, inputs: &[TargetInput<'_>]) -> CoreResult<()> {
        if inputs.len() != self.plan.targets.len() {
            return Err(CoreError::Invalid(
                "target input count does not match the pipeline signature".into(),
            ));
        }
        if self.plan.requirements.len() != self.plan.targets.len() {
            return Err(CoreError::Runtime(
                "target requirements do not match the pipeline signature".into(),
            ));
        }
        let Some(first) = inputs.first() else {
            return Err(CoreError::Invalid(
                "a pipeline call requires at least one target".into(),
            ));
        };
        if first.height == 0 || first.width == 0 {
            return Err(CoreError::Invalid(
                "target dimensions must be positive".into(),
            ));
        }
        for (index, (expected, input)) in self.plan.targets.iter().zip(inputs).enumerate() {
            if expected != &input.role {
                return Err(CoreError::Invalid(format!(
                    "target {index} role does not match the pipeline signature"
                )));
            }
            if (input.height, input.width) != (first.height, first.width) {
                return Err(CoreError::Invalid(format!(
                    "target {index} dimensions do not match the initial coordinate frame"
                )));
            }
            let expected_len = match expected {
                TargetSpec::Image => rgb_len(input.height, input.width)?,
                TargetSpec::Mask { .. } => mask_len(input.height, input.width)?,
            };
            let actual_len = match &input.data {
                TargetBuffer::Borrowed(data) => data.len(),
                TargetBuffer::Owned(data) => data.len(),
            };
            if actual_len != expected_len {
                return Err(CoreError::Invalid(format!(
                    "target {index} buffer does not match its role and shape"
                )));
            }
            let requirements = self.plan.requirements[index];
            if matches!(expected, TargetSpec::Image) && !requirements.hwc && !requirements.chw {
                return Err(CoreError::Invalid(format!(
                    "target {index} requires at least one image layout"
                )));
            }
        }
        Ok(())
    }

    fn sample_and_run_image(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        run_seed: u64,
        workspace: &mut Workspace,
        requirements: TargetRequirements,
    ) -> CoreResult<ImageOutput> {
        let sampled = TransformPlan::sample(&self.plan.transforms, height, width, run_seed)?;
        self.run_image_outputs(data, height, width, &sampled, workspace, requirements)
    }

    fn run_image_outputs(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        sampled: &[SampledTransform],
        workspace: &mut Workspace,
        requirements: TargetRequirements,
    ) -> CoreResult<ImageOutput> {
        let output = self.run_image(data, height, width, sampled, workspace, requirements)?;
        materialize_image_outputs(output, requirements)
    }

    fn run_image(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        sampled: &[SampledTransform],
        workspace: &mut Workspace,
        requirements: TargetRequirements,
    ) -> CoreResult<PipelineOutput> {
        let reuse = self.plan.mode != ExecutionMode::StagedFresh;
        if self.plan.mode == ExecutionMode::Compiled {
            if let Some(output) =
                self.compiled_terminal_entry(data, height, width, sampled, requirements)?
            {
                return Ok(output);
            }
        }
        let (mut image, start) = if self.plan.mode == ExecutionMode::Compiled {
            self.compiled_entry(data, height, width, sampled, workspace)?
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
                    TransformPlan::Resize {
                        interpolation,
                        antialias,
                        ..
                    },
                    SampledTransform::Resize { height, width },
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
                    TransformPlan::RandomResizedCrop {
                        interpolation,
                        antialias,
                        ..
                    },
                    SampledTransform::RandomResizedCrop {
                        crop,
                        height,
                        width,
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
                (
                    TransformPlan::PadIfNeeded {
                        border_mode, fill, ..
                    },
                    SampledTransform::PadIfNeeded(sample),
                ) => {
                    if sample.height == image.height && sample.width == image.width {
                        continue;
                    }
                    let destination = workspace.take_staged_u8(
                        rgb_len(sample.height, sample.width)?,
                        false,
                        reuse,
                    )?;
                    let next = pad_raw(
                        &image.data,
                        image.height,
                        image.width,
                        *sample,
                        *border_mode,
                        *fill,
                        destination,
                    )?;
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
                (
                    TransformPlan::Affine {
                        interpolation,
                        border_mode,
                        fill,
                        ..
                    },
                    SampledTransform::Affine(sample),
                ) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let next = rotate_raw(
                        &image.data,
                        image.height,
                        image.width,
                        *sample,
                        RgbRasterPolicy {
                            interpolation: *interpolation,
                            border_mode: *border_mode,
                            fill: *fill,
                        },
                        destination,
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (
                    TransformPlan::RandomRotation {
                        interpolation,
                        border_mode,
                        fill,
                        ..
                    },
                    SampledTransform::RandomRotation(sample),
                ) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let affine = AffineSample {
                        degrees: sample.degrees,
                        translate: [0.0, 0.0],
                        scale: 1.0,
                        shear: [0.0, 0.0],
                    };
                    let next = rotate_raw(
                        &image.data,
                        image.height,
                        image.width,
                        affine,
                        RgbRasterPolicy {
                            interpolation: *interpolation,
                            border_mode: *border_mode,
                            fill: *fill,
                        },
                        destination,
                    )?;
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
                (
                    TransformPlan::Perspective {
                        interpolation,
                        border_mode,
                        fill,
                        ..
                    },
                    SampledTransform::Perspective(sample),
                ) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let next = perspective_raw(
                        &image.data,
                        image.height,
                        image.width,
                        *sample,
                        RgbRasterPolicy {
                            interpolation: *interpolation,
                            border_mode: *border_mode,
                            fill: *fill,
                        },
                        destination,
                    )?;
                    workspace.recycle_staged_u8(image.data, reuse);
                    image = next;
                }
                (
                    TransformPlan::GridDistortion {
                        interpolation,
                        border_mode,
                        fill,
                        ..
                    },
                    SampledTransform::GridDistortion(sample),
                ) => {
                    let destination = workspace.take_staged_u8(image.data.len(), false, reuse)?;
                    let next = grid_distortion_raw(
                        &image.data,
                        image.height,
                        image.width,
                        sample,
                        RgbRasterPolicy {
                            interpolation: *interpolation,
                            border_mode: *border_mode,
                            fill: *fill,
                        },
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
                    if requirements.chw && !requirements.hwc {
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
                    return Ok(PipelineOutput::F32Hwc {
                        data: output,
                        height,
                        width,
                    });
                }
                _ => {
                    return Err(CoreError::Runtime(
                        "sampled plan does not match pipeline".into(),
                    ))
                }
            }
        }

        if requirements.chw && !requirements.hwc {
            Ok(PipelineOutput::U8Chw {
                data: hwc_u8_to_chw(&image.data, image.height, image.width)?,
                height: image.height,
                width: image.width,
            })
        } else {
            Ok(PipelineOutput::U8Hwc {
                data: image.data,
                height: image.height,
                width: image.width,
            })
        }
    }

    fn compiled_terminal_entry(
        &self,
        data: &[u8],
        height: usize,
        width: usize,
        sampled: &[SampledTransform],
        requirements: TargetRequirements,
    ) -> CoreResult<Option<PipelineOutput>> {
        if requirements.chw
            && !requirements.hwc
            && sampled
                .iter()
                .all(|sampled| matches!(sampled, SampledTransform::Skip))
        {
            return Ok(Some(PipelineOutput::U8Chw {
                data: hwc_u8_to_chw(data, height, width)?,
                height,
                width,
            }));
        }
        if !matches!(
            self.plan.lowering.nodes(),
            [node] if node.kernel
                == KernelSelection::Form(ExecutionForm::BorrowedToOwned)
        ) {
            return Ok(None);
        }
        match (self.plan.transforms.as_slice(), sampled) {
            (
                [TransformPlan::Normalize {
                    mean,
                    std,
                    max_pixel_value,
                    ..
                }],
                [SampledTransform::Normalize],
            ) => Ok(Some(if requirements.chw && !requirements.hwc {
                PipelineOutput::F32Chw {
                    data: normalize_hwc_to_chw(data, height, width, *mean, *std, *max_pixel_value)?,
                    height,
                    width,
                }
            } else {
                PipelineOutput::F32Hwc {
                    data: normalize_hwc(data, *mean, *std, *max_pixel_value)?,
                    height,
                    width,
                }
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
                Some(TransformPlan::RandomResizedCrop {
                    interpolation,
                    antialias,
                    ..
                }),
                Some(SampledTransform::RandomResizedCrop {
                    crop,
                    height: out_h,
                    width: out_w,
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
                Some(TransformPlan::Resize {
                    interpolation,
                    antialias,
                    ..
                }),
                Some(SampledTransform::Resize {
                    height: out_h,
                    width: out_w,
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
                Some(TransformPlan::PadIfNeeded {
                    border_mode, fill, ..
                }),
                Some(SampledTransform::PadIfNeeded(sample)),
            ) => {
                let destination =
                    workspace.take_u8(rgb_len(sample.height, sample.width)?, false)?;
                Ok((
                    pad_raw(
                        data,
                        height,
                        width,
                        *sample,
                        *border_mode,
                        *fill,
                        destination,
                    )?,
                    1,
                ))
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
                Some(TransformPlan::Perspective {
                    interpolation,
                    border_mode,
                    fill,
                    ..
                }),
                Some(SampledTransform::Perspective(sample)),
            ) => {
                let destination = workspace.take_u8(data.len(), false)?;
                Ok((
                    perspective_raw(
                        data,
                        height,
                        width,
                        *sample,
                        RgbRasterPolicy {
                            interpolation: *interpolation,
                            border_mode: *border_mode,
                            fill: *fill,
                        },
                        destination,
                    )?,
                    1,
                ))
            }
            (
                Some(TransformPlan::GridDistortion {
                    interpolation,
                    border_mode,
                    fill,
                    ..
                }),
                Some(SampledTransform::GridDistortion(sample)),
            ) => {
                let destination = workspace.take_u8(data.len(), false)?;
                Ok((
                    grid_distortion_raw(
                        data,
                        height,
                        width,
                        sample,
                        RgbRasterPolicy {
                            interpolation: *interpolation,
                            border_mode: *border_mode,
                            fill: *fill,
                        },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Compiler, PipelineSpec, TargetBuffer, TargetInput, TargetOutput, TargetSpec};

    #[test]
    fn target_entry_validates_all_buffers_before_allocating_output() {
        let pipeline = Compiler::new(ExecutionMode::Compiled)
            .compile(PipelineSpec::with_targets(
                Vec::new(),
                vec![TargetSpec::Image, TargetSpec::Mask { fill: 0 }],
            ))
            .unwrap();
        let image = vec![7; 2 * 3 * 3];
        let mut workspace = Workspace::default();
        assert!(matches!(
            pipeline.apply_targets(
                vec![
                    TargetInput {
                        role: TargetSpec::Image,
                        data: TargetBuffer::Borrowed(&image),
                        height: 2,
                        width: 3,
                    },
                    TargetInput {
                        role: TargetSpec::Mask { fill: 0 },
                        data: TargetBuffer::Borrowed(&[1; 5]),
                        height: 2,
                        width: 3,
                    },
                ],
                137,
                3,
                &mut workspace,
            ),
            Err(CoreError::Invalid(message)) if message.contains("target 1 buffer")
        ));
        assert_eq!(workspace.retained_bytes(), 0);

        let mut outputs = pipeline
            .apply_targets(
                vec![
                    TargetInput {
                        role: TargetSpec::Image,
                        data: TargetBuffer::Borrowed(&image),
                        height: 2,
                        width: 3,
                    },
                    TargetInput {
                        role: TargetSpec::Mask { fill: 0 },
                        data: TargetBuffer::Borrowed(&[0, 1, 2, 3, 254, 255]),
                        height: 2,
                        width: 3,
                    },
                ],
                137,
                3,
                &mut workspace,
            )
            .unwrap();
        assert!(matches!(
            outputs.remove(0),
            TargetOutput::Image(ImageOutput {
                hwc: Some(PipelineOutput::U8Hwc {
                    data,
                    height: 2,
                    width: 3,
                }),
                chw: None,
            }) if data == image
        ));
        assert!(matches!(
            outputs.remove(0),
            TargetOutput::Mask(output)
                if output.data == [0, 1, 2, 3, 254, 255]
                    && (output.height, output.width) == (2, 3)
        ));
    }

    #[test]
    fn owned_mask_target_adopts_the_input_buffer() {
        let pipeline = Compiler::new(ExecutionMode::Compiled)
            .compile(PipelineSpec::with_targets(
                Vec::new(),
                vec![TargetSpec::Image, TargetSpec::Mask { fill: 0 }],
            ))
            .unwrap();
        let image = vec![7; 2 * 3 * 3];
        let mask = vec![0, 1, 2, 3, 254, 255];
        let input_pointer = mask.as_ptr();
        let mut workspace = Workspace::default();
        let mut outputs = pipeline
            .apply_targets(
                vec![
                    TargetInput {
                        role: TargetSpec::Image,
                        data: TargetBuffer::Borrowed(&image),
                        height: 2,
                        width: 3,
                    },
                    TargetInput {
                        role: TargetSpec::Mask { fill: 0 },
                        data: TargetBuffer::Owned(mask),
                        height: 2,
                        width: 3,
                    },
                ],
                137,
                3,
                &mut workspace,
            )
            .unwrap();
        let TargetOutput::Mask(output) = outputs.remove(1) else {
            panic!("expected a mask output")
        };
        assert_eq!(output.data.as_ptr(), input_pointer);
        assert_eq!(output.data, [0, 1, 2, 3, 254, 255]);
    }
}
