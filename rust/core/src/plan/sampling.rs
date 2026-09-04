use super::*;

impl TransformPlan {
    pub(crate) fn sample(
        transforms: &[Self],
        height: usize,
        width: usize,
        seed: u64,
    ) -> CoreResult<Vec<SampledTransform>> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut current_height = height;
        let mut current_width = width;
        let mut sampled = Vec::new();
        sampled
            .try_reserve_exact(transforms.len())
            .map_err(|_| CoreError::Runtime("sampled plan allocation failed".into()))?;
        for transform in transforms {
            let next = transform.sample_one(current_height, current_width, &mut rng)?;
            match next {
                SampledTransform::Resize { height, width, .. }
                | SampledTransform::RandomCrop(CropSample { height, width, .. })
                | SampledTransform::RandomResizedCrop { height, width, .. }
                | SampledTransform::CenterCrop(CropSample { height, width, .. })
                | SampledTransform::PadIfNeeded(PadSample { height, width, .. }) => {
                    current_height = height;
                    current_width = width;
                }
                _ => {}
            }
            sampled.push(next);
        }
        Ok(sampled)
    }

    fn sample_one(
        &self,
        height: usize,
        width: usize,
        rng: &mut SmallRng,
    ) -> CoreResult<SampledTransform> {
        let probability = match self {
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
        };
        if !should_apply(probability, rng) {
            return Ok(SampledTransform::Skip);
        }
        Ok(match self {
            Self::Resize { height, width, .. } => SampledTransform::Resize {
                height: *height,
                width: *width,
            },
            Self::RandomCrop {
                height: crop_height,
                width: crop_width,
                ..
            } => {
                if *crop_height > height || *crop_width > width {
                    return Err(CoreError::Invalid("crop larger than input".into()));
                }
                SampledTransform::RandomCrop(CropSample {
                    top: rng.random_range(0..=height - crop_height),
                    left: rng.random_range(0..=width - crop_width),
                    height: *crop_height,
                    width: *crop_width,
                })
            }
            Self::RandomResizedCrop {
                height: output_height,
                width: output_width,
                scale,
                ratio,
                ..
            } => SampledTransform::RandomResizedCrop {
                crop: sample_resized_crop(height, width, *scale, *ratio, rng),
                height: *output_height,
                width: *output_width,
            },
            Self::HorizontalFlip { .. } => SampledTransform::HorizontalFlip,
            Self::VerticalFlip { .. } => SampledTransform::VerticalFlip,
            Self::CenterCrop {
                height: crop_height,
                width: crop_width,
                ..
            } => {
                if *crop_height > height || *crop_width > width {
                    return Err(CoreError::Invalid("crop larger than input".into()));
                }
                SampledTransform::CenterCrop(CropSample {
                    top: (height - crop_height) / 2,
                    left: (width - crop_width) / 2,
                    height: *crop_height,
                    width: *crop_width,
                })
            }
            Self::PadIfNeeded {
                min_height,
                min_width,
                pad_height_divisor,
                pad_width_divisor,
                position,
                ..
            } => {
                let output_height = padded_dimension(height, *min_height, *pad_height_divisor)?;
                let output_width = padded_dimension(width, *min_width, *pad_width_divisor)?;
                validate_dimensions(output_height, output_width)?;
                let extra_height = output_height - height;
                let extra_width = output_width - width;
                let (top, left) = sample_pad_origin(*position, extra_height, extra_width, rng);
                SampledTransform::PadIfNeeded(PadSample {
                    top,
                    left,
                    height: output_height,
                    width: output_width,
                })
            }
            Self::CoarseDropout {
                num_holes_range,
                hole_height_range,
                hole_width_range,
                fill,
                ..
            } => {
                let count = rng.random_range(num_holes_range[0]..=num_holes_range[1]);
                let mut holes = Vec::new();
                holes
                    .try_reserve_exact(count)
                    .map_err(|_| CoreError::Runtime("dropout sample allocation failed".into()))?;
                for _ in 0..count {
                    let hole_height = sample_dropout_dimension(*hole_height_range, height, rng);
                    let hole_width = sample_dropout_dimension(*hole_width_range, width, rng);
                    holes.push(DropoutHole {
                        top: rng.random_range(0..=height - hole_height),
                        left: rng.random_range(0..=width - hole_width),
                        height: hole_height,
                        width: hole_width,
                    });
                }
                SampledTransform::CoarseDropout { holes, fill: *fill }
            }
            Self::ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
                ..
            } => {
                let brightness = sample_uniform(*brightness, rng);
                let contrast = sample_uniform(*contrast, rng);
                let saturation = sample_uniform(*saturation, rng);
                let hue_enabled = *hue != [0.0, 0.0];
                let hue = sample_uniform(*hue, rng);
                let mut order = [0, 1, 2, 3];
                if hue_enabled {
                    order.shuffle(rng);
                } else {
                    order[..3].shuffle(rng);
                }
                SampledTransform::ColorJitter(ColorJitterSample {
                    brightness,
                    contrast,
                    saturation,
                    hue,
                    hue_enabled,
                    order,
                })
            }
            Self::Affine {
                degrees,
                translate,
                scale,
                shear,
                ..
            } => SampledTransform::Affine(AffineSample {
                degrees: sample_uniform(*degrees, rng),
                translate: [
                    sample_symmetric(translate[0] * width as f32, rng),
                    sample_symmetric(translate[1] * height as f32, rng),
                ],
                scale: sample_uniform(*scale, rng),
                shear: [
                    sample_uniform([shear[0], shear[1]], rng),
                    sample_uniform([shear[2], shear[3]], rng),
                ],
            }),
            Self::RandomRotation { degrees, .. } => {
                SampledTransform::RandomRotation(RotationSample {
                    degrees: sample_uniform(*degrees, rng),
                })
            }
            Self::GaussianNoise {
                mean,
                std,
                per_channel,
                ..
            } => SampledTransform::GaussianNoise(GaussianNoiseSample {
                mean: sample_uniform(*mean, rng),
                std: sample_uniform(*std, rng),
                seed: rng.random(),
                per_channel: *per_channel,
            }),
            Self::Sharpen {
                alpha, lightness, ..
            } => SampledTransform::Sharpen(SharpenSample {
                alpha: sample_uniform(*alpha, rng),
                lightness: sample_uniform(*lightness, rng),
            }),
            Self::Perspective { scale, .. } => SampledTransform::Perspective(PerspectiveSample {
                inverse: sample_perspective(height, width, *scale, rng),
            }),
            Self::GridDistortion {
                num_steps,
                distort_limit,
                ..
            } => SampledTransform::GridDistortion(GridDistortionSample {
                x_map: sample_grid_map(width, *num_steps, *distort_limit, rng)?,
                y_map: sample_grid_map(height, *num_steps, *distort_limit, rng)?,
            }),
            Self::GaussianBlur { sigma, .. } => SampledTransform::GaussianBlur {
                sigma: sample_uniform(*sigma, rng),
            },
            Self::Grayscale { .. } => SampledTransform::Grayscale,
            Self::Invert { .. } => SampledTransform::Invert,
            Self::Solarize { threshold, .. } => SampledTransform::Solarize {
                threshold: *threshold,
            },
            Self::Posterize { bits, .. } => SampledTransform::Posterize { bits: *bits },
            Self::Normalize { .. } => SampledTransform::Normalize,
        })
    }
}
