use super::affine;
use crate::{BorderMode, CoreError, CoreResult, Interpolation};

const BLOCK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Descriptor {
    Invalid,
    Nearest {
        x: isize,
        y: isize,
    },
    Bilinear {
        x0: isize,
        y0: isize,
        wx: u32,
        wy: u32,
    },
}

#[derive(Clone, Copy)]
struct BilinearAxis {
    low: isize,
    weight: u32,
}

#[derive(Default)]
pub(crate) struct AxisRemapScratch {
    nearest_x: Vec<isize>,
    nearest_y: Vec<isize>,
    bilinear_x: Vec<BilinearAxis>,
    bilinear_y: Vec<BilinearAxis>,
}

impl AxisRemapScratch {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.nearest_x
            .capacity()
            .saturating_add(self.nearest_y.capacity())
            .saturating_mul(std::mem::size_of::<isize>())
            .saturating_add(
                self.bilinear_x
                    .capacity()
                    .saturating_add(self.bilinear_y.capacity())
                    .saturating_mul(std::mem::size_of::<BilinearAxis>()),
            )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn perspective(
    data: &[u8],
    height: usize,
    width: usize,
    inverse: [f32; 9],
    interpolation: Interpolation,
    border_mode: BorderMode,
    fill: [u8; 3],
    output: &mut [u8],
) {
    let mut descriptors = [Descriptor::Invalid; BLOCK];
    for y in 0..height {
        for block_start in (0..width).step_by(BLOCK) {
            let count = (width - block_start).min(BLOCK);
            perspective_descriptors(
                inverse,
                y,
                block_start,
                interpolation,
                &mut descriptors[..count],
            );
            sample_descriptors(
                data,
                height,
                width,
                &descriptors[..count],
                y * width + block_start,
                border_mode,
                fill,
                output,
            );
        }
    }
}

fn perspective_descriptors(
    inverse: [f32; 9],
    y: usize,
    x_start: usize,
    interpolation: Interpolation,
    descriptors: &mut [Descriptor],
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection guards AVX2 and the output slice bounds every lane.
        unsafe { perspective_descriptors_avx2(inverse, y, x_start, interpolation, descriptors) };
        return;
    }
    perspective_descriptors_scalar(inverse, y, x_start, interpolation, descriptors, 0);
}

fn perspective_descriptors_scalar(
    inverse: [f32; 9],
    y: usize,
    x_start: usize,
    interpolation: Interpolation,
    descriptors: &mut [Descriptor],
    start: usize,
) {
    for (offset, descriptor) in descriptors.iter_mut().enumerate().skip(start) {
        let x = (x_start + offset) as f32;
        let y = y as f32;
        let denominator = inverse[6] * x + inverse[7] * y + inverse[8];
        *descriptor = if !denominator.is_finite() || denominator.abs() < 1e-8 {
            Descriptor::Invalid
        } else {
            let source_x = (inverse[0] * x + inverse[1] * y + inverse[2]) / denominator;
            let source_y = (inverse[3] * x + inverse[4] * y + inverse[5]) / denominator;
            if source_x.is_finite() && source_y.is_finite() {
                make_descriptor(source_x, source_y, interpolation)
            } else {
                Descriptor::Invalid
            }
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn perspective_descriptors_avx2(
    inverse: [f32; 9],
    y_value: usize,
    x_start: usize,
    interpolation: Interpolation,
    descriptors: &mut [Descriptor],
) {
    // SAFETY: the caller guarantees AVX2 and the loop stores only complete eight-lane blocks.
    unsafe {
        use std::arch::x86_64::*;

        let y = _mm256_set1_ps(y_value as f32);
        let mut offset = 0usize;
        while offset + 8 <= descriptors.len() {
            let x = _mm256_setr_ps(
                (x_start + offset) as f32,
                (x_start + offset + 1) as f32,
                (x_start + offset + 2) as f32,
                (x_start + offset + 3) as f32,
                (x_start + offset + 4) as f32,
                (x_start + offset + 5) as f32,
                (x_start + offset + 6) as f32,
                (x_start + offset + 7) as f32,
            );
            let evaluate = |a: f32, b: f32, c: f32| {
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_mul_ps(_mm256_set1_ps(a), x),
                        _mm256_mul_ps(_mm256_set1_ps(b), y),
                    ),
                    _mm256_set1_ps(c),
                )
            };
            let denominator = evaluate(inverse[6], inverse[7], inverse[8]);
            let source_x = _mm256_div_ps(evaluate(inverse[0], inverse[1], inverse[2]), denominator);
            let source_y = _mm256_div_ps(evaluate(inverse[3], inverse[4], inverse[5]), denominator);
            let mut denominators = [0.0_f32; 8];
            let mut source_xs = [0.0_f32; 8];
            let mut source_ys = [0.0_f32; 8];
            _mm256_storeu_ps(denominators.as_mut_ptr(), denominator);
            _mm256_storeu_ps(source_xs.as_mut_ptr(), source_x);
            _mm256_storeu_ps(source_ys.as_mut_ptr(), source_y);
            for lane in 0..8 {
                descriptors[offset + lane] = if !denominators[lane].is_finite()
                    || denominators[lane].abs() < 1e-8
                    || !source_xs[lane].is_finite()
                    || !source_ys[lane].is_finite()
                {
                    Descriptor::Invalid
                } else {
                    make_descriptor(source_xs[lane], source_ys[lane], interpolation)
                };
            }
            offset += 8;
        }
        perspective_descriptors_scalar(
            inverse,
            y_value,
            x_start,
            interpolation,
            descriptors,
            offset,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn grid(
    data: &[u8],
    height: usize,
    width: usize,
    x_map: &[f32],
    y_map: &[f32],
    interpolation: Interpolation,
    border_mode: BorderMode,
    fill: [u8; 3],
    output: &mut [u8],
    scratch: &mut AxisRemapScratch,
) -> CoreResult<()> {
    match interpolation {
        Interpolation::Nearest => {
            prepare_nearest_axes(x_map, &mut scratch.nearest_x)?;
            prepare_nearest_axes(y_map, &mut scratch.nearest_y)?;
            let x = &scratch.nearest_x;
            let y = &scratch.nearest_y;
            for (destination_y, &source_y) in y.iter().enumerate() {
                let mut x_start = 0usize;
                while x_start < width {
                    let inlier = in_bounds(source_y, height) && in_bounds(x[x_start], width);
                    let mut x_end = x_start + 1;
                    while x_end < width
                        && (in_bounds(source_y, height) && in_bounds(x[x_end], width)) == inlier
                    {
                        x_end += 1;
                    }
                    if inlier {
                        for (destination_x, &source_x) in x[x_start..x_end].iter().enumerate() {
                            let destination_x = x_start + destination_x;
                            copy_triplet(
                                data,
                                (source_y as usize * width + source_x as usize) * 3,
                                output,
                                (destination_y * width + destination_x) * 3,
                            );
                        }
                    } else {
                        for destination_x in x_start..x_end {
                            let pixel = nearest_border(
                                data,
                                height,
                                width,
                                x[destination_x],
                                source_y,
                                border_mode,
                                fill,
                            );
                            output[(destination_y * width + destination_x) * 3
                                ..(destination_y * width + destination_x + 1) * 3]
                                .copy_from_slice(&pixel);
                        }
                    }
                    x_start = x_end;
                }
            }
        }
        Interpolation::Bilinear => {
            prepare_bilinear_axes(x_map, &mut scratch.bilinear_x)?;
            prepare_bilinear_axes(y_map, &mut scratch.bilinear_y)?;
            let x = &scratch.bilinear_x;
            let y = &scratch.bilinear_y;
            for (destination_y, &source_y) in y.iter().enumerate() {
                let mut x_start = 0usize;
                while x_start < width {
                    let inlier = bilinear_axis_in_bounds(source_y, height)
                        && bilinear_axis_in_bounds(x[x_start], width);
                    let mut x_end = x_start + 1;
                    while x_end < width
                        && (bilinear_axis_in_bounds(source_y, height)
                            && bilinear_axis_in_bounds(x[x_end], width))
                            == inlier
                    {
                        x_end += 1;
                    }
                    if inlier {
                        for destination_x in x_start..x_end {
                            bilinear_inlier(
                                data,
                                width,
                                x[destination_x],
                                source_y,
                                &mut output[(destination_y * width + destination_x) * 3..][..3],
                            );
                        }
                    } else {
                        for destination_x in x_start..x_end {
                            let pixel = bilinear_border(
                                data,
                                height,
                                width,
                                x[destination_x],
                                source_y,
                                border_mode,
                                fill,
                            );
                            output[(destination_y * width + destination_x) * 3
                                ..(destination_y * width + destination_x + 1) * 3]
                                .copy_from_slice(&pixel);
                        }
                    }
                    x_start = x_end;
                }
            }
        }
    }
    Ok(())
}

fn make_descriptor(source_x: f32, source_y: f32, interpolation: Interpolation) -> Descriptor {
    match interpolation {
        Interpolation::Nearest => Descriptor::Nearest {
            x: source_x.round() as isize,
            y: source_y.round() as isize,
        },
        Interpolation::Bilinear => {
            let x0 = source_x.floor() as isize;
            let y0 = source_y.floor() as isize;
            Descriptor::Bilinear {
                x0,
                y0,
                wx: ((source_x - x0 as f32) * 256.0).floor().clamp(0.0, 255.0) as u32,
                wy: ((source_y - y0 as f32) * 256.0).floor().clamp(0.0, 255.0) as u32,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_descriptors(
    data: &[u8],
    height: usize,
    width: usize,
    descriptors: &[Descriptor],
    output_pixel: usize,
    border_mode: BorderMode,
    fill: [u8; 3],
    output: &mut [u8],
) {
    let mut start = 0usize;
    while start < descriptors.len() {
        let inlier = descriptor_in_bounds(descriptors[start], height, width);
        let mut end = start + 1;
        while end < descriptors.len()
            && descriptor_in_bounds(descriptors[end], height, width) == inlier
        {
            end += 1;
        }
        for (offset, &descriptor) in descriptors[start..end].iter().enumerate() {
            let destination = (output_pixel + start + offset) * 3;
            match descriptor {
                Descriptor::Invalid => output[destination..destination + 3].copy_from_slice(&fill),
                Descriptor::Nearest { x, y } if inlier => copy_triplet(
                    data,
                    (y as usize * width + x as usize) * 3,
                    output,
                    destination,
                ),
                Descriptor::Nearest { x, y } => {
                    let pixel = nearest_border(data, height, width, x, y, border_mode, fill);
                    output[destination..destination + 3].copy_from_slice(&pixel);
                }
                Descriptor::Bilinear { x0, y0, wx, wy } if inlier => bilinear_inlier(
                    data,
                    width,
                    BilinearAxis {
                        low: x0,
                        weight: wx,
                    },
                    BilinearAxis {
                        low: y0,
                        weight: wy,
                    },
                    &mut output[destination..destination + 3],
                ),
                Descriptor::Bilinear { x0, y0, wx, wy } => {
                    let pixel = bilinear_border(
                        data,
                        height,
                        width,
                        BilinearAxis {
                            low: x0,
                            weight: wx,
                        },
                        BilinearAxis {
                            low: y0,
                            weight: wy,
                        },
                        border_mode,
                        fill,
                    );
                    output[destination..destination + 3].copy_from_slice(&pixel);
                }
            }
        }
        start = end;
    }
}

fn descriptor_in_bounds(descriptor: Descriptor, height: usize, width: usize) -> bool {
    match descriptor {
        Descriptor::Invalid => false,
        Descriptor::Nearest { x, y } => in_bounds(x, width) && in_bounds(y, height),
        Descriptor::Bilinear { x0, y0, .. } => {
            bilinear_axis_in_bounds(BilinearAxis { low: x0, weight: 0 }, width)
                && bilinear_axis_in_bounds(BilinearAxis { low: y0, weight: 0 }, height)
        }
    }
}

fn prepare_nearest_axes(map: &[f32], axes: &mut Vec<isize>) -> CoreResult<()> {
    axes.clear();
    axes.try_reserve(map.len())
        .map_err(|_| CoreError::Runtime("axis remap allocation failed".into()))?;
    axes.extend(map.iter().map(|value| value.round() as isize));
    Ok(())
}

fn prepare_bilinear_axes(map: &[f32], axes: &mut Vec<BilinearAxis>) -> CoreResult<()> {
    axes.clear();
    axes.try_reserve(map.len())
        .map_err(|_| CoreError::Runtime("axis remap allocation failed".into()))?;
    axes.extend(map.iter().map(|&value| {
        let low = value.floor() as isize;
        BilinearAxis {
            low,
            weight: ((value - low as f32) * 256.0).floor().clamp(0.0, 255.0) as u32,
        }
    }));
    Ok(())
}

#[inline]
fn in_bounds(value: isize, length: usize) -> bool {
    value >= 0 && (value as usize) < length
}

#[inline]
fn bilinear_axis_in_bounds(axis: BilinearAxis, length: usize) -> bool {
    axis.low >= 0 && (axis.low as usize).saturating_add(1) < length
}

fn copy_triplet(data: &[u8], source: usize, output: &mut [u8], destination: usize) {
    output[destination..destination + 3].copy_from_slice(&data[source..source + 3]);
}

fn nearest_border(
    data: &[u8],
    height: usize,
    width: usize,
    x: isize,
    y: isize,
    border_mode: BorderMode,
    fill: [u8; 3],
) -> [u8; 3] {
    let Some((x, y)) = resolve(x, y, width, height, border_mode) else {
        return fill;
    };
    let source = (y * width + x) * 3;
    [data[source], data[source + 1], data[source + 2]]
}

fn bilinear_inlier(data: &[u8], width: usize, x: BilinearAxis, y: BilinearAxis, output: &mut [u8]) {
    let x0 = x.low as usize;
    let y0 = y.low as usize;
    let offsets = [
        (y0 * width + x0) * 3,
        (y0 * width + x0 + 1) * 3,
        ((y0 + 1) * width + x0) * 3,
        ((y0 + 1) * width + x0 + 1) * 3,
    ];
    affine::bilinear_rgb(data, offsets, x.weight, y.weight, output);
}

#[allow(clippy::too_many_arguments)]
fn bilinear_border(
    data: &[u8],
    height: usize,
    width: usize,
    x: BilinearAxis,
    y: BilinearAxis,
    border_mode: BorderMode,
    fill: [u8; 3],
) -> [u8; 3] {
    let x1 = x.low.saturating_add(1);
    let y1 = y.low.saturating_add(1);
    let taps = [
        resolve(x.low, y.low, width, height, border_mode),
        resolve(x1, y.low, width, height, border_mode),
        resolve(x.low, y1, width, height, border_mode),
        resolve(x1, y1, width, height, border_mode),
    ];
    let inv_wx = 256 - x.weight;
    let inv_wy = 256 - y.weight;
    let mut output = [0; 3];
    for channel in 0..3 {
        let value = |tap: Option<(usize, usize)>| {
            tap.map_or(fill[channel], |(x, y)| data[(y * width + x) * 3 + channel])
        };
        let top = u32::from(value(taps[0])) * inv_wx + u32::from(value(taps[1])) * x.weight;
        let bottom = u32::from(value(taps[2])) * inv_wx + u32::from(value(taps[3])) * x.weight;
        output[channel] = ((top * inv_wy + bottom * y.weight + 32768) >> 16) as u8;
    }
    output
}

fn resolve(
    x: isize,
    y: isize,
    width: usize,
    height: usize,
    border_mode: BorderMode,
) -> Option<(usize, usize)> {
    match border_mode {
        BorderMode::Constant if !in_bounds(x, width) || !in_bounds(y, height) => None,
        BorderMode::Constant => Some((x as usize, y as usize)),
        BorderMode::Reflect101 => Some((reflect101(x, width), reflect101(y, height))),
    }
}

#[inline]
fn reflect101(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length - 1) as isize;
    let reflected = index.rem_euclid(period);
    if reflected < length as isize {
        reflected as usize
    } else {
        (period - reflected) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perspective_descriptor_dispatch_matches_scalar() {
        for count in 1..=BLOCK {
            for interpolation in [Interpolation::Nearest, Interpolation::Bilinear] {
                let inverse = [0.93, 0.07, -1.3, -0.04, 1.08, 0.6, 0.0007, -0.0004, 1.0];
                let mut expected = vec![Descriptor::Invalid; count];
                let mut actual = vec![Descriptor::Invalid; count];
                perspective_descriptors_scalar(inverse, 17, 13, interpolation, &mut expected, 0);
                perspective_descriptors(inverse, 17, 13, interpolation, &mut actual);
                assert_eq!(actual, expected, "count={count} {interpolation:?}");
            }
        }
    }
}
