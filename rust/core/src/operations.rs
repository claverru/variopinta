use crate::kernels::{affine, blur, color};
use crate::plan::{
    AffineSample, ColorJitterSample, CropSample, GaussianNoiseSample, GridDistortionSample,
    PerspectiveSample, SharpenSample,
};
use crate::{BorderMode, CoreError, CoreResult, Interpolation};
use fast_image_resize as fir;
use fir::images::{Image as FirImage, ImageRef as FirImageRef};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub(crate) const MAX_AFFINE_DIMENSION: usize = 1 << 24;

pub(crate) struct ImageU8 {
    pub(crate) data: Vec<u8>,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

pub(crate) fn rgb_len(height: usize, width: usize) -> CoreResult<usize> {
    height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| CoreError::Invalid("image dimensions overflow".into()))
}

pub(crate) fn random_crop_raw_into(
    data_in: &[u8],
    input_h: usize,
    input_w: usize,
    crop: CropSample,
    mut data: Vec<u8>,
) -> CoreResult<ImageU8> {
    if crop.top + crop.height > input_h || crop.left + crop.width > input_w {
        return Err(CoreError::Runtime("sampled crop exceeds input".into()));
    }
    let row_bytes = crop.width * 3;
    data.resize(crop.height * row_bytes, 0);
    for y in 0..crop.height {
        let src = ((crop.top + y) * input_w + crop.left) * 3;
        let dst = y * row_bytes;
        data[dst..dst + row_bytes].copy_from_slice(&data_in[src..src + row_bytes]);
    }
    Ok(ImageU8 {
        data,
        height: crop.height,
        width: crop.width,
    })
}

pub(crate) fn pad_raw(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    sample: crate::plan::PadSample,
    mut output: Vec<u8>,
) -> CoreResult<ImageU8> {
    let output_len = rgb_len(sample.height, sample.width)?;
    if sample.top.saturating_add(input_height) > sample.height
        || sample.left.saturating_add(input_width) > sample.width
        || input.len() != rgb_len(input_height, input_width)?
    {
        return Err(CoreError::Runtime(
            "invalid sampled padding geometry".into(),
        ));
    }
    output.resize(output_len, 0);
    match sample.border_mode {
        BorderMode::Constant => {
            for pixel in output.chunks_exact_mut(3) {
                pixel.copy_from_slice(&sample.fill);
            }
            let row_bytes = input_width * 3;
            for y in 0..input_height {
                let source = y * row_bytes;
                let destination = ((sample.top + y) * sample.width + sample.left) * 3;
                output[destination..destination + row_bytes]
                    .copy_from_slice(&input[source..source + row_bytes]);
            }
        }
        BorderMode::Reflect101 => {
            for y in 0..sample.height {
                let source_y = reflect101_index(y as isize - sample.top as isize, input_height);
                for x in 0..sample.width {
                    let source_x = reflect101_index(x as isize - sample.left as isize, input_width);
                    let source = (source_y * input_width + source_x) * 3;
                    let destination = (y * sample.width + x) * 3;
                    output[destination..destination + 3]
                        .copy_from_slice(&input[source..source + 3]);
                }
            }
        }
    }
    Ok(ImageU8 {
        data: output,
        height: sample.height,
        width: sample.width,
    })
}

pub(crate) fn coarse_dropout(
    image: &mut ImageU8,
    holes: &[crate::plan::DropoutHole],
    fill: [u8; 3],
) -> CoreResult<()> {
    for hole in holes {
        if hole.top.saturating_add(hole.height) > image.height
            || hole.left.saturating_add(hole.width) > image.width
        {
            return Err(CoreError::Runtime(
                "sampled dropout hole exceeds input".into(),
            ));
        }
        for y in hole.top..hole.top + hole.height {
            let start = (y * image.width + hole.left) * 3;
            let end = start + hole.width * 3;
            for pixel in image.data[start..end].chunks_exact_mut(3) {
                pixel.copy_from_slice(&fill);
            }
        }
    }
    Ok(())
}

pub(crate) fn copy_u8(data: &[u8]) -> CoreResult<Vec<u8>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(data.len())
        .map_err(|_| CoreError::Runtime("output allocation failed".into()))?;
    copy.extend_from_slice(data);
    Ok(copy)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resize_raw(
    data: &[u8],
    input_h: usize,
    input_w: usize,
    out_h: usize,
    out_w: usize,
    interpolation: Interpolation,
    antialias: bool,
    resizer: &mut fir::Resizer,
    destination: Vec<u8>,
) -> CoreResult<ImageU8> {
    if out_h == 0 || out_w == 0 {
        return Err(CoreError::Invalid(
            "resize dimensions must be positive".into(),
        ));
    }
    let src = FirImageRef::new(input_w as u32, input_h as u32, data, fir::PixelType::U8x3)
        .map_err(|error| CoreError::Invalid(format!("invalid RGB buffer: {error}")))?;
    let mut dst = FirImage::from_vec_u8(
        out_w as u32,
        out_h as u32,
        destination,
        fir::PixelType::U8x3,
    )
    .map_err(|error| CoreError::Runtime(format!("invalid resize destination: {error}")))?;
    let algorithm = match interpolation {
        Interpolation::Nearest => fir::ResizeAlg::Nearest,
        Interpolation::Bilinear if antialias => {
            fir::ResizeAlg::Convolution(fir::FilterType::Bilinear)
        }
        Interpolation::Bilinear => fir::ResizeAlg::Interpolation(fir::FilterType::Bilinear),
    };
    let options = fir::ResizeOptions::new().resize_alg(algorithm);
    resizer
        .resize(&src, &mut dst, &options)
        .map_err(|error| CoreError::Runtime(format!("resize failed: {error}")))?;
    Ok(ImageU8 {
        data: dst.into_vec(),
        height: out_h,
        width: out_w,
    })
}

pub(crate) fn color_jitter(image: &mut ImageU8, sample: &ColorJitterSample) {
    let bf = sample.brightness;
    let cf = sample.contrast;
    let sf = sample.saturation;

    let pixels = image.height * image.width;
    let mut sums = [0u64; 3];
    for px in image.data.chunks_exact(3) {
        sums[0] += px[0] as u64;
        sums[1] += px[1] as u64;
        sums[2] += px[2] as u64;
    }
    let source_mean = sums.map(|sum| sum as f32 / pixels as f32);
    let mut matrix = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut offset = [0.0f32; 3];
    for operation in sample.order {
        match operation {
            0 => {
                for row in 0..3 {
                    for value in &mut matrix[row] {
                        *value *= bf;
                    }
                    offset[row] *= bf;
                }
            }
            1 => {
                let current_mean = matrix.map(|row| {
                    row[0] * source_mean[0] + row[1] * source_mean[1] + row[2] * source_mean[2]
                });
                let luminance = 0.299 * (current_mean[0] + offset[0])
                    + 0.587 * (current_mean[1] + offset[1])
                    + 0.114 * (current_mean[2] + offset[2]);
                for row in 0..3 {
                    for value in &mut matrix[row] {
                        *value *= cf;
                    }
                    offset[row] = offset[row] * cf + luminance * (1.0 - cf);
                }
            }
            2 => {
                let gray = [0.299 * (1.0 - sf), 0.587 * (1.0 - sf), 0.114 * (1.0 - sf)];
                let saturation_matrix = [
                    [sf + gray[0], gray[1], gray[2]],
                    [gray[0], sf + gray[1], gray[2]],
                    [gray[0], gray[1], sf + gray[2]],
                ];
                let old_matrix = matrix;
                let old_offset = offset;
                for row in 0..3 {
                    for column in 0..3 {
                        matrix[row][column] = saturation_matrix[row][0] * old_matrix[0][column]
                            + saturation_matrix[row][1] * old_matrix[1][column]
                            + saturation_matrix[row][2] * old_matrix[2][column];
                    }
                    offset[row] = saturation_matrix[row][0] * old_offset[0]
                        + saturation_matrix[row][1] * old_offset[1]
                        + saturation_matrix[row][2] * old_offset[2];
                }
            }
            3 => {}
            _ => unreachable!(),
        }
    }

    if !try_apply_color_matrix_q14(image, matrix, offset[0]) {
        let source_mean_f64 = sums.map(|sum| sum as f64 / pixels as f64);
        let (safe_matrix, safe_offset) = compose_color_matrix_f64(sample, source_mean_f64);
        apply_color_matrix_f64(&mut image.data, safe_matrix, safe_offset[0]);
    }
}

fn compose_color_matrix_f64(
    sample: &ColorJitterSample,
    source_mean: [f64; 3],
) -> ([[f64; 3]; 3], [f64; 3]) {
    let bf = f64::from(sample.brightness);
    let cf = f64::from(sample.contrast);
    let sf = f64::from(sample.saturation);
    let mut matrix = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut offset = [0.0_f64; 3];
    for operation in sample.order {
        match operation {
            0 => {
                for row in 0..3 {
                    for value in &mut matrix[row] {
                        *value *= bf;
                    }
                    offset[row] *= bf;
                }
            }
            1 => {
                let current_mean = matrix.map(|row| {
                    row[0] * source_mean[0] + row[1] * source_mean[1] + row[2] * source_mean[2]
                });
                let luminance = 0.299 * (current_mean[0] + offset[0])
                    + 0.587 * (current_mean[1] + offset[1])
                    + 0.114 * (current_mean[2] + offset[2]);
                for row in 0..3 {
                    for value in &mut matrix[row] {
                        *value *= cf;
                    }
                    offset[row] = offset[row] * cf + luminance * (1.0 - cf);
                }
            }
            2 => {
                let gray = [0.299 * (1.0 - sf), 0.587 * (1.0 - sf), 0.114 * (1.0 - sf)];
                let saturation_matrix = [
                    [sf + gray[0], gray[1], gray[2]],
                    [gray[0], sf + gray[1], gray[2]],
                    [gray[0], gray[1], sf + gray[2]],
                ];
                let old_matrix = matrix;
                let old_offset = offset;
                for row in 0..3 {
                    for column in 0..3 {
                        matrix[row][column] = saturation_matrix[row][0] * old_matrix[0][column]
                            + saturation_matrix[row][1] * old_matrix[1][column]
                            + saturation_matrix[row][2] * old_matrix[2][column];
                    }
                    offset[row] = saturation_matrix[row][0] * old_offset[0]
                        + saturation_matrix[row][1] * old_offset[1]
                        + saturation_matrix[row][2] * old_offset[2];
                }
            }
            3 => {}
            _ => unreachable!(),
        }
    }
    (matrix, offset)
}

pub(crate) fn color_jitter_staged(image: &mut ImageU8, sample: &ColorJitterSample) {
    let bf = sample.brightness;
    let cf = sample.contrast;
    let sf = sample.saturation;

    for operation in sample.order {
        match operation {
            0 => apply_color_matrix(image, [[bf, 0.0, 0.0], [0.0, bf, 0.0], [0.0, 0.0, bf]], 0.0),
            1 => {
                let mut sums = [0u64; 3];
                for pixel in image.data.chunks_exact(3) {
                    sums[0] += pixel[0] as u64;
                    sums[1] += pixel[1] as u64;
                    sums[2] += pixel[2] as u64;
                }
                let pixels = (image.height * image.width) as f32;
                let luminance = 0.299 * sums[0] as f32 / pixels
                    + 0.587 * sums[1] as f32 / pixels
                    + 0.114 * sums[2] as f32 / pixels;
                let matrix = [[cf, 0.0, 0.0], [0.0, cf, 0.0], [0.0, 0.0, cf]];
                let safe_factor = f64::from(cf);
                let safe_pixels = (image.height * image.width) as f64;
                let safe_luminance =
                    (0.299 * sums[0] as f64 + 0.587 * sums[1] as f64 + 0.114 * sums[2] as f64)
                        / safe_pixels;
                apply_color_matrix_with_safe(
                    image,
                    matrix,
                    luminance * (1.0 - cf),
                    matrix.map(|row| row.map(f64::from)),
                    safe_luminance * (1.0 - safe_factor),
                );
            }
            2 => {
                let gray = [0.299 * (1.0 - sf), 0.587 * (1.0 - sf), 0.114 * (1.0 - sf)];
                apply_color_matrix(
                    image,
                    [
                        [sf + gray[0], gray[1], gray[2]],
                        [gray[0], sf + gray[1], gray[2]],
                        [gray[0], gray[1], sf + gray[2]],
                    ],
                    0.0,
                );
            }
            3 => adjust_hue(image, sample.hue),
            _ => unreachable!(),
        }
    }
}

pub(crate) fn gaussian_noise(image: &mut ImageU8, sample: GaussianNoiseSample) {
    let mut rng = SmallRng::seed_from_u64(sample.seed);
    for pixel in image.data.chunks_exact_mut(3) {
        if sample.per_channel {
            for channel in pixel {
                *channel = noisy_value(*channel, sample.mean, sample.std, &mut rng);
            }
        } else {
            let noise = normal_sample(&mut rng) * sample.std + sample.mean;
            for channel in pixel {
                *channel = (f32::from(*channel) + noise).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn noisy_value(value: u8, mean: f32, std: f32, rng: &mut SmallRng) -> u8 {
    (f32::from(value) + mean + normal_sample(rng) * std)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn normal_sample(rng: &mut SmallRng) -> f32 {
    let u1 = 1.0 - rng.random::<f32>();
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

pub(crate) fn sharpen_raw(
    data: &[u8],
    height: usize,
    width: usize,
    sample: SharpenSample,
    mut output: Vec<u8>,
) -> CoreResult<ImageU8> {
    let expected = rgb_len(height, width)?;
    if data.len() != expected || output.len() != expected {
        return Err(CoreError::Runtime(
            "sharpen requires matching RGB buffers".into(),
        ));
    }
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let center = f32::from(data[(y * width + x) * 3 + channel]);
                let neighbors = [
                    border_sample(
                        data,
                        height,
                        width,
                        x as isize,
                        y as isize - 1,
                        channel,
                        BorderMode::Reflect101,
                        [0; 3],
                    ),
                    border_sample(
                        data,
                        height,
                        width,
                        x as isize,
                        y as isize + 1,
                        channel,
                        BorderMode::Reflect101,
                        [0; 3],
                    ),
                    border_sample(
                        data,
                        height,
                        width,
                        x as isize - 1,
                        y as isize,
                        channel,
                        BorderMode::Reflect101,
                        [0; 3],
                    ),
                    border_sample(
                        data,
                        height,
                        width,
                        x as isize + 1,
                        y as isize,
                        channel,
                        BorderMode::Reflect101,
                        [0; 3],
                    ),
                ];
                let sharpened = center * (1.0 + 4.0 * sample.lightness)
                    - neighbors.iter().map(|value| f32::from(*value)).sum::<f32>()
                        * sample.lightness;
                output[(y * width + x) * 3 + channel] =
                    (center + sample.alpha * (sharpened - center))
                        .round()
                        .clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(ImageU8 {
        data: output,
        height,
        width,
    })
}

pub(crate) fn perspective_raw(
    data: &[u8],
    height: usize,
    width: usize,
    sample: PerspectiveSample,
    output: Vec<u8>,
) -> CoreResult<ImageU8> {
    remap_raw(
        data,
        height,
        width,
        sample.interpolation,
        sample.border_mode,
        sample.fill,
        output,
        |y, x| {
            let x = x as f32;
            let y = y as f32;
            let denominator = sample.inverse[6] * x + sample.inverse[7] * y + sample.inverse[8];
            if !denominator.is_finite() || denominator.abs() < 1e-8 {
                return None;
            }
            let source_x =
                (sample.inverse[0] * x + sample.inverse[1] * y + sample.inverse[2]) / denominator;
            let source_y =
                (sample.inverse[3] * x + sample.inverse[4] * y + sample.inverse[5]) / denominator;
            (source_x.is_finite() && source_y.is_finite()).then_some((source_y, source_x))
        },
    )
}

pub(crate) fn grid_distortion_raw(
    data: &[u8],
    height: usize,
    width: usize,
    sample: &GridDistortionSample,
    output: Vec<u8>,
) -> CoreResult<ImageU8> {
    if sample.x_map.len() != width || sample.y_map.len() != height {
        return Err(CoreError::Runtime(
            "grid maps must match the image dimensions".into(),
        ));
    }
    remap_raw(
        data,
        height,
        width,
        sample.interpolation,
        sample.border_mode,
        sample.fill,
        output,
        |y, x| Some((sample.y_map[y], sample.x_map[x])),
    )
}

#[allow(clippy::too_many_arguments)]
fn remap_raw(
    data: &[u8],
    height: usize,
    width: usize,
    interpolation: Interpolation,
    border_mode: BorderMode,
    fill: [u8; 3],
    mut output: Vec<u8>,
    coordinates: impl Fn(usize, usize) -> Option<(f32, f32)>,
) -> CoreResult<ImageU8> {
    let expected = rgb_len(height, width)?;
    if data.len() != expected || output.len() != expected {
        return Err(CoreError::Runtime(
            "remapping requires matching RGB buffers".into(),
        ));
    }
    for y in 0..height {
        for x in 0..width {
            let destination = (y * width + x) * 3;
            let Some((source_y, source_x)) = coordinates(y, x) else {
                output[destination..destination + 3].copy_from_slice(&fill);
                continue;
            };
            match interpolation {
                Interpolation::Nearest => {
                    let source_y = source_y.round() as isize;
                    let source_x = source_x.round() as isize;
                    for channel in 0..3 {
                        output[destination + channel] = border_sample(
                            data,
                            height,
                            width,
                            source_x,
                            source_y,
                            channel,
                            border_mode,
                            fill,
                        );
                    }
                }
                Interpolation::Bilinear => {
                    let y0 = source_y.floor() as isize;
                    let x0 = source_x.floor() as isize;
                    let wx = ((source_x - x0 as f32) * 256.0).floor().clamp(0.0, 255.0) as u32;
                    let wy = ((source_y - y0 as f32) * 256.0).floor().clamp(0.0, 255.0) as u32;
                    let inv_wx = 256 - wx;
                    let inv_wy = 256 - wy;
                    for channel in 0..3 {
                        let p00 = u32::from(border_sample(
                            data,
                            height,
                            width,
                            x0,
                            y0,
                            channel,
                            border_mode,
                            fill,
                        ));
                        let p01 = u32::from(border_sample(
                            data,
                            height,
                            width,
                            x0 + 1,
                            y0,
                            channel,
                            border_mode,
                            fill,
                        ));
                        let p10 = u32::from(border_sample(
                            data,
                            height,
                            width,
                            x0,
                            y0 + 1,
                            channel,
                            border_mode,
                            fill,
                        ));
                        let p11 = u32::from(border_sample(
                            data,
                            height,
                            width,
                            x0 + 1,
                            y0 + 1,
                            channel,
                            border_mode,
                            fill,
                        ));
                        let top = p00 * inv_wx + p01 * wx;
                        let bottom = p10 * inv_wx + p11 * wx;
                        output[destination + channel] =
                            ((top * inv_wy + bottom * wy + 32768) >> 16) as u8;
                    }
                }
            }
        }
    }
    Ok(ImageU8 {
        data: output,
        height,
        width,
    })
}

pub(crate) fn adjust_hue(image: &mut ImageU8, factor: f32) {
    if factor == 0.0 {
        return;
    }
    for pixel in image.data.chunks_exact_mut(3) {
        let r = f32::from(pixel[0]) / 255.0;
        let g = f32::from(pixel[1]) / 255.0;
        let b = f32::from(pixel[2]) / 255.0;
        let maximum = r.max(g).max(b);
        let minimum = r.min(g).min(b);
        let delta = maximum - minimum;
        let hue = if delta == 0.0 {
            0.0
        } else if maximum == r {
            ((g - b) / delta).rem_euclid(6.0) / 6.0
        } else if maximum == g {
            ((b - r) / delta + 2.0) / 6.0
        } else {
            ((r - g) / delta + 4.0) / 6.0
        };
        let hue = (hue + factor).rem_euclid(1.0);
        let saturation = if maximum == 0.0 { 0.0 } else { delta / maximum };
        let sector_value = hue * 6.0;
        let sector_floor = sector_value.floor();
        let sector = sector_floor as u8 % 6;
        let fraction = sector_value - sector_floor;
        let p = maximum * (1.0 - saturation);
        let q = maximum * (1.0 - saturation * fraction);
        let t = maximum * (1.0 - saturation * (1.0 - fraction));
        let (r, g, b) = match sector {
            0 => (maximum, t, p),
            1 => (q, maximum, p),
            2 => (p, maximum, t),
            3 => (p, q, maximum),
            4 => (t, p, maximum),
            _ => (maximum, p, q),
        };
        pixel[0] = (r * 255.0).round().clamp(0.0, 255.0) as u8;
        pixel[1] = (g * 255.0).round().clamp(0.0, 255.0) as u8;
        pixel[2] = (b * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

pub(crate) fn apply_color_matrix(image: &mut ImageU8, matrix: [[f32; 3]; 3], bias: f32) {
    apply_color_matrix_with_safe(
        image,
        matrix,
        bias,
        matrix.map(|row| row.map(f64::from)),
        f64::from(bias),
    );
}

fn apply_color_matrix_with_safe(
    image: &mut ImageU8,
    matrix: [[f32; 3]; 3],
    bias: f32,
    safe_matrix: [[f64; 3]; 3],
    safe_bias: f64,
) {
    if !try_apply_color_matrix_q14(image, matrix, bias) {
        apply_color_matrix_f64(&mut image.data, safe_matrix, safe_bias);
    }
}

fn try_apply_color_matrix_q14(image: &mut ImageU8, matrix: [[f32; 3]; 3], bias: f32) -> bool {
    let Some((matrix, bias)) = quantize_safe_q14(matrix, bias) else {
        return false;
    };
    color::apply_q14(&mut image.data, matrix, bias);
    true
}

fn quantize_safe_q14(matrix: [[f32; 3]; 3], bias: f32) -> Option<([[i32; 3]; 3], i32)> {
    const Q: f32 = 16384.0;
    let quantize = |value: f32| {
        let rounded = (value * Q).round();
        (rounded.is_finite()
            && f64::from(rounded) >= f64::from(i32::MIN)
            && f64::from(rounded) <= f64::from(i32::MAX))
        .then_some(rounded as i32)
    };
    let mut quantized = [[0_i32; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            quantized[row][column] = quantize(matrix[row][column])?;
        }
    }
    let bias = quantize(bias)?;
    color::q14_accumulator_fits(quantized, bias).then_some((quantized, bias))
}

fn apply_color_matrix_f64(data: &mut [u8], matrix: [[f64; 3]; 3], bias: f64) {
    for pixel in data.chunks_exact_mut(3) {
        let source = [
            f64::from(pixel[0]),
            f64::from(pixel[1]),
            f64::from(pixel[2]),
        ];
        for channel in 0..3 {
            let value = matrix[channel][0] * source[0]
                + matrix[channel][1] * source[1]
                + matrix[channel][2] * source[2]
                + bias;
            pixel[channel] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
}

pub(crate) fn rotate_raw(
    data: &[u8],
    height: usize,
    width: usize,
    sample: AffineSample,
    destination: Vec<u8>,
) -> CoreResult<ImageU8> {
    if height > MAX_AFFINE_DIMENSION || width > MAX_AFFINE_DIMENSION {
        return Err(CoreError::Invalid(format!(
            "Affine image dimensions must not exceed {MAX_AFFINE_DIMENSION} per axis"
        )));
    }
    let matrix = inverse_affine_matrix(width, height, sample);
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(CoreError::Invalid(
            "Affine parameters produce non-finite coordinates".into(),
        ));
    }
    if sample.interpolation == Interpolation::Nearest {
        let mut destination = destination;
        rotate_nearest(data, height, width, sample, matrix, &mut destination);
        return Ok(ImageU8 {
            data: destination,
            height,
            width,
        });
    }
    if sample.border_mode == BorderMode::Reflect101 {
        let mut destination = destination;
        rotate_bilinear_border(data, height, width, sample, matrix, &mut destination, false);
        return Ok(ImageU8 {
            data: destination,
            height,
            width,
        });
    }
    let mut image = ImageU8 {
        data: affine::bilinear_constant(data, height, width, matrix, destination)?,
        height,
        width,
    };
    rotate_bilinear_border(data, height, width, sample, matrix, &mut image.data, true);
    Ok(image)
}

pub(crate) fn inverse_affine_matrix(width: usize, height: usize, sample: AffineSample) -> [f32; 6] {
    let rotation = sample.degrees.to_radians();
    let shear_x = sample.shear[0].to_radians();
    let shear_y = sample.shear[1].to_radians();
    let cos_shear_y = shear_y.cos();
    let a = (rotation - shear_y).cos() / cos_shear_y;
    let b = -(rotation - shear_y).cos() * shear_x.tan() / cos_shear_y - rotation.sin();
    let c = (rotation - shear_y).sin() / cos_shear_y;
    let d = -(rotation - shear_y).sin() * shear_x.tan() / cos_shear_y + rotation.cos();
    let inverse_scale = sample.scale.recip();
    let mut matrix = [
        d * inverse_scale,
        -b * inverse_scale,
        0.0,
        -c * inverse_scale,
        a * inverse_scale,
        0.0,
    ];
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    matrix[2] =
        matrix[0] * (-cx - sample.translate[0]) + matrix[1] * (-cy - sample.translate[1]) + cx;
    matrix[5] =
        matrix[3] * (-cx - sample.translate[0]) + matrix[4] * (-cy - sample.translate[1]) + cy;
    matrix
}

pub(crate) fn source_coordinates(y: usize, matrix: [f32; 6]) -> (f64, f64, f64, f64) {
    (
        f64::from(matrix[1]) * y as f64 + f64::from(matrix[2]),
        f64::from(matrix[4]) * y as f64 + f64::from(matrix[5]),
        f64::from(matrix[0]),
        f64::from(matrix[3]),
    )
}

pub(crate) fn rotate_nearest(
    data: &[u8],
    height: usize,
    width: usize,
    sample: AffineSample,
    matrix: [f32; 6],
    output: &mut [u8],
) {
    for y in 0..height {
        let (sx0, sy0, dsx, dsy) = source_coordinates(y, matrix);
        for x in 0..width {
            let sx = (sx0 + dsx * x as f64).round() as isize;
            let sy = (sy0 + dsy * x as f64).round() as isize;
            let destination = (y * width + x) * 3;
            for channel in 0..3 {
                output[destination + channel] = border_sample(
                    data,
                    height,
                    width,
                    sx,
                    sy,
                    channel,
                    sample.border_mode,
                    sample.fill,
                );
            }
        }
    }
}

pub(crate) fn rotate_bilinear_border(
    data: &[u8],
    height: usize,
    width: usize,
    sample: AffineSample,
    matrix: [f32; 6],
    output: &mut [u8],
    border_only: bool,
) {
    for y in 0..height {
        let (sx0, sy0, dsx, dsy) = source_coordinates(y, matrix);
        let mut render = |x: usize| {
            let sx = sx0 + dsx * x as f64;
            let sy = sy0 + dsy * x as f64;
            let x0 = sx.floor() as isize;
            let y0 = sy.floor() as isize;
            let wx = ((sx - sx.floor()) * 256.0) as u32;
            let wy = ((sy - sy.floor()) * 256.0) as u32;
            let destination = (y * width + x) * 3;
            for channel in 0..3 {
                let top = u32::from(border_sample(
                    data,
                    height,
                    width,
                    x0,
                    y0,
                    channel,
                    sample.border_mode,
                    sample.fill,
                )) * (256 - wx)
                    + u32::from(border_sample(
                        data,
                        height,
                        width,
                        x0 + 1,
                        y0,
                        channel,
                        sample.border_mode,
                        sample.fill,
                    )) * wx;
                let bottom = u32::from(border_sample(
                    data,
                    height,
                    width,
                    x0,
                    y0 + 1,
                    channel,
                    sample.border_mode,
                    sample.fill,
                )) * (256 - wx)
                    + u32::from(border_sample(
                        data,
                        height,
                        width,
                        x0 + 1,
                        y0 + 1,
                        channel,
                        sample.border_mode,
                        sample.fill,
                    )) * wx;
                output[destination + channel] =
                    ((top * (256 - wy) + bottom * wy + 32768) >> 16) as u8;
            }
        };
        if border_only {
            let (mut start, mut end) = (0, width as i64);
            affine::valid_span(
                &mut start,
                &mut end,
                sx0,
                dsx,
                width.saturating_sub(1) as f64,
                width,
            );
            affine::valid_span(
                &mut start,
                &mut end,
                sy0,
                dsy,
                height.saturating_sub(1) as f64,
                width,
            );
            for x in 0..start as usize {
                render(x);
            }
            for x in end as usize..width {
                render(x);
            }
        } else {
            for x in 0..width {
                render(x);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn border_sample(
    data: &[u8],
    height: usize,
    width: usize,
    x: isize,
    y: isize,
    channel: usize,
    border_mode: BorderMode,
    fill: [u8; 3],
) -> u8 {
    let position = match border_mode {
        BorderMode::Constant if x < 0 || y < 0 || x >= width as isize || y >= height as isize => {
            return fill[channel];
        }
        BorderMode::Constant => (y as usize, x as usize),
        BorderMode::Reflect101 => (reflect101_index(y, height), reflect101_index(x, width)),
    };
    data[(position.0 * width + position.1) * 3 + channel]
}

pub(crate) fn gaussian_blur_in_place(
    image: &mut ImageU8,
    kernel: &[u16],
    temp: &mut Vec<u16>,
) -> CoreResult<()> {
    if temp.capacity() < image.data.len() {
        temp.try_reserve_exact(image.data.len() - temp.len())
            .map_err(|_| CoreError::Runtime("blur workspace allocation failed".into()))?;
    }
    temp.resize(image.data.len(), 0);
    if kernel.len() == 5 && image.width >= 5 && image.height >= 5 {
        gaussian_blur_5x5_q8(&mut image.data, image.height, image.width, kernel, temp)?;
        return Ok(());
    }

    let radius = kernel.len() / 2;
    for y in 0..image.height {
        for x in 0..image.width {
            for c in 0..3 {
                let mut acc = 0u32;
                for (k, &weight) in kernel.iter().enumerate() {
                    let xx =
                        reflect101_index(x as isize + k as isize - radius as isize, image.width);
                    acc += image.data[(y * image.width + xx) * 3 + c] as u32 * weight as u32;
                }
                temp[(y * image.width + x) * 3 + c] = acc as u16;
            }
        }
    }
    for y in 0..image.height {
        for x in 0..image.width {
            for c in 0..3 {
                let mut acc = 0u32;
                for (k, &weight) in kernel.iter().enumerate() {
                    let yy =
                        reflect101_index(y as isize + k as isize - radius as isize, image.height);
                    acc += temp[(yy * image.width + x) * 3 + c] as u32 * weight as u32;
                }
                image.data[(y * image.width + x) * 3 + c] = ((acc + 32768) >> 16).min(255) as u8;
            }
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn reflect101_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1) as isize;
    let reflected = index.rem_euclid(period);
    if reflected < len as isize {
        reflected as usize
    } else {
        (period - reflected) as usize
    }
}

pub(crate) fn gaussian_blur_5x5_q8(
    data: &mut [u8],
    height: usize,
    width: usize,
    kernel: &[u16],
    temp: &mut [u16],
) -> CoreResult<()> {
    let expected = rgb_len(height, width)?;
    if height < 5
        || width < 5
        || kernel.len() != 5
        || data.len() != expected
        || temp.len() < expected
    {
        return Err(CoreError::Runtime(
            "5x5 blur requires matching RGB buffers and five weights".into(),
        ));
    }
    let bytes_per_row = width * 3;
    let (k0, k1, k2, k3, k4) = (
        kernel[0] as u32,
        kernel[1] as u32,
        kernel[2] as u32,
        kernel[3] as u32,
        kernel[4] as u32,
    );
    for y in 0..height {
        let row = y * bytes_per_row;
        for c in 0..3 {
            temp[row + c] = (data[row + 6 + c] as u32 * k0
                + data[row + 3 + c] as u32 * k1
                + data[row + c] as u32 * k2
                + data[row + 3 + c] as u32 * k3
                + data[row + 6 + c] as u32 * k4) as u16;
            temp[row + 3 + c] = (data[row + 3 + c] as u32 * k0
                + data[row + c] as u32 * k1
                + data[row + 3 + c] as u32 * k2
                + data[row + 6 + c] as u32 * k3
                + data[row + 9 + c] as u32 * k4) as u16;
        }
        let start = row + 6;
        let end = row + bytes_per_row - 6;
        blur::horizontal_5x5(data, temp, start, end, [k0, k1, k2, k3, k4])?;
        for c in 0..3 {
            let last = row + bytes_per_row - 3 + c;
            temp[last - 3] = (data[last - 9] as u32 * k0
                + data[last - 6] as u32 * k1
                + data[last - 3] as u32 * k2
                + data[last] as u32 * k3
                + data[last - 3] as u32 * k4) as u16;
            temp[last] = (data[last - 6] as u32 * k0
                + data[last - 3] as u32 * k1
                + data[last] as u32 * k2
                + data[last - 3] as u32 * k3
                + data[last - 6] as u32 * k4) as u16;
        }
    }

    for i in 0..bytes_per_row {
        let value = temp[2 * bytes_per_row + i] as u32 * k0
            + temp[bytes_per_row + i] as u32 * k1
            + temp[i] as u32 * k2
            + temp[bytes_per_row + i] as u32 * k3
            + temp[2 * bytes_per_row + i] as u32 * k4;
        data[i] = ((value + 32768) >> 16).min(255) as u8;
        let value = temp[bytes_per_row + i] as u32 * k0
            + temp[i] as u32 * k1
            + temp[bytes_per_row + i] as u32 * k2
            + temp[2 * bytes_per_row + i] as u32 * k3
            + temp[3 * bytes_per_row + i] as u32 * k4;
        data[bytes_per_row + i] = ((value + 32768) >> 16).min(255) as u8;
    }
    for y in 2..height - 2 {
        let row = y * bytes_per_row;
        blur::vertical_5x5(temp, data, row, bytes_per_row, [k0, k1, k2, k3, k4])?;
    }
    let penultimate = (height - 2) * bytes_per_row;
    let last = (height - 1) * bytes_per_row;
    for i in 0..bytes_per_row {
        let value = temp[penultimate + i - 2 * bytes_per_row] as u32 * k0
            + temp[penultimate + i - bytes_per_row] as u32 * k1
            + temp[penultimate + i] as u32 * k2
            + temp[last + i] as u32 * k3
            + temp[penultimate + i] as u32 * k4;
        data[penultimate + i] = ((value + 32768) >> 16).min(255) as u8;
        let value = temp[last + i - 2 * bytes_per_row] as u32 * k0
            + temp[last + i - bytes_per_row] as u32 * k1
            + temp[last + i] as u32 * k2
            + temp[last + i - bytes_per_row] as u32 * k3
            + temp[last + i - 2 * bytes_per_row] as u32 * k4;
        data[last + i] = ((value + 32768) >> 16).min(255) as u8;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::layout::normalize_hwc;
    use crate::kernels::point;
    use crate::{Compiler, ExecutionMode, PipelineOutput, PipelineSpec, TransformSpec, Workspace};

    fn pixels(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| ((i * 73 + i / 7 * 19) & 255) as u8)
            .collect()
    }

    fn affine_sample(
        degrees: f32,
        interpolation: Interpolation,
        border_mode: BorderMode,
    ) -> AffineSample {
        AffineSample {
            degrees,
            translate: [0.0, 0.0],
            scale: 1.0,
            shear: [0.0, 0.0],
            interpolation,
            border_mode,
            fill: [3, 5, 7],
        }
    }

    #[test]
    fn resizer_crop_box_is_not_a_materialized_crop_oracle() {
        let height = 9;
        let width = 11;
        let source = pixels(height * width * 3);
        let crop = CropSample {
            top: 2,
            left: 3,
            height: 5,
            width: 6,
        };
        let cropped = random_crop_raw_into(&source, height, width, crop, Vec::new()).unwrap();
        let mut oracle_resizer = fir::Resizer::new();
        let oracle = resize_raw(
            &cropped.data,
            cropped.height,
            cropped.width,
            13,
            17,
            Interpolation::Bilinear,
            true,
            &mut oracle_resizer,
            vec![0; 13 * 17 * 3],
        )
        .unwrap();

        let src =
            FirImageRef::new(width as u32, height as u32, &source, fir::PixelType::U8x3).unwrap();
        let mut dst =
            FirImage::from_vec_u8(17, 13, vec![0; 13 * 17 * 3], fir::PixelType::U8x3).unwrap();
        let options = fir::ResizeOptions::new()
            .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear))
            .crop(
                crop.left as f64,
                crop.top as f64,
                crop.width as f64,
                crop.height as f64,
            );
        fir::Resizer::new()
            .resize(&src, &mut dst, &options)
            .unwrap();

        assert_ne!(dst.buffer(), oracle.data);
    }

    #[test]
    fn bilinear_antialias_policy_changes_downscaling() {
        let height = 19;
        let width = 17;
        let source = pixels(height * width * 3);
        let mut oracle_resizer = fir::Resizer::new();
        let adaptive = resize_raw(
            &source,
            height,
            width,
            7,
            11,
            Interpolation::Bilinear,
            true,
            &mut oracle_resizer,
            vec![0; 7 * 11 * 3],
        )
        .unwrap();
        let fixed = resize_raw(
            &source,
            height,
            width,
            7,
            11,
            Interpolation::Bilinear,
            false,
            &mut oracle_resizer,
            vec![0; 7 * 11 * 3],
        )
        .unwrap();

        assert_ne!(fixed.data, adaptive.data);
    }

    #[test]
    fn rgb_resize_does_not_require_alpha_handling() {
        let height = 19;
        let width = 17;
        let source = pixels(height * width * 3);
        let src =
            FirImageRef::new(width as u32, height as u32, &source, fir::PixelType::U8x3).unwrap();
        let options = fir::ResizeOptions::new()
            .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear));
        let mut with_alpha =
            FirImage::from_vec_u8(11, 7, vec![0; 7 * 11 * 3], fir::PixelType::U8x3).unwrap();
        let mut without_alpha =
            FirImage::from_vec_u8(11, 7, vec![0; 7 * 11 * 3], fir::PixelType::U8x3).unwrap();
        let mut resizer = fir::Resizer::new();
        resizer.resize(&src, &mut with_alpha, &options).unwrap();
        resizer
            .resize(&src, &mut without_alpha, &options.use_alpha(false))
            .unwrap();

        assert_eq!(with_alpha.buffer(), without_alpha.buffer());
    }

    #[test]
    fn arbitrary_rectangular_sizes_preserve_contracts() {
        for (height, width) in [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 3),
            (7, 11),
            (15, 17),
            (16, 31),
            (17, 33),
            (63, 65),
        ] {
            let source = pixels(height * width * 3);
            let mut flipped = ImageU8 {
                data: source.clone(),
                height,
                width,
            };
            point::horizontal_flip(&mut flipped.data, flipped.height, flipped.width);
            point::horizontal_flip(&mut flipped.data, flipped.height, flipped.width);
            assert_eq!(flipped.data, source);

            let mut identity_jitter = ImageU8 {
                data: source.clone(),
                height,
                width,
            };
            color_jitter(
                &mut identity_jitter,
                &ColorJitterSample {
                    brightness: 1.0,
                    contrast: 1.0,
                    saturation: 1.0,
                    hue: 0.0,
                    hue_enabled: false,
                    order: [0, 1, 2, 3],
                },
            );
            assert_eq!(identity_jitter.data, source);

            let kernel = crate::plan::make_gaussian_kernel(5, 1.1).unwrap();
            let mut constant = ImageU8 {
                data: vec![73; source.len()],
                height,
                width,
            };
            let mut temp = Vec::new();
            gaussian_blur_in_place(&mut constant, &kernel, &mut temp).unwrap();
            assert!(constant.data.iter().all(|&value| value == 73));

            let normalized =
                normalize_hwc(&source, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 255.0)
                    .unwrap();
            assert_eq!(normalized.len(), source.len());
            assert!(normalized.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn reflect101_matches_expected_period() {
        let values: Vec<_> = (-8..=12).map(|index| reflect101_index(index, 5)).collect();
        assert_eq!(
            values,
            vec![0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4]
        );
        assert_eq!(reflect101_index(-100, 1), 0);
    }

    #[test]
    fn wide_gaussian_blur_preserves_a_constant_image() {
        let kernel = crate::plan::make_gaussian_kernel(101, 1_000_000.0).unwrap();
        let mut image = ImageU8 {
            data: vec![73; 3 * 4 * 3],
            height: 3,
            width: 4,
        };
        gaussian_blur_in_place(&mut image, &kernel, &mut Vec::new()).unwrap();
        assert!(image.data.iter().all(|&value| value == 73));
    }

    #[test]
    fn extreme_color_jitter_brightness_saturates_without_wrapping() {
        let mut image = ImageU8 {
            data: vec![255, 2, 3, 0, 0, 0],
            height: 1,
            width: 2,
        };
        color_jitter(
            &mut image,
            &ColorJitterSample {
                brightness: 1_000.0,
                contrast: 1.0,
                saturation: 1.0,
                hue: 0.0,
                hue_enabled: false,
                order: [0, 1, 2, 3],
            },
        );
        assert_eq!(image.data, [255, 255, 255, 0, 0, 0]);
    }

    #[test]
    fn q14_color_dispatch_matches_the_wide_oracle_at_vector_boundaries() {
        let matrix = [[1.25, -0.25, 0.0], [0.0, 0.75, 0.25], [0.5, 0.0, 0.5]];
        let safe_matrix = matrix.map(|row| row.map(f64::from));
        for pixel_count in 0..=65 {
            let source: Vec<_> = (0..pixel_count * 3)
                .map(|index| (index * 73) as u8)
                .collect();
            let mut expected = source.clone();
            apply_color_matrix_f64(&mut expected, safe_matrix, 1.5);
            let mut actual = ImageU8 {
                data: source,
                height: 1,
                width: pixel_count,
            };
            apply_color_matrix(&mut actual, matrix, 1.5);
            assert_eq!(actual.data, expected, "pixel_count={pixel_count}");
        }
    }

    #[test]
    fn extreme_color_jitter_factors_match_the_wide_oracle() {
        let source = vec![255, 2, 3, 17, 73, 251, 19, 19, 19];
        for factors in [
            [1_000.0, 1.0, 1.0],
            [1.0, 1_000.0, 1.0],
            [1.0, 1.0, 1_000.0],
            [f32::MAX, f32::MAX, f32::MAX],
        ] {
            let sample = ColorJitterSample {
                brightness: factors[0],
                contrast: factors[1],
                saturation: factors[2],
                hue: 0.0,
                hue_enabled: false,
                order: [0, 1, 2, 3],
            };
            let sums: [u64; 3] = [
                source
                    .iter()
                    .step_by(3)
                    .map(|&value| u64::from(value))
                    .sum(),
                source
                    .iter()
                    .skip(1)
                    .step_by(3)
                    .map(|&value| u64::from(value))
                    .sum(),
                source
                    .iter()
                    .skip(2)
                    .step_by(3)
                    .map(|&value| u64::from(value))
                    .sum(),
            ];
            let source_mean = sums.map(|sum| sum as f64 / 3.0);
            let (matrix, offset) = compose_color_matrix_f64(&sample, source_mean);
            let mut expected = source.clone();
            apply_color_matrix_f64(&mut expected, matrix, offset[0]);

            let mut actual = ImageU8 {
                data: source.clone(),
                height: 1,
                width: 3,
            };
            color_jitter(&mut actual, &sample);
            assert_eq!(actual.data, expected, "factors={factors:?}");
        }
    }

    #[test]
    fn pad_constant_places_input_and_overwrites_destination() {
        let source = pixels(2 * 3 * 3);
        let sample = crate::plan::PadSample {
            top: 1,
            left: 2,
            height: 5,
            width: 8,
            border_mode: BorderMode::Constant,
            fill: [3, 5, 7],
        };
        let clean = pad_raw(&source, 2, 3, sample, vec![0; 5 * 8 * 3]).unwrap();
        let dirty = pad_raw(&source, 2, 3, sample, vec![0xa5; 5 * 8 * 3]).unwrap();
        assert_eq!(clean.data, dirty.data);
        for y in 0..5 {
            for x in 0..8 {
                let offset = (y * 8 + x) * 3;
                if (1..3).contains(&y) && (2..5).contains(&x) {
                    let source_offset = ((y - 1) * 3 + x - 2) * 3;
                    assert_eq!(
                        &clean.data[offset..offset + 3],
                        &source[source_offset..source_offset + 3]
                    );
                } else {
                    assert_eq!(&clean.data[offset..offset + 3], &[3, 5, 7]);
                }
            }
        }
    }

    #[test]
    fn pad_reflect101_matches_known_mapping() {
        let source: Vec<_> = (0..6).flat_map(|value| [value; 3]).collect();
        let sample = crate::plan::PadSample {
            top: 1,
            left: 1,
            height: 4,
            width: 5,
            border_mode: BorderMode::Reflect101,
            fill: [0; 3],
        };
        let output = pad_raw(&source, 2, 3, sample, Vec::new()).unwrap();
        let expected = [
            [4, 3, 4, 5, 4],
            [1, 0, 1, 2, 1],
            [4, 3, 4, 5, 4],
            [1, 0, 1, 2, 1],
        ];
        for (pixel, expected) in output
            .data
            .chunks_exact(3)
            .zip(expected.into_iter().flatten())
        {
            assert_eq!(pixel, [expected; 3]);
        }
    }

    #[test]
    fn coarse_dropout_fills_rectangles_and_allows_overlap() {
        let mut image = ImageU8 {
            data: pixels(4 * 6 * 3),
            height: 4,
            width: 6,
        };
        let original = image.data.clone();
        let holes = [
            crate::plan::DropoutHole {
                top: 1,
                left: 2,
                height: 2,
                width: 3,
            },
            crate::plan::DropoutHole {
                top: 2,
                left: 4,
                height: 2,
                width: 2,
            },
        ];
        coarse_dropout(&mut image, &holes, [3, 5, 7]).unwrap();
        for y in 0..4 {
            for x in 0..6 {
                let offset = (y * 6 + x) * 3;
                let covered = ((1..3).contains(&y) && (2..5).contains(&x))
                    || ((2..4).contains(&y) && (4..6).contains(&x));
                let expected = if covered {
                    &[3, 5, 7][..]
                } else {
                    &original[offset..offset + 3]
                };
                assert_eq!(&image.data[offset..offset + 3], expected);
            }
        }
    }

    #[test]
    fn affine_border_modes_cover_arbitrary_dimensions() {
        for (height, width) in [(1, 1), (1, 7), (7, 1), (7, 11)] {
            let source = vec![73; height * width * 3];
            let reflect = rotate_raw(
                &source,
                height,
                width,
                affine_sample(37.0, Interpolation::Bilinear, BorderMode::Reflect101),
                vec![0; source.len()],
            )
            .unwrap();
            assert!(reflect.data.iter().all(|&value| value == 73));

            let nearest = rotate_raw(
                &source,
                height,
                width,
                affine_sample(37.0, Interpolation::Nearest, BorderMode::Reflect101),
                vec![0; source.len()],
            )
            .unwrap();
            assert!(nearest.data.iter().all(|&value| value == 73));
        }

        let source = vec![73; 7 * 11 * 3];
        let constant = rotate_raw(
            &source,
            7,
            11,
            affine_sample(37.0, Interpolation::Bilinear, BorderMode::Constant),
            vec![0; source.len()],
        )
        .unwrap();
        assert!(constant.data.iter().any(|&value| value != 73));
    }

    #[test]
    fn affine_q16_boundary_matches_portable_oracle() {
        for (height, width) in [(1, 32_769), (1, 32_770), (32_769, 1), (32_770, 1)] {
            let source = pixels(height * width * 3);
            let identity = affine_sample(0.0, Interpolation::Bilinear, BorderMode::Constant);
            let output =
                rotate_raw(&source, height, width, identity, vec![0xa5; source.len()]).unwrap();
            assert_eq!(output.data, source, "identity {height}x{width}");

            let sample = affine_sample(0.25, Interpolation::Bilinear, BorderMode::Constant);
            let matrix = inverse_affine_matrix(width, height, sample);
            let mut expected = vec![0; source.len()];
            rotate_bilinear_border(&source, height, width, sample, matrix, &mut expected, false);
            let actual = rotate_raw(&source, height, width, sample, vec![0; source.len()]).unwrap();
            assert_eq!(actual.data, expected, "non-identity {height}x{width}");
        }
    }

    #[test]
    fn affine_rejects_dimensions_beyond_exact_f32_coordinates() {
        let error = match rotate_raw(
            &[],
            1,
            MAX_AFFINE_DIMENSION + 1,
            affine_sample(0.0, Interpolation::Bilinear, BorderMode::Constant),
            Vec::new(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("oversized Affine input was accepted"),
        };
        assert!(matches!(error, CoreError::Invalid(_)));
        assert!(error.to_string().contains("16777216"));
    }

    #[test]
    fn affine_overwrites_every_destination_byte() {
        for (height, width) in [(1, 1), (3, 5), (17, 19), (31, 33)] {
            let source = pixels(height * width * 3);
            for interpolation in [Interpolation::Nearest, Interpolation::Bilinear] {
                for border_mode in [BorderMode::Constant, BorderMode::Reflect101] {
                    let sample = affine_sample(37.0, interpolation, border_mode);
                    let clean =
                        rotate_raw(&source, height, width, sample, vec![0; source.len()]).unwrap();
                    let dirty =
                        rotate_raw(&source, height, width, sample, vec![0xa5; source.len()])
                            .unwrap();
                    assert_eq!(clean.data, dirty.data);
                }
            }
        }
    }

    #[test]
    fn affine_translation_uses_inverse_mapping() {
        let source = pixels(3 * 3);
        let mut sample = affine_sample(0.0, Interpolation::Nearest, BorderMode::Constant);
        sample.translate = [1.0, 0.0];
        let output = rotate_raw(&source, 1, 3, sample, vec![0; source.len()]).unwrap();
        assert_eq!(&output.data[..3], &[3, 5, 7]);
        assert_eq!(&output.data[3..], &source[..6]);
    }

    #[test]
    fn hue_rotation_preserves_value_and_saturation() {
        let mut image = ImageU8 {
            data: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 73, 73, 73],
            height: 1,
            width: 4,
        };
        adjust_hue(&mut image, 1.0 / 3.0);
        assert_eq!(
            image.data,
            vec![0, 255, 0, 0, 0, 255, 255, 0, 0, 73, 73, 73]
        );
    }

    #[test]
    fn compiled_plan_is_shareable_with_per_run_workspaces() {
        use std::sync::Arc;

        let pipeline = Arc::new(
            Compiler::new(ExecutionMode::Compiled)
                .compile(PipelineSpec::new(vec![
                    TransformSpec::RandomCrop {
                        height: 29,
                        width: 35,
                        p: 1.0,
                    },
                    TransformSpec::Resize {
                        height: 17,
                        width: 19,
                        interpolation: Interpolation::Bilinear,
                        antialias: false,
                        p: 1.0,
                    },
                    TransformSpec::HorizontalFlip { p: 0.5 },
                ]))
                .unwrap(),
        );
        let source = Arc::new(pixels(31 * 37 * 3));
        let handles: Vec<_> = (0..16)
            .map(|key| {
                let pipeline = Arc::clone(&pipeline);
                let source = Arc::clone(&source);
                std::thread::spawn(move || {
                    let mut workspace = Workspace::default();
                    let output = pipeline
                        .apply(source.as_slice(), 31, 37, 137, key, &mut workspace)
                        .unwrap();
                    (key, output)
                })
            })
            .collect();

        for handle in handles {
            let (key, concurrent) = handle.join().unwrap();
            let mut workspace = Workspace::default();
            let sequential = pipeline
                .apply(source.as_slice(), 31, 37, 137, key, &mut workspace)
                .unwrap();
            match (concurrent, sequential) {
                (
                    PipelineOutput::U8Hwc {
                        data: concurrent, ..
                    },
                    PipelineOutput::U8Hwc {
                        data: sequential, ..
                    },
                ) => assert_eq!(concurrent, sequential, "key={key}"),
                _ => panic!("unexpected output type"),
            }
        }
    }
}
