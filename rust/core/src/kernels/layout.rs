use std::mem::MaybeUninit;

use crate::{CoreError, CoreResult};

pub(crate) fn hwc_to_chw<T: Copy + Default>(
    data: &[T],
    height: usize,
    width: usize,
) -> CoreResult<Vec<T>> {
    let plane = height * width;
    let mut output = Vec::new();
    output
        .try_reserve_exact(data.len())
        .map_err(|_| CoreError::Runtime("output allocation failed".into()))?;
    output.resize(data.len(), T::default());
    for pixel in 0..plane {
        let source = pixel * 3;
        output[pixel] = data[source];
        output[plane + pixel] = data[source + 1];
        output[2 * plane + pixel] = data[source + 2];
    }
    Ok(output)
}

pub(crate) fn normalize_hwc(
    data: &[u8],
    mean: [f32; 3],
    std: [f32; 3],
    max_pixel_value: f32,
) -> CoreResult<Vec<f32>> {
    let scale = [
        1.0 / (max_pixel_value * std[0]),
        1.0 / (max_pixel_value * std[1]),
        1.0 / (max_pixel_value * std[2]),
    ];
    let bias = [-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]];
    let mut output = Vec::new();
    output
        .try_reserve_exact(data.len())
        .map_err(|_| CoreError::Runtime("output allocation failed".into()))?;
    output.resize(data.len(), MaybeUninit::uninit());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection guards AVX2 and the output has data.len() elements.
        unsafe { normalize_hwc_avx2(data, &mut output, scale, bias) };
        // SAFETY: the AVX2 implementation and its scalar tail initialize every element.
        return Ok(unsafe { assume_init_f32(output) });
    }

    normalize_hwc_scalar(data, &mut output, scale, bias, 0);
    // SAFETY: the scalar implementation initializes every element.
    Ok(unsafe { assume_init_f32(output) })
}

pub(crate) fn normalize_hwc_to_chw(
    data: &[u8],
    height: usize,
    width: usize,
    mean: [f32; 3],
    std: [f32; 3],
    max_pixel_value: f32,
) -> CoreResult<Vec<f32>> {
    let plane = height
        .checked_mul(width)
        .ok_or_else(|| CoreError::Invalid("image dimensions overflow".into()))?;
    let len = plane
        .checked_mul(3)
        .ok_or_else(|| CoreError::Invalid("image dimensions overflow".into()))?;
    if data.len() != len {
        return Err(CoreError::Invalid("invalid RGB buffer".into()));
    }
    let scale = [
        1.0 / (max_pixel_value * std[0]),
        1.0 / (max_pixel_value * std[1]),
        1.0 / (max_pixel_value * std[2]),
    ];
    let bias = [-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]];
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| CoreError::Runtime("output allocation failed".into()))?;
    output.resize(len, MaybeUninit::uninit());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection guards AVX2 and the validated buffers contain three planes.
        unsafe { normalize_hwc_to_chw_avx2(data, &mut output, plane, scale, bias) };
        // SAFETY: the AVX2 implementation and its scalar tail initialize every element.
        return Ok(unsafe { assume_init_f32(output) });
    }

    normalize_hwc_to_chw_scalar(data, &mut output, plane, scale, bias, 0);
    // SAFETY: the scalar implementation initializes every element.
    Ok(unsafe { assume_init_f32(output) })
}

fn normalize_hwc_scalar(
    data: &[u8],
    output: &mut [MaybeUninit<f32>],
    scale: [f32; 3],
    bias: [f32; 3],
    start: usize,
) {
    for index in start..data.len() {
        let channel = index % 3;
        output[index].write(data[index] as f32 * scale[channel] + bias[channel]);
    }
}

fn normalize_hwc_to_chw_scalar(
    data: &[u8],
    output: &mut [MaybeUninit<f32>],
    plane: usize,
    scale: [f32; 3],
    bias: [f32; 3],
    start_pixel: usize,
) {
    for pixel in start_pixel..plane {
        let source = pixel * 3;
        output[pixel].write(data[source] as f32 * scale[0] + bias[0]);
        output[plane + pixel].write(data[source + 1] as f32 * scale[1] + bias[1]);
        output[2 * plane + pixel].write(data[source + 2] as f32 * scale[2] + bias[2]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_hwc_avx2(
    data: &[u8],
    output: &mut [MaybeUninit<f32>],
    scale: [f32; 3],
    bias: [f32; 3],
) {
    // SAFETY: the caller guarantees AVX2 and output has at least data.len() elements.
    unsafe {
        use std::arch::x86_64::*;

        let scales = [
            _mm256_setr_ps(
                scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1],
            ),
            _mm256_setr_ps(
                scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2],
            ),
            _mm256_setr_ps(
                scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0],
            ),
        ];
        let biases = [
            _mm256_setr_ps(
                bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1],
            ),
            _mm256_setr_ps(
                bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2],
            ),
            _mm256_setr_ps(
                bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0],
            ),
        ];
        let mut index = 0usize;
        while index + 8 <= data.len() {
            let bytes = _mm_loadl_epi64(data.as_ptr().add(index) as *const __m128i);
            let values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
            let phase = index % 3;
            _mm256_storeu_ps(
                output.as_mut_ptr().add(index) as *mut f32,
                _mm256_add_ps(_mm256_mul_ps(values, scales[phase]), biases[phase]),
            );
            index += 8;
        }
        normalize_hwc_scalar(data, output, scale, bias, index);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_hwc_to_chw_avx2(
    data: &[u8],
    output: &mut [MaybeUninit<f32>],
    plane: usize,
    scale: [f32; 3],
    bias: [f32; 3],
) {
    // SAFETY: the caller guarantees AVX2 and validated input and output lengths.
    unsafe {
        use std::arch::x86_64::*;

        let scales = scale.map(|value| _mm256_set1_ps(value));
        let biases = bias.map(|value| _mm256_set1_ps(value));
        let a_masks = [
            _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
        ];
        let b_masks = [
            _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, -1, -1, -1, -1, -1, -1, -1, -1),
            _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1),
            _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1),
        ];
        let mut pixel = 0usize;
        while pixel + 8 <= plane {
            let source = data.as_ptr().add(pixel * 3);
            let a = _mm_loadu_si128(source as *const __m128i);
            let b = _mm_loadl_epi64(source.add(16) as *const __m128i);
            for channel in 0..3 {
                let bytes = _mm_or_si128(
                    _mm_shuffle_epi8(a, a_masks[channel]),
                    _mm_shuffle_epi8(b, b_masks[channel]),
                );
                let values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
                let normalized =
                    _mm256_add_ps(_mm256_mul_ps(values, scales[channel]), biases[channel]);
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(channel * plane + pixel) as *mut f32,
                    normalized,
                );
            }
            pixel += 8;
        }
        normalize_hwc_to_chw_scalar(data, output, plane, scale, bias, pixel);
    }
}

unsafe fn assume_init_f32(mut values: Vec<MaybeUninit<f32>>) -> Vec<f32> {
    // SAFETY: the caller guarantees every MaybeUninit element contains an f32.
    unsafe {
        let ptr = values.as_mut_ptr().cast::<f32>();
        let len = values.len();
        let capacity = values.capacity();
        std::mem::forget(values);
        Vec::from_raw_parts(ptr, len, capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixels(len: usize) -> Vec<u8> {
        (0..len)
            .map(|index| ((index * 73 + index / 7 * 19) & 255) as u8)
            .collect()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_normalize_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let scale = [0.017, 0.013, 0.021];
        let bias = [-2.1, -1.8, -1.4];
        for pixel_count in 1..=128 {
            let source = pixels(pixel_count * 3);
            let mut scalar = vec![MaybeUninit::uninit(); source.len()];
            let mut simd = vec![MaybeUninit::uninit(); source.len()];
            normalize_hwc_scalar(&source, &mut scalar, scale, bias, 0);
            // SAFETY: AVX2 support is checked above.
            unsafe { normalize_hwc_avx2(&source, &mut simd, scale, bias) };
            // SAFETY: both kernels initialized every element.
            let scalar = unsafe { assume_init_f32(scalar) };
            // SAFETY: both kernels initialized every element.
            let simd = unsafe { assume_init_f32(simd) };
            assert_eq!(simd, scalar, "pixel_count={pixel_count}");
        }
    }

    #[test]
    fn configurable_scale_matches_the_formula() {
        let source = pixels(17 * 19 * 3);
        let mean = [0.25, 0.5, 0.75];
        let std = [0.5, 0.25, 0.125];
        let max_pixel_value = 127.5;
        let actual = normalize_hwc(&source, mean, std, max_pixel_value).unwrap();
        let mut expected: Vec<f32> = source
            .iter()
            .map(|&value| value as f32 / max_pixel_value)
            .collect();
        for pixel in expected.chunks_exact_mut(3) {
            for channel in 0..3 {
                pixel[channel] = (pixel[channel] - mean[channel]) / std[channel];
            }
        }

        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn direct_normalize_chw_matches_staged_layout_for_arbitrary_rectangles() {
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        for (height, width) in [(1, 1), (1, 7), (7, 1), (3, 5), (17, 33), (31, 32)] {
            let source = pixels(height * width * 3);
            let staged = hwc_to_chw(
                &normalize_hwc(&source, mean, std, 255.0).unwrap(),
                height,
                width,
            )
            .unwrap();
            let direct = normalize_hwc_to_chw(&source, height, width, mean, std, 255.0).unwrap();
            assert_eq!(direct, staged, "{height}x{width}");
        }
    }

    #[test]
    fn chw_conversion_preserves_arbitrary_rectangles() {
        for (height, width) in [(1, 1), (1, 7), (7, 1), (3, 5), (17, 33)] {
            let source = pixels(height * width * 3);
            let output = hwc_to_chw(&source, height, width).unwrap();
            let plane = height * width;
            for pixel in 0..plane {
                assert_eq!(output[pixel], source[pixel * 3]);
                assert_eq!(output[plane + pixel], source[pixel * 3 + 1]);
                assert_eq!(output[2 * plane + pixel], source[pixel * 3 + 2]);
            }
        }
    }
}
