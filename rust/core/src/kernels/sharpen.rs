use crate::operations::reflect101_index;
use crate::plan::SharpenSample;

pub(crate) fn apply(
    data: &[u8],
    height: usize,
    width: usize,
    sample: SharpenSample,
    output: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    if simd_coefficients_are_finite(sample)
        && std::arch::is_x86_feature_detected!("avx2")
        && width >= 3
    {
        // SAFETY: runtime detection guards AVX2; the caller validated equal RGB buffers.
        unsafe { apply_avx2(data, height, width, sample, output) };
        return;
    }
    apply_scalar(data, height, width, sample, output);
}

fn apply_scalar(
    data: &[u8],
    height: usize,
    width: usize,
    sample: SharpenSample,
    output: &mut [u8],
) {
    let center_scale = 1.0 + 4.0 * sample.lightness;
    for y in 0..height {
        let top = reflect101_index(y as isize - 1, height);
        let bottom = reflect101_index(y as isize + 1, height);
        for x in 0..width {
            let left = reflect101_index(x as isize - 1, width);
            let right = reflect101_index(x as isize + 1, width);
            for channel in 0..3 {
                let destination = (y * width + x) * 3 + channel;
                output[destination] = pixel(
                    data[destination],
                    data[(top * width + x) * 3 + channel],
                    data[(bottom * width + x) * 3 + channel],
                    data[(y * width + left) * 3 + channel],
                    data[(y * width + right) * 3 + channel],
                    center_scale,
                    sample,
                );
            }
        }
    }
}

#[inline]
fn pixel(
    center: u8,
    top: u8,
    bottom: u8,
    left: u8,
    right: u8,
    center_scale: f32,
    sample: SharpenSample,
) -> u8 {
    let center = f32::from(center);
    let mut neighbors = f32::from(top);
    neighbors += f32::from(bottom);
    neighbors += f32::from(left);
    neighbors += f32::from(right);
    let sharpened = center * center_scale - neighbors * sample.lightness;
    (center + sample.alpha * (sharpened - center))
        .round()
        .clamp(0.0, 255.0) as u8
}

#[cfg(target_arch = "x86_64")]
fn simd_coefficients_are_finite(sample: SharpenSample) -> bool {
    let center_scale = 1.0 + 4.0 * sample.lightness;
    let maximum = f64::from(f32::MAX);
    center_scale.is_finite()
        && f64::from(center_scale).abs() * 255.0 <= maximum
        && f64::from(sample.lightness).abs() * 1020.0 <= maximum
        && f64::from(sample.alpha).abs()
            * (f64::from(center_scale).abs() * 255.0
                + f64::from(sample.lightness).abs() * 1020.0
                + 255.0)
            + 255.0
            <= maximum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_avx2(
    data: &[u8],
    height: usize,
    width: usize,
    sample: SharpenSample,
    output: &mut [u8],
) {
    // SAFETY: the caller guarantees AVX2, finite intermediates, width >= 3, and matching buffers.
    unsafe {
        use std::arch::x86_64::*;

        let row_bytes = width * 3;
        let center_scale_value = 1.0 + 4.0 * sample.lightness;
        let center_scale = _mm256_set1_ps(center_scale_value);
        let lightness = _mm256_set1_ps(sample.lightness);
        let alpha = _mm256_set1_ps(sample.alpha);
        let zero = _mm256_setzero_ps();
        let maximum = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);

        for y in 0..height {
            let top_y = reflect101_index(y as isize - 1, height);
            let bottom_y = reflect101_index(y as isize + 1, height);
            for channel in 0..3 {
                let center = y * row_bytes + channel;
                output[center] = pixel(
                    data[center],
                    data[top_y * row_bytes + channel],
                    data[bottom_y * row_bytes + channel],
                    data[y * row_bytes + 3 + channel],
                    data[y * row_bytes + 3 + channel],
                    center_scale_value,
                    sample,
                );
                let last = y * row_bytes + (width - 1) * 3 + channel;
                output[last] = pixel(
                    data[last],
                    data[top_y * row_bytes + (width - 1) * 3 + channel],
                    data[bottom_y * row_bytes + (width - 1) * 3 + channel],
                    data[y * row_bytes + (width - 2) * 3 + channel],
                    data[y * row_bytes + (width - 2) * 3 + channel],
                    center_scale_value,
                    sample,
                );
            }

            let start = y * row_bytes + 3;
            let end = y * row_bytes + (width - 1) * 3;
            let top = top_y * row_bytes + 3;
            let bottom = bottom_y * row_bytes + 3;
            let mut offset = 0usize;
            while start + offset + 8 <= end {
                let load = |pointer: *const u8| {
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(pointer.cast())))
                };
                let center = load(data.as_ptr().add(start + offset));
                let top = load(data.as_ptr().add(top + offset));
                let bottom = load(data.as_ptr().add(bottom + offset));
                let left = load(data.as_ptr().add(start + offset - 3));
                let right = load(data.as_ptr().add(start + offset + 3));
                let mut neighbors = _mm256_add_ps(top, bottom);
                neighbors = _mm256_add_ps(neighbors, left);
                neighbors = _mm256_add_ps(neighbors, right);
                let sharpened = _mm256_sub_ps(
                    _mm256_mul_ps(center, center_scale),
                    _mm256_mul_ps(neighbors, lightness),
                );
                let blended = _mm256_add_ps(
                    center,
                    _mm256_mul_ps(alpha, _mm256_sub_ps(sharpened, center)),
                );
                let clamped = _mm256_min_ps(_mm256_max_ps(blended, zero), maximum);
                let rounded = _mm256_floor_ps(_mm256_add_ps(clamped, half));
                let integers = _mm256_cvttps_epi32(rounded);
                let words = _mm_packus_epi32(
                    _mm256_castsi256_si128(integers),
                    _mm256_extracti128_si256(integers, 1),
                );
                let bytes = _mm_packus_epi16(words, words);
                _mm_storel_epi64(output.as_mut_ptr().add(start + offset).cast(), bytes);
                offset += 8;
            }
            for index in start + offset..end {
                output[index] = pixel(
                    data[index],
                    data[top_y * row_bytes + index - y * row_bytes],
                    data[bottom_y * row_bytes + index - y * row_bytes],
                    data[index - 3],
                    data[index + 3],
                    center_scale_value,
                    sample,
                );
            }
        }
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
    fn dispatch_matches_scalar_for_arbitrary_rectangles_and_coefficients() {
        for (height, width) in [
            (1, 1),
            (1, 9),
            (9, 1),
            (2, 3),
            (3, 5),
            (7, 11),
            (9, 17),
            (17, 33),
        ] {
            let source = pixels(height * width * 3);
            for (alpha, lightness) in [(0.0, 0.0), (0.25, 0.7), (0.5, 1.0), (1.0, 3.5)] {
                let sample = SharpenSample { alpha, lightness };
                let mut expected = vec![0xa5; source.len()];
                let mut actual = vec![0x5a; source.len()];
                apply_scalar(&source, height, width, sample, &mut expected);
                apply(&source, height, width, sample, &mut actual);
                assert_eq!(actual, expected, "{height}x{width} {sample:?}");
            }
        }
    }

    #[test]
    fn extreme_coefficients_use_the_scalar_contract() {
        let source = pixels(7 * 11 * 3);
        let sample = SharpenSample {
            alpha: 1.0,
            lightness: f32::MAX,
        };
        let mut expected = vec![0; source.len()];
        let mut actual = vec![0; source.len()];
        apply_scalar(&source, 7, 11, sample, &mut expected);
        apply(&source, 7, 11, sample, &mut actual);
        assert_eq!(actual, expected);
    }
}
