pub(crate) fn apply_independent(channels: &mut [u8], normals: &[f32], mean: f32, std: f32) {
    #[cfg(target_arch = "x86_64")]
    if mean.abs() <= 1.0e6 && std.abs() <= 1.0e6 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection guards AVX2 and the slices have equal lengths.
        unsafe { apply_independent_avx2(channels, normals, mean, std) };
        return;
    }
    apply_independent_scalar(channels, normals, mean, std, 0);
}

fn apply_independent_scalar(
    channels: &mut [u8],
    normals: &[f32],
    mean: f32,
    std: f32,
    start: usize,
) {
    for index in start..channels.len() {
        channels[index] = (f32::from(channels[index]) + mean + normals[index] * std)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_independent_avx2(channels: &mut [u8], normals: &[f32], mean: f32, std: f32) {
    // SAFETY: the caller guarantees AVX2 and equal-length input slices.
    unsafe {
        use std::arch::x86_64::*;

        let mean_value = mean;
        let std_value = std;
        let mean = _mm256_set1_ps(mean_value);
        let std = _mm256_set1_ps(std_value);
        let zero = _mm256_setzero_ps();
        let maximum = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);
        let mut index = 0usize;
        while index + 8 <= channels.len() {
            let bytes = _mm_loadl_epi64(channels.as_ptr().add(index).cast());
            let values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
            let noise = _mm256_loadu_ps(normals.as_ptr().add(index));
            let noisy = _mm256_add_ps(_mm256_add_ps(values, mean), _mm256_mul_ps(noise, std));
            let clamped = _mm256_min_ps(_mm256_max_ps(noisy, zero), maximum);
            let rounded = _mm256_floor_ps(_mm256_add_ps(clamped, half));
            let integers = _mm256_cvttps_epi32(rounded);
            let words = _mm_packus_epi32(
                _mm256_castsi256_si128(integers),
                _mm256_extracti128_si256(integers, 1),
            );
            let packed = _mm_packus_epi16(words, words);
            _mm_storel_epi64(channels.as_mut_ptr().add(index).cast(), packed);
            index += 8;
        }
        apply_independent_scalar(channels, normals, mean_value, std_value, index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_dispatch_matches_scalar_at_vector_boundaries() {
        for len in 1..=65 {
            let mut expected: Vec<_> = (0..len).map(|index| (index * 73) as u8).collect();
            let mut actual = expected.clone();
            let normals: Vec<_> = (0..len)
                .map(|index| (index as f32 * 0.37).sin() * 2.5)
                .collect();
            apply_independent_scalar(&mut expected, &normals, 1.25, 7.5, 0);
            apply_independent(&mut actual, &normals, 1.25, 7.5);
            assert_eq!(actual, expected, "len={len}");
        }
    }
}
