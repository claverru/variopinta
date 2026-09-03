pub(crate) fn apply_q14(data: &mut [u8], matrix: [[i32; 3]; 3], bias: i32) {
    debug_assert!(q14_accumulator_fits(matrix, bias));
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected at runtime and the implementation handles the scalar tail.
        unsafe { apply_q14_avx2(data, matrix, bias) };
        return;
    }
    apply_q14_scalar(data, matrix, bias);
}

pub(crate) fn q14_accumulator_fits(matrix: [[i32; 3]; 3], bias: i32) -> bool {
    const ROUND: i64 = 8192;
    matrix.iter().all(|row| {
        let products = row
            .iter()
            .map(|&coefficient| i64::from(coefficient).abs() * 255)
            .sum::<i64>();
        products + i64::from(bias).abs() + ROUND <= i64::from(i32::MAX)
    })
}

#[inline]
fn apply_q14_scalar(data: &mut [u8], matrix: [[i32; 3]; 3], bias: i32) {
    for pixel in data.chunks_exact_mut(3) {
        apply_q14_pixel(pixel, matrix, bias);
    }
}

#[inline]
pub(crate) fn apply_q14_pixel(pixel: &mut [u8], matrix: [[i32; 3]; 3], bias: i32) {
    const ROUND: i32 = 8192;
    let [r, g, b] = [pixel[0] as i32, pixel[1] as i32, pixel[2] as i32];
    for channel in 0..3 {
        let value = matrix[channel][0] * r + matrix[channel][1] * g + matrix[channel][2] * b + bias;
        pixel[channel] = ((value + ROUND) >> 14).clamp(0, 255) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_q14_avx2(data: &mut [u8], matrix: [[i32; 3]; 3], bias: i32) {
    // SAFETY: the caller guarantees AVX2. Loads and stores cover complete 48-byte blocks;
    // the remainder is passed to the safe scalar oracle.
    unsafe {
        use std::arch::x86_64::*;

        let m_a_r = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let m_b_r = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
        let m_c_r = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
        let m_a_g = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let m_b_g = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
        let m_c_g = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
        let m_a_b = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        let m_b_b = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
        let m_c_b = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);

        let coeff: [[__m128i; 3]; 3] = matrix.map(|row| row.map(|value| _mm_set1_epi32(value)));
        let bias_v = _mm_set1_epi32(bias + 8192);
        let zero = _mm_setzero_si128();
        let max = _mm_set1_epi32(255);
        let ptr = data.as_mut_ptr();
        let mut offset = 0usize;
        while offset + 48 <= data.len() {
            let a = _mm_loadu_si128(ptr.add(offset).cast());
            let bb = _mm_loadu_si128(ptr.add(offset + 16).cast());
            let c = _mm_loadu_si128(ptr.add(offset + 32).cast());
            let mut r = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, m_a_r), _mm_shuffle_epi8(bb, m_b_r)),
                _mm_shuffle_epi8(c, m_c_r),
            );
            let mut g = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, m_a_g), _mm_shuffle_epi8(bb, m_b_g)),
                _mm_shuffle_epi8(c, m_c_g),
            );
            let mut b = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, m_a_b), _mm_shuffle_epi8(bb, m_b_b)),
                _mm_shuffle_epi8(c, m_c_b),
            );
            let mut out_r = [zero; 4];
            let mut out_g = [zero; 4];
            let mut out_b = [zero; 4];
            for group in 0..4 {
                let rv = _mm_cvtepu8_epi32(r);
                let gv = _mm_cvtepu8_epi32(g);
                let bv = _mm_cvtepu8_epi32(b);
                let compute = |row: usize| {
                    let value = _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_add_epi32(
                                _mm_mullo_epi32(rv, coeff[row][0]),
                                _mm_mullo_epi32(gv, coeff[row][1]),
                            ),
                            _mm_mullo_epi32(bv, coeff[row][2]),
                        ),
                        bias_v,
                    );
                    _mm_min_epi32(max, _mm_max_epi32(zero, _mm_srai_epi32::<14>(value)))
                };
                out_r[group] = compute(0);
                out_g[group] = compute(1);
                out_b[group] = compute(2);
                r = _mm_srli_si128::<4>(r);
                g = _mm_srli_si128::<4>(g);
                b = _mm_srli_si128::<4>(b);
            }
            let pack = |values: [__m128i; 4]| {
                _mm_packus_epi16(
                    _mm_packs_epi32(values[0], values[1]),
                    _mm_packs_epi32(values[2], values[3]),
                )
            };
            let r = pack(out_r);
            let g = pack(out_g);
            let b = pack(out_b);
            let scatter = |rv: __m128i, gv: __m128i, bv: __m128i, group: usize| {
                let (mr, mg, mb) = match group {
                    0 => (
                        _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5),
                        _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1),
                        _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1),
                    ),
                    1 => (
                        _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1),
                        _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10),
                        _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1),
                    ),
                    _ => (
                        _mm_setr_epi8(
                            -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
                        ),
                        _mm_setr_epi8(
                            -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
                        ),
                        _mm_setr_epi8(
                            10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
                        ),
                    ),
                };
                _mm_or_si128(
                    _mm_or_si128(_mm_shuffle_epi8(rv, mr), _mm_shuffle_epi8(gv, mg)),
                    _mm_shuffle_epi8(bv, mb),
                )
            };
            _mm_storeu_si128(ptr.add(offset).cast(), scatter(r, g, b, 0));
            _mm_storeu_si128(ptr.add(offset + 16).cast(), scatter(r, g, b, 1));
            _mm_storeu_si128(ptr.add(offset + 32).cast(), scatter(r, g, b, 2));
            offset += 48;
        }
        apply_q14_scalar(&mut data[offset..], matrix, bias);
    }
}

pub(crate) fn adjust_hue(data: &mut [u8], factor: f32) {
    if factor == 0.0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection guards AVX2 and the kernel retains a scalar tail.
        unsafe { adjust_hue_avx2(data, factor) };
        return;
    }
    for pixel in data.chunks_exact_mut(3) {
        adjust_hue_pixel(pixel, factor);
    }
}

#[inline]
fn adjust_hue_pixel(pixel: &mut [u8], factor: f32) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn adjust_hue_avx2(data: &mut [u8], factor_value: f32) {
    // SAFETY: the caller guarantees AVX2. Loads cover complete 24-byte blocks and the
    // remaining pixels use the safe scalar helper.
    unsafe {
        use std::arch::x86_64::*;

        let masks = [
            [
                _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, -1, -1, -1, -1, -1, -1, -1, -1),
            ],
            [
                _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1),
            ],
            [
                _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1),
            ],
        ];
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);
        let four = _mm256_set1_ps(4.0);
        let six = _mm256_set1_ps(6.0);
        let scale = _mm256_set1_ps(255.0);
        let factor = _mm256_set1_ps(factor_value);
        let half = _mm256_set1_ps(0.5);
        let mut offset = 0usize;
        while offset + 24 <= data.len() {
            let pointer = data.as_ptr().add(offset);
            let a = _mm_loadu_si128(pointer.cast());
            let b = _mm_loadl_epi64(pointer.add(16).cast());
            let mut channels = [_mm256_setzero_ps(); 3];
            for (channel, masks) in masks.iter().enumerate() {
                let bytes =
                    _mm_or_si128(_mm_shuffle_epi8(a, masks[0]), _mm_shuffle_epi8(b, masks[1]));
                channels[channel] =
                    _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)), scale);
            }
            let [r, g, b] = channels;
            let maximum = _mm256_max_ps(_mm256_max_ps(r, g), b);
            let minimum = _mm256_min_ps(_mm256_min_ps(r, g), b);
            let delta = _mm256_sub_ps(maximum, minimum);
            let r_mask = _mm256_cmp_ps::<_CMP_EQ_OQ>(maximum, r);
            let g_mask = _mm256_andnot_ps(r_mask, _mm256_cmp_ps::<_CMP_EQ_OQ>(maximum, g));
            let mut r_hue = _mm256_div_ps(_mm256_sub_ps(g, b), delta);
            r_hue = _mm256_add_ps(
                r_hue,
                _mm256_and_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(r_hue, zero), six),
            );
            r_hue = _mm256_div_ps(r_hue, six);
            let g_hue = _mm256_div_ps(
                _mm256_add_ps(_mm256_div_ps(_mm256_sub_ps(b, r), delta), two),
                six,
            );
            let b_hue = _mm256_div_ps(
                _mm256_add_ps(_mm256_div_ps(_mm256_sub_ps(r, g), delta), four),
                six,
            );
            let mut hue = _mm256_blendv_ps(b_hue, g_hue, g_mask);
            hue = _mm256_blendv_ps(hue, r_hue, r_mask);
            hue = _mm256_blendv_ps(hue, zero, _mm256_cmp_ps::<_CMP_EQ_OQ>(delta, zero));
            hue = _mm256_add_ps(hue, factor);
            hue = _mm256_add_ps(
                hue,
                _mm256_and_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(hue, zero), one),
            );
            hue = _mm256_sub_ps(
                hue,
                _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(hue, one), one),
            );
            let saturation = _mm256_blendv_ps(
                _mm256_div_ps(delta, maximum),
                zero,
                _mm256_cmp_ps::<_CMP_EQ_OQ>(maximum, zero),
            );
            let sector_value = _mm256_mul_ps(hue, six);
            let sector_floor = _mm256_floor_ps(sector_value);
            let sectors = _mm256_cvttps_epi32(sector_floor);
            let fraction = _mm256_sub_ps(sector_value, sector_floor);
            let p = _mm256_mul_ps(maximum, _mm256_sub_ps(one, saturation));
            let q = _mm256_mul_ps(
                maximum,
                _mm256_sub_ps(one, _mm256_mul_ps(saturation, fraction)),
            );
            let t = _mm256_mul_ps(
                maximum,
                _mm256_sub_ps(one, _mm256_mul_ps(saturation, _mm256_sub_ps(one, fraction))),
            );
            let sector_mask = |sector: i32| {
                _mm256_castsi256_ps(_mm256_cmpeq_epi32(sectors, _mm256_set1_epi32(sector)))
            };
            let select = |values: [__m256; 6]| {
                let mut result = values[5];
                for sector in (0..5).rev() {
                    result = _mm256_blendv_ps(result, values[sector], sector_mask(sector as i32));
                }
                result
            };
            let outputs = [
                select([maximum, q, p, p, t, maximum]),
                select([t, maximum, maximum, q, p, p]),
                select([p, p, t, maximum, maximum, q]),
            ];
            let mut packed = [[0u8; 8]; 3];
            for channel in 0..3 {
                let rounded =
                    _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(outputs[channel], scale), half));
                let integers = _mm256_cvttps_epi32(rounded);
                let words = _mm_packus_epi32(
                    _mm256_castsi256_si128(integers),
                    _mm256_extracti128_si256(integers, 1),
                );
                let bytes = _mm_packus_epi16(words, words);
                _mm_storel_epi64(packed[channel].as_mut_ptr().cast(), bytes);
            }
            for pixel in 0..8 {
                data[offset + pixel * 3] = packed[0][pixel];
                data[offset + pixel * 3 + 1] = packed[1][pixel];
                data[offset + pixel * 3 + 2] = packed[2][pixel];
            }
            offset += 24;
        }
        for pixel in data[offset..].chunks_exact_mut(3) {
            adjust_hue_pixel(pixel, factor_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_kernel_matches_scalar_at_vector_boundaries() {
        let matrix = [
            [18841, -1724, -335],
            [-901, 18018, -335],
            [-901, -1724, 19742],
        ];
        for pixel_count in 0..=129 {
            let mut expected: Vec<_> = (0..pixel_count * 3).map(|i| (i * 73) as u8).collect();
            let mut actual = expected.clone();
            apply_q14_scalar(&mut expected, matrix, -4915);
            apply_q14(&mut actual, matrix, -4915);
            assert_eq!(actual, expected, "pixel_count={pixel_count}");
        }
    }

    #[test]
    fn accumulator_check_has_an_exact_single_coefficient_boundary() {
        let maximum = (i32::MAX - 8192) / 255;
        let matrix = |coefficient| {
            [
                [coefficient, 0, 0],
                [0, coefficient, 0],
                [0, 0, coefficient],
            ]
        };
        assert!(q14_accumulator_fits(matrix(maximum), 0));
        assert!(!q14_accumulator_fits(matrix(maximum + 1), 0));
        assert!(!q14_accumulator_fits([[0; 3]; 3], i32::MAX));
    }

    #[test]
    fn hue_dispatch_matches_scalar_at_vector_and_sector_boundaries() {
        for pixel_count in 0..=65 {
            let mut source: Vec<_> = (0..pixel_count * 3)
                .map(|index| ((index * 73 + index / 7 * 19) & 255) as u8)
                .collect();
            source.extend_from_slice(&[
                0, 0, 0, 73, 73, 73, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255,
                255, 0, 255, 255, 0, 255,
            ]);
            for factor in [-0.5, -0.231, -0.0001, 0.0001, 0.137, 0.5] {
                let mut expected = source.clone();
                let mut actual = source.clone();
                for pixel in expected.chunks_exact_mut(3) {
                    adjust_hue_pixel(pixel, factor);
                }
                adjust_hue(&mut actual, factor);
                assert_eq!(actual, expected, "pixels={pixel_count} factor={factor}");
            }
        }
    }
}
