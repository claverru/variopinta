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
    const ROUND: i32 = 8192;
    for pixel in data.chunks_exact_mut(3) {
        let [r, g, b] = [pixel[0] as i32, pixel[1] as i32, pixel[2] as i32];
        for channel in 0..3 {
            let value =
                matrix[channel][0] * r + matrix[channel][1] * g + matrix[channel][2] * b + bias;
            pixel[channel] = ((value + ROUND) >> 14).clamp(0, 255) as u8;
        }
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
}
