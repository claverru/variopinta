use crate::{CoreError, CoreResult};

pub(crate) fn horizontal_5x5(
    data: &[u8],
    temp: &mut [u16],
    start: usize,
    end: usize,
    kernel: [u32; 5],
) -> CoreResult<()> {
    if start > end
        || end > data.len()
        || temp.len() < data.len()
        || start < 6
        || end.saturating_add(6) > data.len()
    {
        return Err(CoreError::Runtime(
            "invalid horizontal blur kernel span".into(),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and the checked interior span bounds every access.
        unsafe { horizontal_5x5_avx2(data, temp, start, end, kernel) };
        return Ok(());
    }
    horizontal_5x5_scalar(data, temp, start, end, kernel);
    Ok(())
}

pub(crate) fn vertical_5x5(
    temp: &[u16],
    data: &mut [u8],
    row: usize,
    bytes_per_row: usize,
    kernel: [u32; 5],
) -> CoreResult<()> {
    let minimum_row = bytes_per_row.checked_mul(2);
    let upper_bound = bytes_per_row
        .checked_mul(3)
        .and_then(|span| row.checked_add(span));
    if bytes_per_row == 0
        || data.len() < temp.len()
        || minimum_row.is_none_or(|minimum| row < minimum)
        || upper_bound.is_none_or(|upper| upper > temp.len())
    {
        return Err(CoreError::Runtime(
            "invalid vertical blur kernel span".into(),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and the checked interior row bounds every access.
        unsafe { vertical_5x5_avx2(temp, data, row, bytes_per_row, kernel) };
        return Ok(());
    }
    vertical_5x5_scalar(temp, data, row, bytes_per_row, kernel);
    Ok(())
}

#[inline]
fn horizontal_5x5_scalar(
    data: &[u8],
    temp: &mut [u16],
    mut index: usize,
    end: usize,
    kernel: [u32; 5],
) {
    while index < end {
        temp[index] = (data[index - 6] as u32 * kernel[0]
            + data[index - 3] as u32 * kernel[1]
            + data[index] as u32 * kernel[2]
            + data[index + 3] as u32 * kernel[3]
            + data[index + 6] as u32 * kernel[4]) as u16;
        index += 1;
    }
}

#[inline]
fn vertical_5x5_scalar(
    temp: &[u16],
    data: &mut [u8],
    row: usize,
    bytes_per_row: usize,
    kernel: [u32; 5],
) {
    for index in 0..bytes_per_row {
        let value = temp[row + index - 2 * bytes_per_row] as u32 * kernel[0]
            + temp[row + index - bytes_per_row] as u32 * kernel[1]
            + temp[row + index] as u32 * kernel[2]
            + temp[row + index + bytes_per_row] as u32 * kernel[3]
            + temp[row + index + 2 * bytes_per_row] as u32 * kernel[4];
        data[row + index] = ((value + 32768) >> 16).min(255) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_5x5_avx2(
    data: &[u8],
    temp: &mut [u16],
    mut index: usize,
    end: usize,
    kernel: [u32; 5],
) {
    // SAFETY: the safe wrapper checks the complete source and destination spans.
    unsafe {
        use std::arch::x86_64::*;
        let weight = kernel.map(|value| _mm256_set1_epi16(value as i16));
        while index + 16 <= end {
            let load = |offset: isize| {
                let bytes = _mm_loadu_si128(
                    data.as_ptr()
                        .offset(index as isize + offset)
                        .cast::<__m128i>(),
                );
                _mm256_cvtepu8_epi16(bytes)
            };
            let value = _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mullo_epi16(load(-6), weight[0]),
                        _mm256_mullo_epi16(load(-3), weight[1]),
                    ),
                    _mm256_mullo_epi16(load(0), weight[2]),
                ),
                _mm256_add_epi16(
                    _mm256_mullo_epi16(load(3), weight[3]),
                    _mm256_mullo_epi16(load(6), weight[4]),
                ),
            );
            _mm256_storeu_si256(temp.as_mut_ptr().add(index).cast(), value);
            index += 16;
        }
        horizontal_5x5_scalar(data, temp, index, end, kernel);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vertical_5x5_avx2(
    temp: &[u16],
    data: &mut [u8],
    row: usize,
    bytes_per_row: usize,
    kernel: [u32; 5],
) {
    // SAFETY: the safe wrapper checks two complete source rows on each side and the output row.
    unsafe {
        use std::arch::x86_64::*;
        let weight = kernel.map(|value| _mm256_set1_epi32(value as i32));
        let rounding = _mm256_set1_epi32(32768);
        let zero = _mm_setzero_si128();
        let mut index = 0usize;
        while index + 8 <= bytes_per_row {
            let load = |row_offset: isize| {
                let source = temp
                    .as_ptr()
                    .offset(row as isize + row_offset * bytes_per_row as isize + index as isize);
                _mm256_cvtepu16_epi32(_mm_loadu_si128(source.cast()))
            };
            let value = _mm256_srli_epi32::<16>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(load(-2), weight[0]),
                        _mm256_mullo_epi32(load(-1), weight[1]),
                    ),
                    _mm256_mullo_epi32(load(0), weight[2]),
                ),
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(load(1), weight[3]),
                        _mm256_mullo_epi32(load(2), weight[4]),
                    ),
                    rounding,
                ),
            ));
            let packed16 = _mm_packs_epi32(
                _mm256_castsi256_si128(value),
                _mm256_extracti128_si256::<1>(value),
            );
            let packed8 = _mm_packus_epi16(packed16, zero);
            _mm_storel_epi64(data.as_mut_ptr().add(row + index).cast(), packed8);
            index += 8;
        }
        for trailing in index..bytes_per_row {
            let value = temp[row + trailing - 2 * bytes_per_row] as u32 * kernel[0]
                + temp[row + trailing - bytes_per_row] as u32 * kernel[1]
                + temp[row + trailing] as u32 * kernel[2]
                + temp[row + trailing + bytes_per_row] as u32 * kernel[3]
                + temp[row + trailing + 2 * bytes_per_row] as u32 * kernel[4];
            data[row + trailing] = ((value + 32768) >> 16).min(255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_passes_match_scalar_at_vector_boundaries() {
        let kernel = [16, 64, 96, 64, 16];
        for interior in 0..=65 {
            let source: Vec<_> = (0..interior + 12).map(|i| (i * 73) as u8).collect();
            let mut expected = vec![0; source.len()];
            let mut actual = expected.clone();
            horizontal_5x5_scalar(&source, &mut expected, 6, 6 + interior, kernel);
            horizontal_5x5(&source, &mut actual, 6, 6 + interior, kernel).unwrap();
            assert_eq!(actual, expected, "horizontal interior={interior}");
        }

        for bytes_per_row in 1..=65 {
            let temp: Vec<_> = (0..bytes_per_row * 5)
                .map(|i| ((i * 977 + 313) % 65281) as u16)
                .collect();
            let row = 2 * bytes_per_row;
            let mut expected = vec![0; temp.len()];
            let mut actual = expected.clone();
            vertical_5x5_scalar(&temp, &mut expected, row, bytes_per_row, kernel);
            vertical_5x5(&temp, &mut actual, row, bytes_per_row, kernel).unwrap();
            assert_eq!(actual, expected, "vertical width={bytes_per_row}");
        }
    }
}
