use crate::{CoreError, CoreResult};

pub(crate) fn bilinear_constant(
    data: &[u8],
    height: usize,
    width: usize,
    matrix: [f32; 6],
    mut output: Vec<u8>,
) -> CoreResult<Vec<u8>> {
    let expected = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| CoreError::Runtime("affine dimensions overflow".into()))?;
    if height == 0 || width == 0 || data.len() != expected || output.len() != expected {
        return Err(CoreError::Runtime(
            "affine kernel requires matching non-empty RGB buffers".into(),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and the safe wrapper validates both RGB buffers.
        unsafe { bilinear_avx2(data, height, width, matrix, &mut output) }?;
        return Ok(output);
    }
    bilinear_scalar(data, height, width, matrix, &mut output)?;
    Ok(output)
}

fn bilinear_scalar(
    data: &[u8],
    height: usize,
    width: usize,
    matrix: [f32; 6],
    output: &mut [u8],
) -> CoreResult<()> {
    const Q: i32 = 16;
    const Q_SCALE: f64 = (1 << Q) as f64;
    for y in 0..height {
        let (sx0, sy0, dsx, dsy) = source_coordinates(y, matrix);
        let dsx_q = quantize_q16(dsx, Q_SCALE)?;
        let dsy_q = quantize_q16(dsy, Q_SCALE)?;
        let (mut x_lo, mut x_hi) = (0i64, width as i64);
        valid_span(
            &mut x_lo,
            &mut x_hi,
            sx0,
            dsx,
            width.saturating_sub(1) as f64,
            width,
        );
        valid_span(
            &mut x_lo,
            &mut x_hi,
            sy0,
            dsy,
            height.saturating_sub(1) as f64,
            width,
        );
        if x_lo >= x_hi {
            continue;
        }
        let x_lo = x_lo as usize;
        let x_hi = x_hi as usize;
        let mut sx_q = quantize_q16(sx0 + dsx * x_lo as f64, Q_SCALE)?;
        let mut sy_q = quantize_q16(sy0 + dsy * x_lo as f64, Q_SCALE)?;
        let mut destination = (y * width + x_lo) * 3;
        for x in x_lo..x_hi {
            let x0 = (sx_q >> Q) as usize;
            let y0 = (sy_q >> Q) as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let wx = ((sx_q & 0xffff) as u32) >> 8;
            let wy = ((sy_q & 0xffff) as u32) >> 8;
            let inv_wx = 256 - wx;
            let inv_wy = 256 - wy;
            let row0 = y0 * width * 3;
            let row1 = y1 * width * 3;
            let [p00, p01, p10, p11] = [row0 + x0 * 3, row0 + x1 * 3, row1 + x0 * 3, row1 + x1 * 3];
            for channel in 0..3 {
                let top = data[p00 + channel] as u32 * inv_wx + data[p01 + channel] as u32 * wx;
                let bottom = data[p10 + channel] as u32 * inv_wx + data[p11 + channel] as u32 * wx;
                output[destination + channel] = ((top * inv_wy + bottom * wy + 32768) >> 16) as u8;
            }
            if x + 1 < x_hi {
                sx_q = sx_q
                    .checked_add(dsx_q)
                    .ok_or_else(|| CoreError::Invalid("Affine Q16 coordinate overflow".into()))?;
                sy_q = sy_q
                    .checked_add(dsy_q)
                    .ok_or_else(|| CoreError::Invalid("Affine Q16 coordinate overflow".into()))?;
            }
            destination += 3;
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bilinear_avx2(
    data: &[u8],
    height: usize,
    width: usize,
    matrix: [f32; 6],
    output: &mut [u8],
) -> CoreResult<()> {
    // SAFETY: the safe wrapper validates RGB lengths. The analytical span keeps source
    // coordinates in bounds; the four-byte load is separately guarded at the final pixel.
    unsafe {
        use std::arch::x86_64::*;
        const Q: i32 = 16;
        const Q_SCALE: f64 = (1 << Q) as f64;
        let zero = _mm_setzero_si128();
        let rounding = _mm_set1_epi32(32768);
        for y in 0..height {
            let (sx0, sy0, dsx, dsy) = source_coordinates(y, matrix);
            let dsx_q = quantize_q16(dsx, Q_SCALE)?;
            let dsy_q = quantize_q16(dsy, Q_SCALE)?;
            let (mut x_lo, mut x_hi) = (0i64, width as i64);
            valid_span(
                &mut x_lo,
                &mut x_hi,
                sx0,
                dsx,
                width.saturating_sub(1) as f64,
                width,
            );
            valid_span(
                &mut x_lo,
                &mut x_hi,
                sy0,
                dsy,
                height.saturating_sub(1) as f64,
                width,
            );
            if x_lo >= x_hi {
                continue;
            }
            let x_lo = x_lo as usize;
            let x_hi = x_hi as usize;
            let mut sx_q = quantize_q16(sx0 + dsx * x_lo as f64, Q_SCALE)?;
            let mut sy_q = quantize_q16(sy0 + dsy * x_lo as f64, Q_SCALE)?;
            let mut destination = (y * width + x_lo) * 3;
            for x in x_lo..x_hi {
                let x0 = (sx_q >> Q) as usize;
                let y0 = (sy_q >> Q) as usize;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);
                let wx = ((sx_q & 0xffff) as u32) >> 8;
                let wy = ((sy_q & 0xffff) as u32) >> 8;
                let inv_wx = 256 - wx;
                let inv_wy = 256 - wy;
                let row0 = y0 * width * 3;
                let row1 = y1 * width * 3;
                let [p00, p01, p10, p11] =
                    [row0 + x0 * 3, row0 + x1 * 3, row1 + x0 * 3, row1 + x1 * 3];
                if p11 + 4 <= data.len() {
                    let load = |index: usize| {
                        let packed = std::ptr::read_unaligned(data.as_ptr().add(index).cast());
                        _mm_cvtepu8_epi16(_mm_cvtsi32_si128(packed))
                    };
                    let wxv = _mm_set1_epi16(wx as i16);
                    let inv_wxv = _mm_set1_epi16(inv_wx as i16);
                    let top = _mm_add_epi16(
                        _mm_mullo_epi16(load(p00), inv_wxv),
                        _mm_mullo_epi16(load(p01), wxv),
                    );
                    let bottom = _mm_add_epi16(
                        _mm_mullo_epi16(load(p10), inv_wxv),
                        _mm_mullo_epi16(load(p11), wxv),
                    );
                    let value = _mm_srli_epi32::<16>(_mm_add_epi32(
                        _mm_add_epi32(
                            _mm_mullo_epi32(_mm_cvtepu16_epi32(top), _mm_set1_epi32(inv_wy as i32)),
                            _mm_mullo_epi32(_mm_cvtepu16_epi32(bottom), _mm_set1_epi32(wy as i32)),
                        ),
                        rounding,
                    ));
                    let packed = _mm_packus_epi16(_mm_packs_epi32(value, zero), zero);
                    let rgb = _mm_cvtsi128_si32(packed) as u32;
                    output[destination] = rgb as u8;
                    output[destination + 1] = (rgb >> 8) as u8;
                    output[destination + 2] = (rgb >> 16) as u8;
                } else {
                    for channel in 0..3 {
                        let top =
                            data[p00 + channel] as u32 * inv_wx + data[p01 + channel] as u32 * wx;
                        let bottom =
                            data[p10 + channel] as u32 * inv_wx + data[p11 + channel] as u32 * wx;
                        output[destination + channel] =
                            ((top * inv_wy + bottom * wy + 32768) >> 16) as u8;
                    }
                }
                if x + 1 < x_hi {
                    sx_q = sx_q.checked_add(dsx_q).ok_or_else(|| {
                        CoreError::Invalid("Affine Q16 coordinate overflow".into())
                    })?;
                    sy_q = sy_q.checked_add(dsy_q).ok_or_else(|| {
                        CoreError::Invalid("Affine Q16 coordinate overflow".into())
                    })?;
                }
                destination += 3;
            }
        }
    }
    Ok(())
}

fn quantize_q16(value: f64, scale: f64) -> CoreResult<i64> {
    let scaled = value * scale;
    if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return Err(CoreError::Invalid("Affine Q16 coordinate overflow".into()));
    }
    Ok(scaled as i64)
}

fn source_coordinates(y: usize, matrix: [f32; 6]) -> (f64, f64, f64, f64) {
    (
        f64::from(matrix[1]) * y as f64 + f64::from(matrix[2]),
        f64::from(matrix[4]) * y as f64 + f64::from(matrix[5]),
        f64::from(matrix[0]),
        f64::from(matrix[3]),
    )
}

pub(crate) fn valid_span(
    lower: &mut i64,
    upper: &mut i64,
    start: f64,
    step: f64,
    source_upper: f64,
    destination_width: usize,
) {
    if step.abs() < 1e-12 {
        if !(start >= 0.0 && start < source_upper) {
            *lower = 0;
            *upper = 0;
        }
        return;
    }
    let a = -start / step;
    let b = (source_upper - start) / step;
    let (minimum, maximum) = if step > 0.0 { (a, b) } else { (b, a) };
    *lower = (*lower).max(minimum.ceil().max(0.0) as i64);
    *upper = (*upper).min(maximum.ceil().min(destination_width as f64) as i64);
    if *lower > *upper {
        *lower = 0;
        *upper = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_kernel_matches_safe_oracle_at_boundaries() {
        for (height, width) in [
            (1, 1),
            (1, 17),
            (17, 1),
            (3, 5),
            (17, 19),
            (31, 33),
            (63, 65),
        ] {
            let source: Vec<_> = (0..height * width * 3).map(|i| (i * 73) as u8).collect();
            for matrix in [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.91, -0.27, 2.3, 0.34, 1.07, -1.7],
                [-0.7, 0.4, width as f32, 0.2, -0.8, height as f32],
            ] {
                let mut expected = vec![0xa5; source.len()];
                bilinear_scalar(&source, height, width, matrix, &mut expected).unwrap();
                let actual =
                    bilinear_constant(&source, height, width, matrix, vec![0xa5; source.len()])
                        .unwrap();
                assert_eq!(actual, expected, "{height}x{width} {matrix:?}");
            }
        }
    }
}
