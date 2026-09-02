pub(crate) fn grayscale(data: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection guards the AVX2 implementation.
        unsafe { grayscale_avx2(data) };
        return;
    }
    grayscale_scalar(data);
}

pub(crate) fn horizontal_flip(data: &mut [u8], height: usize, width: usize) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection guards the AVX2 implementation.
        unsafe { horizontal_flip_avx2(data, height, width) };
        return;
    }
    horizontal_flip_scalar(data, height, width);
}

fn horizontal_flip_scalar(data: &mut [u8], height: usize, width: usize) {
    let ptr = data.as_mut_ptr();
    for y in 0..height {
        for x in 0..width / 2 {
            let a = (y * width + x) * 3;
            let b = (y * width + width - 1 - x) * 3;
            // SAFETY: the RGB blocks are disjoint and in bounds.
            unsafe {
                std::ptr::swap_nonoverlapping(ptr.add(a), ptr.add(b), 3);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
const fn reverse_rgb_mask(out_chunk: usize, in_chunk: usize) -> [i8; 16] {
    let mut mask = [-1i8; 16];
    let mut lane = 0;
    while lane < 16 {
        let output = out_chunk * 16 + lane;
        let pixel = output / 3;
        let channel = output % 3;
        let source = (15 - pixel) * 3 + channel;
        if source / 16 == in_chunk {
            mask[lane] = (source % 16) as i8;
        }
        lane += 1;
    }
    mask
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reverse_rgb16(
    a: std::arch::x86_64::__m128i,
    b: std::arch::x86_64::__m128i,
    c: std::arch::x86_64::__m128i,
) -> (
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
) {
    // SAFETY: the caller guarantees AVX2; inputs and outputs are register values.
    unsafe {
        use std::arch::x86_64::*;
        const MASKS: [[[i8; 16]; 3]; 3] = [
            [
                reverse_rgb_mask(0, 0),
                reverse_rgb_mask(0, 1),
                reverse_rgb_mask(0, 2),
            ],
            [
                reverse_rgb_mask(1, 0),
                reverse_rgb_mask(1, 1),
                reverse_rgb_mask(1, 2),
            ],
            [
                reverse_rgb_mask(2, 0),
                reverse_rgb_mask(2, 1),
                reverse_rgb_mask(2, 2),
            ],
        ];
        let input = [a, b, c];
        let mut output = [_mm_setzero_si128(); 3];
        for out_chunk in 0..3 {
            for (in_chunk, input_chunk) in input.iter().enumerate() {
                let mask = _mm_loadu_si128(MASKS[out_chunk][in_chunk].as_ptr() as *const __m128i);
                output[out_chunk] =
                    _mm_or_si128(output[out_chunk], _mm_shuffle_epi8(*input_chunk, mask));
            }
        }
        (output[0], output[1], output[2])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_flip_avx2(data: &mut [u8], height: usize, width: usize) {
    // SAFETY: the caller guarantees AVX2 and each access stays within an RGB row.
    unsafe {
        use std::arch::x86_64::*;
        let ptr = data.as_mut_ptr();
        for y in 0..height {
            let row = y * width * 3;
            let mut x = 0usize;
            while x + 16 <= width / 2 {
                let left = row + x * 3;
                let right = row + (width - x - 16) * 3;
                let la = _mm_loadu_si128(ptr.add(left) as *const __m128i);
                let lb = _mm_loadu_si128(ptr.add(left + 16) as *const __m128i);
                let lc = _mm_loadu_si128(ptr.add(left + 32) as *const __m128i);
                let ra = _mm_loadu_si128(ptr.add(right) as *const __m128i);
                let rb = _mm_loadu_si128(ptr.add(right + 16) as *const __m128i);
                let rc = _mm_loadu_si128(ptr.add(right + 32) as *const __m128i);
                let (la, lb, lc) = reverse_rgb16(la, lb, lc);
                let (ra, rb, rc) = reverse_rgb16(ra, rb, rc);
                _mm_storeu_si128(ptr.add(left) as *mut __m128i, ra);
                _mm_storeu_si128(ptr.add(left + 16) as *mut __m128i, rb);
                _mm_storeu_si128(ptr.add(left + 32) as *mut __m128i, rc);
                _mm_storeu_si128(ptr.add(right) as *mut __m128i, la);
                _mm_storeu_si128(ptr.add(right + 16) as *mut __m128i, lb);
                _mm_storeu_si128(ptr.add(right + 32) as *mut __m128i, lc);
                x += 16;
            }
            while x < width / 2 {
                let left = row + x * 3;
                let right = row + (width - 1 - x) * 3;
                std::ptr::swap_nonoverlapping(ptr.add(left), ptr.add(right), 3);
                x += 1;
            }
        }
    }
}

pub(crate) fn vertical_flip(data: &mut [u8], height: usize, width: usize) {
    let row_len = width * 3;
    for top in 0..height / 2 {
        let bottom = height - 1 - top;
        let (head, tail) = data.split_at_mut(bottom * row_len);
        head[top * row_len..(top + 1) * row_len].swap_with_slice(&mut tail[..row_len]);
    }
}

pub(crate) fn vertical_flip_into(
    source: &[u8],
    destination: &mut [u8],
    height: usize,
    width: usize,
) {
    let row_len = width * 3;
    assert_eq!(source.len(), destination.len());
    assert_eq!(source.len(), height * row_len);
    for (output_row, input_row) in destination
        .chunks_exact_mut(row_len)
        .zip(source.chunks_exact(row_len).rev())
        .take(height)
    {
        output_row.copy_from_slice(input_row);
    }
}

pub(crate) fn grayscale_scalar(data: &mut [u8]) {
    for pixel in data.chunks_exact_mut(3) {
        let gray =
            (77 * u16::from(pixel[0]) + 150 * u16::from(pixel[1]) + 29 * u16::from(pixel[2]) + 128)
                >> 8;
        pixel.fill(gray as u8);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn grayscale_avx2(data: &mut [u8]) {
    // SAFETY: the caller guarantees AVX2 and the loop accesses complete 48-byte RGB blocks.
    unsafe {
        use std::arch::x86_64::*;
        let masks = [
            [
                _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13),
            ],
            [
                _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14),
            ],
            [
                _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1),
                _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15),
            ],
        ];
        let ptr = data.as_mut_ptr();
        let mut offset = 0;
        while offset + 48 <= data.len() {
            let chunks = [
                _mm_loadu_si128(ptr.add(offset) as *const __m128i),
                _mm_loadu_si128(ptr.add(offset + 16) as *const __m128i),
                _mm_loadu_si128(ptr.add(offset + 32) as *const __m128i),
            ];
            let channel = |index: usize| {
                _mm_or_si128(
                    _mm_or_si128(
                        _mm_shuffle_epi8(chunks[0], masks[index][0]),
                        _mm_shuffle_epi8(chunks[1], masks[index][1]),
                    ),
                    _mm_shuffle_epi8(chunks[2], masks[index][2]),
                )
            };
            let r = channel(0);
            let g = channel(1);
            let b = channel(2);
            let weighted = |r: __m128i, g: __m128i, b: __m128i| {
                let sum = _mm_add_epi16(
                    _mm_add_epi16(
                        _mm_mullo_epi16(r, _mm_set1_epi16(77)),
                        _mm_mullo_epi16(g, _mm_set1_epi16(150)),
                    ),
                    _mm_add_epi16(_mm_mullo_epi16(b, _mm_set1_epi16(29)), _mm_set1_epi16(128)),
                );
                _mm_srli_epi16::<8>(sum)
            };
            let low = weighted(
                _mm_cvtepu8_epi16(r),
                _mm_cvtepu8_epi16(g),
                _mm_cvtepu8_epi16(b),
            );
            let high = weighted(
                _mm_cvtepu8_epi16(_mm_srli_si128::<8>(r)),
                _mm_cvtepu8_epi16(_mm_srli_si128::<8>(g)),
                _mm_cvtepu8_epi16(_mm_srli_si128::<8>(b)),
            );
            let gray = _mm_packus_epi16(low, high);
            let out_masks = [
                _mm_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5),
                _mm_setr_epi8(5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10),
                _mm_setr_epi8(
                    10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15,
                ),
            ];
            for (chunk, mask) in out_masks.into_iter().enumerate() {
                _mm_storeu_si128(
                    ptr.add(offset + chunk * 16) as *mut __m128i,
                    _mm_shuffle_epi8(gray, mask),
                );
            }
            offset += 48;
        }
        grayscale_scalar(&mut data[offset..]);
    }
}

pub(crate) fn invert(data: &mut [u8]) {
    apply_byte_map_in_place(data, ByteMap::Invert);
}

pub(crate) fn invert_into(source: &[u8], destination: &mut [u8]) {
    apply_byte_map_into(source, destination, ByteMap::Invert);
}

pub(crate) fn solarize(data: &mut [u8], threshold: u8) {
    apply_byte_map_in_place(data, ByteMap::Solarize(threshold));
}

pub(crate) fn solarize_into(source: &[u8], destination: &mut [u8], threshold: u8) {
    apply_byte_map_into(source, destination, ByteMap::Solarize(threshold));
}

pub(crate) fn posterize(data: &mut [u8], bits: u8) {
    apply_byte_map_in_place(data, ByteMap::Posterize(bits));
}

pub(crate) fn posterize_into(source: &[u8], destination: &mut [u8], bits: u8) {
    apply_byte_map_into(source, destination, ByteMap::Posterize(bits));
}

#[derive(Clone, Copy)]
enum ByteMap {
    Invert,
    Solarize(u8),
    Posterize(u8),
}

fn apply_byte(value: u8, operation: ByteMap) -> u8 {
    match operation {
        ByteMap::Invert => 255 - value,
        ByteMap::Solarize(threshold) if value >= threshold => 255 - value,
        ByteMap::Solarize(_) => value,
        ByteMap::Posterize(bits) => value & (u8::MAX << (8 - bits)),
    }
}

fn apply_byte_map_scalar(data: &mut [u8], operation: ByteMap) {
    for value in data {
        *value = apply_byte(*value, operation);
    }
}

fn apply_byte_map_into_scalar(source: &[u8], destination: &mut [u8], operation: ByteMap) {
    for (&input, output) in source.iter().zip(destination) {
        *output = apply_byte(input, operation);
    }
}

fn apply_byte_map_in_place(data: &mut [u8], operation: ByteMap) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let ptr = data.as_mut_ptr();
        // SAFETY: runtime detection guards AVX2; source and destination cover data.len() bytes.
        unsafe { apply_byte_map_avx2(ptr, ptr, data.len(), operation) };
        return;
    }
    apply_byte_map_scalar(data, operation);
}

fn apply_byte_map_into(source: &[u8], destination: &mut [u8], operation: ByteMap) {
    assert_eq!(source.len(), destination.len());
    let len = source.len();
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection guards AVX2 and both pointers cover len bytes.
        unsafe { apply_byte_map_avx2(source.as_ptr(), destination.as_mut_ptr(), len, operation) };
        return;
    }
    apply_byte_map_into_scalar(&source[..len], &mut destination[..len], operation);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_byte_map_avx2(
    source: *const u8,
    destination: *mut u8,
    len: usize,
    operation: ByteMap,
) {
    // SAFETY: the caller guarantees AVX2 and valid source/destination ranges of len bytes.
    unsafe {
        use std::arch::x86_64::*;
        let invert = _mm256_set1_epi8(-1);
        let mut offset = 0;
        while offset + 32 <= len {
            let values = _mm256_loadu_si256(source.add(offset) as *const __m256i);
            let result = match operation {
                ByteMap::Invert => _mm256_xor_si256(values, invert),
                ByteMap::Solarize(threshold) => {
                    let threshold = _mm256_set1_epi8(threshold as i8);
                    let mask = _mm256_cmpeq_epi8(_mm256_max_epu8(values, threshold), values);
                    _mm256_blendv_epi8(values, _mm256_xor_si256(values, invert), mask)
                }
                ByteMap::Posterize(bits) => {
                    _mm256_and_si256(values, _mm256_set1_epi8((u8::MAX << (8 - bits)) as i8))
                }
            };
            _mm256_storeu_si256(destination.add(offset) as *mut __m256i, result);
            offset += 32;
        }
        while offset < len {
            destination
                .add(offset)
                .write(apply_byte(source.add(offset).read(), operation));
            offset += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_kernels_handle_scalar_tails() {
        let mut image = vec![0, 64, 255, 17, 128, 240];
        grayscale_scalar(&mut image);
        assert_eq!(image[0], image[1]);
        assert_eq!(image[1], image[2]);
        assert_eq!(image[3], image[4]);
        assert_eq!(image[4], image[5]);

        invert(&mut image);
        solarize(&mut image, 128);
        posterize(&mut image, 4);
        assert!(image.iter().all(|value| value & 0x0f == 0));
    }

    #[test]
    fn solarize_simd_matches_scalar() {
        let input: Vec<u8> = (0..=255).chain(0..19).collect();
        for threshold in [0, 1, 127, 128, 254, 255] {
            let mut expected = input.clone();
            apply_byte_map_scalar(&mut expected, ByteMap::Solarize(threshold));
            let mut actual = input.clone();
            solarize(&mut actual, threshold);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn byte_map_simd_and_direct_output_match_scalar() {
        let input: Vec<u8> = (0..=255).chain(0..19).collect();
        let operations = [
            ByteMap::Invert,
            ByteMap::Solarize(0),
            ByteMap::Solarize(127),
            ByteMap::Solarize(255),
            ByteMap::Posterize(1),
            ByteMap::Posterize(4),
            ByteMap::Posterize(8),
        ];
        for operation in operations {
            let mut expected = input.clone();
            apply_byte_map_scalar(&mut expected, operation);

            let mut in_place = input.clone();
            apply_byte_map_in_place(&mut in_place, operation);
            assert_eq!(in_place, expected);

            let mut direct = vec![0; input.len()];
            apply_byte_map_into(&input, &mut direct, operation);
            assert_eq!(direct, expected);
        }
    }

    #[test]
    fn grayscale_simd_matches_scalar() {
        let input: Vec<u8> = (0..=255).chain(0..=255).chain(0..37).collect();
        let mut expected = input[..input.len() / 3 * 3].to_vec();
        grayscale_scalar(&mut expected);
        let mut actual = input[..input.len() / 3 * 3].to_vec();
        grayscale(&mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn vertical_flip_handles_odd_rectangles() {
        let mut image: Vec<u8> = (0..45).collect();
        let middle = image[15..30].to_vec();
        vertical_flip(&mut image, 3, 5);
        assert_eq!(&image[15..30], middle);
        assert_eq!(image[0], 30);
        assert_eq!(image[30], 0);

        let source: Vec<u8> = (0..45).collect();
        let mut direct = vec![0; source.len()];
        vertical_flip_into(&source, &mut direct, 3, 5);
        assert_eq!(direct, image);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn horizontal_flip_simd_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for height in [1, 2, 3, 7] {
            for width in 1..=96 {
                let source: Vec<_> = (0..height * width * 3)
                    .map(|index| ((index * 73 + index / 7 * 19) & 255) as u8)
                    .collect();
                let mut scalar = source.clone();
                let mut simd = source;
                horizontal_flip_scalar(&mut scalar, height, width);
                // SAFETY: AVX2 support is checked above.
                unsafe { horizontal_flip_avx2(&mut simd, height, width) };
                assert_eq!(simd, scalar, "{height}x{width}");
            }
        }
    }
}
