use crate::operations::reflect101_index;
use crate::{CoreError, CoreResult};

#[allow(clippy::too_many_arguments)]
pub(crate) fn constant(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    top: usize,
    left: usize,
    _output_height: usize,
    output_width: usize,
    fill: [u8; 3],
    output: &mut [u8],
) {
    let input_row_bytes = input_width * 3;
    let output_row_bytes = output_width * 3;
    let left_bytes = left * 3;
    let right_start = left_bytes + input_row_bytes;

    for row in output[..top * output_row_bytes].chunks_exact_mut(3) {
        row.copy_from_slice(&fill);
    }
    for y in 0..input_height {
        let source = y * input_row_bytes;
        let destination = (top + y) * output_row_bytes;
        let row = &mut output[destination..destination + output_row_bytes];
        for pixel in row[..left_bytes].chunks_exact_mut(3) {
            pixel.copy_from_slice(&fill);
        }
        row[left_bytes..right_start].copy_from_slice(&input[source..source + input_row_bytes]);
        for pixel in row[right_start..].chunks_exact_mut(3) {
            pixel.copy_from_slice(&fill);
        }
    }
    let bottom_start = (top + input_height) * output_row_bytes;
    for pixel in output[bottom_start..].chunks_exact_mut(3) {
        pixel.copy_from_slice(&fill);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reflect101(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    top: usize,
    left: usize,
    output_height: usize,
    output_width: usize,
    output: &mut [u8],
) -> CoreResult<()> {
    let input_row_bytes = input_width * 3;
    let output_row_bytes = output_width * 3;
    let interior_start = left * 3;
    let interior_end = interior_start + input_row_bytes;
    let right = output_width - left - input_width;
    let left_isize = isize::try_from(left)
        .map_err(|_| CoreError::Invalid("padding dimensions overflow".into()))?;
    let input_width_isize = isize::try_from(input_width)
        .map_err(|_| CoreError::Invalid("padding dimensions overflow".into()))?;

    let mut horizontal = Vec::new();
    horizontal
        .try_reserve_exact(left.saturating_add(right))
        .map_err(|_| CoreError::Runtime("padding map allocation failed".into()))?;
    for x in 0..left {
        let coordinate = isize::try_from(x)
            .map_err(|_| CoreError::Invalid("padding dimensions overflow".into()))?
            - left_isize;
        horizontal.push(reflect101_index(coordinate, input_width) * 3);
    }
    for x in 0..right {
        let coordinate = input_width_isize
            .checked_add(
                isize::try_from(x)
                    .map_err(|_| CoreError::Invalid("padding dimensions overflow".into()))?,
            )
            .ok_or_else(|| CoreError::Invalid("padding dimensions overflow".into()))?;
        horizontal.push(reflect101_index(coordinate, input_width) * 3);
    }

    for y in 0..input_height {
        let source = y * input_row_bytes;
        let destination = (top + y) * output_row_bytes;
        let row = &mut output[destination..destination + output_row_bytes];
        row[interior_start..interior_end].copy_from_slice(&input[source..source + input_row_bytes]);
        for (x, &column) in horizontal[..left].iter().enumerate() {
            let source = source + column;
            row[x * 3..x * 3 + 3].copy_from_slice(&input[source..source + 3]);
        }
        for (x, &column) in horizontal[left..].iter().enumerate() {
            let source = source + column;
            let destination = interior_end + x * 3;
            row[destination..destination + 3].copy_from_slice(&input[source..source + 3]);
        }
    }

    let top_isize = isize::try_from(top)
        .map_err(|_| CoreError::Invalid("padding dimensions overflow".into()))?;
    for y in 0..top {
        let coordinate = isize::try_from(y)
            .map_err(|_| CoreError::Invalid("padding dimensions overflow".into()))?
            - top_isize;
        let source_y = top + reflect101_index(coordinate, input_height);
        output.copy_within(
            source_y * output_row_bytes..(source_y + 1) * output_row_bytes,
            y * output_row_bytes,
        );
    }
    for y in top + input_height..output_height {
        let coordinate = isize::try_from(y - top)
            .map_err(|_| CoreError::Invalid("padding dimensions overflow".into()))?;
        let source_y = top + reflect101_index(coordinate, input_height);
        output.copy_within(
            source_y * output_row_bytes..(source_y + 1) * output_row_bytes,
            y * output_row_bytes,
        );
    }
    Ok(())
}
