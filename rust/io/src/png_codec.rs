use crate::model::{validate_dimensions, MAX_OUTPUT_BYTES};
use crate::{
    CodecError, CodecErrorKind, ColorModel, DecodeMode, DecodeOptions, DecodedImage, ImageView,
    PixelData, PixelDataRef,
};
use std::io::Cursor;

pub(crate) fn decode(encoded: &[u8], options: DecodeOptions) -> Result<DecodedImage, CodecError> {
    let mut decoder = png::Decoder::new(Cursor::new(encoded));
    decoder.set_transformations(if options.mode == DecodeMode::Unchanged {
        png::Transformations::IDENTITY
    } else {
        png::Transformations::EXPAND
    });
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let mut reader = decoder.read_info().map_err(map_decode_error)?;
    if reader.info().animation_control.is_some() {
        return Err(CodecError::new(
            CodecErrorKind::UnsupportedFormat,
            "animated PNG is not supported",
        ));
    }
    let stored_color = reader.info().color_type;
    let stored_depth = reader.info().bit_depth;
    let source_has_alpha = matches!(
        stored_color,
        png::ColorType::GrayscaleAlpha | png::ColorType::Rgba
    ) || reader.info().trns.is_some();
    let width = reader.info().width as usize;
    let height = reader.info().height as usize;
    let preserves_label_samples = options.mode == DecodeMode::Unchanged
        && matches!(
            stored_color,
            png::ColorType::Grayscale | png::ColorType::Indexed
        )
        && stored_depth != png::BitDepth::Sixteen;
    if preserves_label_samples {
        validate_dimensions(width, height, 1, 1, options.max_pixels, "image")?;
        let buffer_size = reader
            .output_buffer_size()
            .ok_or_else(|| decode_error("PNG dimensions overflow"))?;
        if buffer_size > MAX_OUTPUT_BYTES {
            return Err(CodecError::new(
                CodecErrorKind::LimitExceeded,
                "decoded image exceeds the supported memory limit",
            ));
        }
        let mut packed = vec![0; buffer_size];
        let output = reader.next_frame(&mut packed).map_err(map_decode_error)?;
        packed.truncate(output.buffer_size());
        return Ok(DecodedImage {
            pixels: PixelData::U8(unpack_samples(
                &packed,
                width,
                height,
                stored_depth as usize,
            )?),
            height,
            width,
            color: ColorModel::Gray,
            source_has_alpha,
        });
    }
    let (output_color, output_depth) = reader.output_color_type();
    let color = map_color(output_color)?;
    let sample_bytes = match output_depth {
        png::BitDepth::Eight => 1,
        png::BitDepth::Sixteen => 2,
        _ => {
            return Err(CodecError::new(
                CodecErrorKind::Decode,
                "PNG decoder did not expand packed samples",
            ));
        }
    };
    validate_dimensions(
        width,
        height,
        color.channels(),
        sample_bytes,
        options.max_pixels,
        "image",
    )?;
    let buffer_size = reader
        .output_buffer_size()
        .ok_or_else(|| decode_error("PNG dimensions overflow"))?;
    if buffer_size > MAX_OUTPUT_BYTES {
        return Err(CodecError::new(
            CodecErrorKind::LimitExceeded,
            "decoded image exceeds the supported memory limit",
        ));
    }
    let mut bytes = vec![0; buffer_size];
    let output = reader.next_frame(&mut bytes).map_err(map_decode_error)?;
    bytes.truncate(output.buffer_size());
    let pixels = match output.bit_depth {
        png::BitDepth::Eight => PixelData::U8(bytes),
        png::BitDepth::Sixteen => PixelData::U16(
            bytes
                .chunks_exact(2)
                .map(|sample| u16::from_be_bytes([sample[0], sample[1]]))
                .collect(),
        ),
        _ => {
            return Err(CodecError::new(
                CodecErrorKind::Decode,
                "PNG decoder returned an unsupported bit depth",
            ));
        }
    };
    DecodedImage {
        pixels,
        height,
        width,
        color,
        source_has_alpha,
    }
    .convert(options.mode)
}

fn unpack_samples(
    packed: &[u8],
    width: usize,
    height: usize,
    bits: usize,
) -> Result<Vec<u8>, CodecError> {
    let samples_per_byte = 8 / bits;
    let row_bytes = width
        .checked_add(samples_per_byte - 1)
        .map(|value| value / samples_per_byte)
        .ok_or_else(|| decode_error("PNG row size overflow"))?;
    let packed_len = row_bytes
        .checked_mul(height)
        .ok_or_else(|| decode_error("PNG buffer size overflow"))?;
    if packed.len() != packed_len {
        return Err(decode_error("decoded PNG buffer has an invalid length"));
    }
    if bits == 8 {
        return Ok(packed.to_vec());
    }
    let output_len = width
        .checked_mul(height)
        .ok_or_else(|| decode_error("PNG dimensions overflow"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| CodecError::new(CodecErrorKind::LimitExceeded, "image allocation failed"))?;
    let sample_mask = (1_u8 << bits) - 1;
    for row in packed.chunks_exact(row_bytes) {
        for x in 0..width {
            let shift = 8 - bits - (x % samples_per_byte) * bits;
            output.push((row[x / samples_per_byte] >> shift) & sample_mask);
        }
    }
    Ok(output)
}

pub(crate) fn encode(image: ImageView<'_>, compression: u8) -> Result<Vec<u8>, CodecError> {
    let color = match image.color {
        ColorModel::Gray => png::ColorType::Grayscale,
        ColorModel::GrayAlpha => png::ColorType::GrayscaleAlpha,
        ColorModel::Rgb => png::ColorType::Rgb,
        ColorModel::Rgba => png::ColorType::Rgba,
        ColorModel::Cmyk => {
            return Err(CodecError::new(
                CodecErrorKind::InvalidInput,
                "PNG does not support CMYK pixels",
            ));
        }
    };
    match image.pixels {
        PixelDataRef::U8(data) => encode_bytes(
            data,
            image.width,
            image.height,
            color,
            png::BitDepth::Eight,
            compression,
        ),
        PixelDataRef::U16(data) => {
            let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_be_bytes()).collect();
            encode_bytes(
                &bytes,
                image.width,
                image.height,
                color,
                png::BitDepth::Sixteen,
                compression,
            )
        }
    }
}

fn encode_bytes(
    data: &[u8],
    width: usize,
    height: usize,
    color: png::ColorType,
    depth: png::BitDepth,
    compression: u8,
) -> Result<Vec<u8>, CodecError> {
    if compression > 9 {
        return Err(CodecError::new(
            CodecErrorKind::InvalidInput,
            "PNG compression must be between 0 and 9",
        ));
    }
    let width = u32::try_from(width).map_err(|_| encode_error("PNG width exceeds u32"))?;
    let height = u32::try_from(height).map_err(|_| encode_error("PNG height exceeds u32"))?;
    let mut output = Vec::new();
    let mut encoder = png::Encoder::new(&mut output, width, height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    encoder.set_deflate_compression(png::DeflateCompression::Level(compression));
    let mut writer = encoder.write_header().map_err(map_encode_error)?;
    writer.write_image_data(data).map_err(map_encode_error)?;
    writer.finish().map_err(map_encode_error)?;
    Ok(output)
}

fn map_color(color: png::ColorType) -> Result<ColorModel, CodecError> {
    match color {
        png::ColorType::Grayscale => Ok(ColorModel::Gray),
        png::ColorType::GrayscaleAlpha => Ok(ColorModel::GrayAlpha),
        png::ColorType::Rgb => Ok(ColorModel::Rgb),
        png::ColorType::Rgba => Ok(ColorModel::Rgba),
        png::ColorType::Indexed => Err(CodecError::new(
            CodecErrorKind::Decode,
            "PNG decoder did not expand the color palette",
        )),
    }
}

fn map_decode_error(error: png::DecodingError) -> CodecError {
    decode_error(error.to_string())
}

fn map_encode_error(error: png::EncodingError) -> CodecError {
    encode_error(error.to_string())
}

fn decode_error(message: impl Into<String>) -> CodecError {
    CodecError::new(CodecErrorKind::Decode, message)
}

fn encode_error(message: impl Into<String>) -> CodecError {
    CodecError::new(CodecErrorKind::Encode, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed_png(
        color: png::ColorType,
        depth: png::BitDepth,
        bits: usize,
        samples: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        let row_bytes = (width * bits).div_ceil(8);
        let mut packed = vec![0_u8; row_bytes * height];
        for y in 0..height {
            for x in 0..width {
                let bit = x * bits;
                let shift = 8 - bits - bit % 8;
                packed[y * row_bytes + bit / 8] |= samples[y * width + x] << shift;
            }
        }
        let mut encoded = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut encoded, width as u32, height as u32);
            encoder.set_color(color);
            encoder.set_depth(depth);
            if color == png::ColorType::Indexed {
                let entries = 1_usize << bits;
                encoder.set_palette(vec![0_u8; entries * 3]);
            }
            let mut writer = encoder.write_header().expect("valid PNG header");
            writer
                .write_image_data(&packed)
                .expect("valid packed pixels");
        }
        encoded
    }

    #[test]
    fn unchanged_preserves_packed_grayscale_and_indexed_samples() {
        let width = 7;
        let height = 3;
        for color in [png::ColorType::Grayscale, png::ColorType::Indexed] {
            for (depth, bits) in [
                (png::BitDepth::One, 1),
                (png::BitDepth::Two, 2),
                (png::BitDepth::Four, 4),
                (png::BitDepth::Eight, 8),
            ] {
                let max_sample = (1_u16 << bits) - 1;
                let samples = (0..width * height)
                    .map(|index| ((index * 5 + index / 3) as u16 % (max_sample + 1)) as u8)
                    .collect::<Vec<_>>();
                let encoded = packed_png(color, depth, bits, &samples, width, height);
                let decoded = decode(
                    &encoded,
                    DecodeOptions {
                        mode: DecodeMode::Unchanged,
                        max_pixels: None,
                    },
                )
                .expect("packed PNG should decode");
                assert_eq!((decoded.height, decoded.width), (height, width));
                assert_eq!(decoded.color, ColorModel::Gray);
                match decoded.pixels {
                    PixelData::U8(actual) => assert_eq!(actual, samples),
                    PixelData::U16(_) => panic!("packed samples must decode as uint8"),
                }
            }
        }
    }
}
