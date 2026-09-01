use crate::model::{validate_dimensions, MAX_OUTPUT_BYTES};
use crate::{
    CodecError, CodecErrorKind, ColorModel, DecodeOptions, DecodedImage, OwnedImage, PixelData,
};
use std::io::Cursor;

pub(crate) fn decode(encoded: &[u8], options: DecodeOptions) -> Result<DecodedImage, CodecError> {
    let mut decoder = png::Decoder::new(Cursor::new(encoded));
    decoder.set_transformations(png::Transformations::EXPAND);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let mut reader = decoder.read_info().map_err(map_decode_error)?;
    if reader.info().animation_control.is_some() {
        return Err(CodecError::new(
            CodecErrorKind::UnsupportedFormat,
            "animated PNG is not supported",
        ));
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
    let width = reader.info().width as usize;
    let height = reader.info().height as usize;
    validate_dimensions(
        width,
        height,
        color.channels(),
        sample_bytes,
        options.max_pixels,
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
    }
    .convert(options.mode)
}

pub(crate) fn encode(image: &OwnedImage, compression: u8) -> Result<Vec<u8>, CodecError> {
    if compression > 9 {
        return Err(CodecError::new(
            CodecErrorKind::InvalidInput,
            "PNG compression must be between 0 and 9",
        ));
    }
    let width = u32::try_from(image.width).map_err(|_| encode_error("PNG width exceeds u32"))?;
    let height = u32::try_from(image.height).map_err(|_| encode_error("PNG height exceeds u32"))?;
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
    let depth = match image.pixels {
        PixelData::U8(_) => png::BitDepth::Eight,
        PixelData::U16(_) => png::BitDepth::Sixteen,
    };
    let mut output = Vec::new();
    let mut encoder = png::Encoder::new(&mut output, width, height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    encoder.set_deflate_compression(png::DeflateCompression::Level(compression));
    let mut writer = encoder.write_header().map_err(map_encode_error)?;
    match &image.pixels {
        PixelData::U8(data) => writer.write_image_data(data).map_err(map_encode_error)?,
        PixelData::U16(data) => {
            let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_be_bytes()).collect();
            writer.write_image_data(&bytes).map_err(map_encode_error)?;
        }
    }
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
