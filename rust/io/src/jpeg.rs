use crate::model::validate_dimensions;
use crate::{
    CodecError, CodecErrorKind, ColorModel, DecodeOptions, DecodedImage, ImageView, PixelData,
    PixelDataRef,
};
use std::cell::RefCell;

const MAX_PROGRESSIVE_SCANS: u32 = 100;

thread_local! {
    static DECODER: RefCell<Option<turbojpeg::Decompressor>> = const { RefCell::new(None) };
    static ENCODER: RefCell<Option<turbojpeg::Compressor>> = const { RefCell::new(None) };
}

pub(crate) fn decode(encoded: &[u8], options: DecodeOptions) -> Result<DecodedImage, CodecError> {
    DECODER.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| decode_error("JPEG decoder is already in use"))?;
        if slot.is_none() {
            *slot = Some(turbojpeg::Decompressor::new().map_err(map_decode_error)?);
        }
        let Some(decoder) = slot.as_mut() else {
            return Err(decode_error("JPEG decoder initialization failed"));
        };
        decoder
            .set_scan_limit(MAX_PROGRESSIVE_SCANS)
            .map_err(map_decode_error)?;
        let header = decoder.read_header(encoded).map_err(map_decode_error)?;
        let (color, format) = match header.colorspace {
            turbojpeg::Colorspace::Gray => (ColorModel::Gray, turbojpeg::PixelFormat::GRAY),
            turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK => {
                (ColorModel::Cmyk, turbojpeg::PixelFormat::CMYK)
            }
            _ => (ColorModel::Rgb, turbojpeg::PixelFormat::RGB),
        };
        validate_dimensions(
            header.width,
            header.height,
            color.channels(),
            1,
            options.max_pixels,
            "image",
        )?;
        let len = header
            .width
            .checked_mul(header.height)
            .and_then(|pixels| pixels.checked_mul(color.channels()))
            .ok_or_else(|| decode_error("JPEG dimensions overflow"))?;
        let mut pixels = vec![0; len];
        decoder
            .decompress(
                encoded,
                turbojpeg::Image {
                    pixels: pixels.as_mut_slice(),
                    width: header.width,
                    pitch: header.width * color.channels(),
                    height: header.height,
                    format,
                },
            )
            .map_err(map_decode_error)?;
        if color == ColorModel::Cmyk {
            for sample in &mut pixels {
                *sample = 255 - *sample;
            }
        }
        DecodedImage {
            pixels: PixelData::U8(pixels),
            height: header.height,
            width: header.width,
            color,
            source_has_alpha: false,
        }
        .convert(options.mode)
    })
}

pub(crate) fn encode(image: ImageView<'_>, quality: u8) -> Result<Vec<u8>, CodecError> {
    if !(1..=100).contains(&quality) {
        return Err(CodecError::new(
            CodecErrorKind::InvalidInput,
            "JPEG quality must be between 1 and 100",
        ));
    }
    let (pixels, format, subsampling) = match (image.pixels, image.color) {
        (PixelDataRef::U8(data), ColorModel::Gray) => {
            (data, turbojpeg::PixelFormat::GRAY, turbojpeg::Subsamp::Gray)
        }
        (PixelDataRef::U8(data), ColorModel::Rgb) => (
            data,
            turbojpeg::PixelFormat::RGB,
            turbojpeg::Subsamp::Sub2x2,
        ),
        (PixelDataRef::U8(_), ColorModel::GrayAlpha | ColorModel::Rgba) => {
            return Err(CodecError::new(
                CodecErrorKind::InvalidInput,
                "JPEG does not support alpha channels",
            ));
        }
        (PixelDataRef::U8(_), ColorModel::Cmyk) => {
            return Err(CodecError::new(
                CodecErrorKind::UnsupportedFormat,
                "CMYK JPEG encoding is not supported",
            ));
        }
        (PixelDataRef::U16(_), _) => {
            return Err(CodecError::new(
                CodecErrorKind::InvalidInput,
                "JPEG encoding requires uint8 pixels",
            ));
        }
    };
    ENCODER.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|_| encode_error("JPEG encoder is already in use"))?;
        if slot.is_none() {
            *slot = Some(turbojpeg::Compressor::new().map_err(map_encode_error)?);
        }
        let Some(encoder) = slot.as_mut() else {
            return Err(encode_error("JPEG encoder initialization failed"));
        };
        encoder
            .set_quality(i32::from(quality))
            .map_err(map_encode_error)?;
        encoder.set_subsamp(subsampling).map_err(map_encode_error)?;
        encoder.set_progressive(false).map_err(map_encode_error)?;
        encoder.set_optimize(false).map_err(map_encode_error)?;
        encoder
            .compress_to_vec(turbojpeg::Image {
                pixels,
                width: image.width,
                pitch: image.width * image.color.channels(),
                height: image.height,
                format,
            })
            .map_err(map_encode_error)
    })
}

fn map_decode_error(error: turbojpeg::Error) -> CodecError {
    decode_error(error.to_string())
}

fn map_encode_error(error: turbojpeg::Error) -> CodecError {
    encode_error(error.to_string())
}

fn decode_error(message: impl Into<String>) -> CodecError {
    CodecError::new(CodecErrorKind::Decode, message)
}

fn encode_error(message: impl Into<String>) -> CodecError {
    CodecError::new(CodecErrorKind::Encode, message)
}
