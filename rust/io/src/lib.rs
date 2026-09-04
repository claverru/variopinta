mod error;
mod jpeg;
mod model;
mod png_codec;

pub use error::{CodecError, CodecErrorKind};
pub use model::{
    ColorModel, DecodeMode, DecodeOptions, DecodedImage, EncodeOptions, ImageFormat, ImageView,
    OwnedImage, PixelData, PixelDataRef,
};

pub fn decode_image(encoded: &[u8], options: DecodeOptions) -> Result<DecodedImage, CodecError> {
    match ImageFormat::detect(encoded)? {
        ImageFormat::Jpeg => jpeg::decode(encoded, options),
        ImageFormat::Png => png_codec::decode(encoded, options),
    }
}

pub fn encode_image(image: &OwnedImage, options: EncodeOptions) -> Result<Vec<u8>, CodecError> {
    encode_image_view(image.into(), options)
}

pub fn encode_image_view(
    image: ImageView<'_>,
    options: EncodeOptions,
) -> Result<Vec<u8>, CodecError> {
    image.validate()?;
    match options {
        EncodeOptions::Jpeg { quality } => jpeg::encode(image, quality),
        EncodeOptions::Png { compression } => png_codec::encode(image, compression),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_options(mode: DecodeMode) -> DecodeOptions {
        DecodeOptions {
            mode,
            max_pixels: Some(1_000),
        }
    }

    #[test]
    fn png_round_trips_u8_color_models() {
        for color in [
            ColorModel::Gray,
            ColorModel::GrayAlpha,
            ColorModel::Rgb,
            ColorModel::Rgba,
        ] {
            let len = 3 * 5 * color.channels();
            let pixels: Vec<u8> = (0..len).map(|value| (value * 37 % 256) as u8).collect();
            let image = OwnedImage {
                pixels: PixelData::U8(pixels.clone()),
                height: 3,
                width: 5,
                color,
            };
            let encoded = encode_image(&image, EncodeOptions::Png { compression: 3 }).unwrap();
            let decoded = decode_image(&encoded, decode_options(DecodeMode::Unchanged)).unwrap();
            assert_eq!(decoded.color, color);
            assert_eq!((decoded.height, decoded.width), (3, 5));
            match decoded.pixels {
                PixelData::U8(actual) => assert_eq!(actual, pixels),
                PixelData::U16(_) => panic!("expected uint8 pixels"),
            }
        }
    }

    #[test]
    fn png_round_trips_u16_color_models() {
        for color in [
            ColorModel::Gray,
            ColorModel::GrayAlpha,
            ColorModel::Rgb,
            ColorModel::Rgba,
        ] {
            let len = 3 * 5 * color.channels();
            let pixels: Vec<u16> = (0..len)
                .map(|value| (value as u16).wrapping_mul(4_099))
                .collect();
            let image = OwnedImage {
                pixels: PixelData::U16(pixels.clone()),
                height: 3,
                width: 5,
                color,
            };
            let encoded = encode_image(&image, EncodeOptions::Png { compression: 6 }).unwrap();
            let decoded = decode_image(&encoded, decode_options(DecodeMode::Unchanged)).unwrap();
            assert_eq!(decoded.color, color);
            match decoded.pixels {
                PixelData::U16(actual) => assert_eq!(actual, pixels),
                PixelData::U8(_) => panic!("expected uint16 pixels"),
            }
        }
    }

    #[test]
    fn decode_converts_color_modes() {
        let image = OwnedImage {
            pixels: PixelData::U8(vec![10, 20, 30, 40, 50, 60, 70, 80]),
            height: 1,
            width: 2,
            color: ColorModel::Rgba,
        };
        let encoded = encode_image(&image, EncodeOptions::Png { compression: 0 }).unwrap();
        let rgb = decode_image(&encoded, decode_options(DecodeMode::Rgb)).unwrap();
        assert_eq!(rgb.color, ColorModel::Rgb);
        assert!(matches!(rgb.pixels, PixelData::U8(ref data) if data == &[10, 20, 30, 50, 60, 70]));
        let gray = decode_image(&encoded, decode_options(DecodeMode::Gray)).unwrap();
        assert_eq!(gray.color, ColorModel::Gray);
        assert_eq!(gray.pixels.len(), 2);
    }

    #[test]
    fn decode_enforces_pixel_limit() {
        let image = OwnedImage {
            pixels: PixelData::U8(vec![0; 4 * 5 * 3]),
            height: 4,
            width: 5,
            color: ColorModel::Rgb,
        };
        let encoded = encode_image(&image, EncodeOptions::Png { compression: 6 }).unwrap();
        let error = decode_image(
            &encoded,
            DecodeOptions {
                mode: DecodeMode::Rgb,
                max_pixels: Some(19),
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), CodecErrorKind::LimitExceeded);
    }

    #[test]
    fn jpeg_encodes_rgb_and_gray() {
        for color in [ColorModel::Gray, ColorModel::Rgb] {
            let image = OwnedImage {
                pixels: PixelData::U8(vec![127; 7 * 11 * color.channels()]),
                height: 7,
                width: 11,
                color,
            };
            let encoded = encode_image(&image, EncodeOptions::Jpeg { quality: 95 }).unwrap();
            let decoded = decode_image(&encoded, decode_options(DecodeMode::Unchanged)).unwrap();
            assert_eq!(decoded.color, color);
            assert_eq!((decoded.height, decoded.width), (7, 11));
        }
    }
}
