use crate::{CodecError, CodecErrorKind};

pub(crate) const MAX_OUTPUT_BYTES: usize = 1 << 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    Jpeg,
    Png,
}

impl ImageFormat {
    pub fn detect(encoded: &[u8]) -> Result<Self, CodecError> {
        if encoded.starts_with(&[0xff, 0xd8]) {
            return Ok(Self::Jpeg);
        }
        if encoded.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]) {
            return Ok(Self::Png);
        }
        Err(CodecError::new(
            CodecErrorKind::UnsupportedFormat,
            "input is not a supported JPEG or PNG image",
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorModel {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
    Cmyk,
}

impl ColorModel {
    pub fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba | Self::Cmyk => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeMode {
    Unchanged,
    Gray,
    Rgb,
    Rgba,
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    pub mode: DecodeMode,
    pub max_pixels: Option<usize>,
}

#[derive(Debug)]
pub enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl PixelData {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(data) => data.len(),
            Self::U16(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::U8(data) => data.is_empty(),
            Self::U16(data) => data.is_empty(),
        }
    }
}

#[derive(Debug)]
pub struct DecodedImage {
    pub pixels: PixelData,
    pub height: usize,
    pub width: usize,
    pub color: ColorModel,
    pub source_has_alpha: bool,
}

impl DecodedImage {
    pub(crate) fn convert(self, mode: DecodeMode) -> Result<Self, CodecError> {
        let target = match mode {
            DecodeMode::Unchanged => return Ok(self),
            DecodeMode::Gray => ColorModel::Gray,
            DecodeMode::Rgb => ColorModel::Rgb,
            DecodeMode::Rgba => ColorModel::Rgba,
        };
        if self.color == target {
            return Ok(self);
        }
        let pixels = match self.pixels {
            PixelData::U8(data) => PixelData::U8(convert_u8(&data, self.color, target)?),
            PixelData::U16(data) => PixelData::U16(convert_u16(&data, self.color, target)?),
        };
        Ok(Self {
            pixels,
            height: self.height,
            width: self.width,
            color: target,
            source_has_alpha: self.source_has_alpha,
        })
    }
}

#[derive(Debug)]
pub struct OwnedImage {
    pub pixels: PixelData,
    pub height: usize,
    pub width: usize,
    pub color: ColorModel,
}

impl OwnedImage {
    pub fn validate(&self) -> Result<(), CodecError> {
        ImageView::from(self).validate()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PixelDataRef<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

impl PixelDataRef<'_> {
    fn len(self) -> usize {
        match self {
            Self::U8(data) => data.len(),
            Self::U16(data) => data.len(),
        }
    }

    fn sample_bytes(self) -> usize {
        match self {
            Self::U8(_) => 1,
            Self::U16(_) => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    pub pixels: PixelDataRef<'a>,
    pub height: usize,
    pub width: usize,
    pub color: ColorModel,
}

impl ImageView<'_> {
    pub fn validate(self) -> Result<(), CodecError> {
        if self.height == 0 || self.width == 0 {
            return Err(CodecError::new(
                CodecErrorKind::InvalidInput,
                "image dimensions must be positive",
            ));
        }
        let expected = checked_samples(self.width, self.height, self.color.channels())?;
        if self.pixels.len() != expected {
            return Err(CodecError::new(
                CodecErrorKind::InvalidInput,
                "pixel buffer length does not match image shape",
            ));
        }
        let bytes = expected
            .checked_mul(self.pixels.sample_bytes())
            .ok_or_else(output_too_large)?;
        if bytes > MAX_OUTPUT_BYTES {
            return Err(output_too_large());
        }
        Ok(())
    }
}

impl<'a> From<&'a OwnedImage> for ImageView<'a> {
    fn from(image: &'a OwnedImage) -> Self {
        Self {
            pixels: match &image.pixels {
                PixelData::U8(data) => PixelDataRef::U8(data),
                PixelData::U16(data) => PixelDataRef::U16(data),
            },
            height: image.height,
            width: image.width,
            color: image.color,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum EncodeOptions {
    Jpeg { quality: u8 },
    Png { compression: u8 },
}

pub(crate) fn validate_dimensions(
    width: usize,
    height: usize,
    channels: usize,
    sample_bytes: usize,
    max_pixels: Option<usize>,
    role: &str,
) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::new(
            CodecErrorKind::Decode,
            format!("{role} dimensions must be positive"),
        ));
    }
    let pixels = width.checked_mul(height).ok_or_else(output_too_large)?;
    if max_pixels.is_some_and(|limit| pixels > limit) {
        return Err(CodecError::new(
            CodecErrorKind::LimitExceeded,
            format!("{role} has {pixels} pixels, exceeding the configured limit"),
        ));
    }
    let bytes = pixels
        .checked_mul(channels)
        .and_then(|samples| samples.checked_mul(sample_bytes))
        .ok_or_else(output_too_large)?;
    if bytes > MAX_OUTPUT_BYTES {
        return Err(output_too_large());
    }
    Ok(())
}

fn checked_samples(width: usize, height: usize, channels: usize) -> Result<usize, CodecError> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(output_too_large)
}

fn output_too_large() -> CodecError {
    CodecError::new(
        CodecErrorKind::LimitExceeded,
        "decoded image exceeds the supported memory limit",
    )
}

fn convert_u8(
    source: &[u8],
    source_color: ColorModel,
    target: ColorModel,
) -> Result<Vec<u8>, CodecError> {
    convert_samples(source, source_color, target, 255, |r, g, b| {
        ((299 * u32::from(r) + 587 * u32::from(g) + 114 * u32::from(b) + 500) / 1000) as u8
    })
}

fn convert_u16(
    source: &[u16],
    source_color: ColorModel,
    target: ColorModel,
) -> Result<Vec<u16>, CodecError> {
    if source_color == ColorModel::Cmyk {
        return Err(CodecError::new(
            CodecErrorKind::UnsupportedFormat,
            "16-bit CMYK conversion is unsupported",
        ));
    }
    convert_samples(source, source_color, target, 65_535, |r, g, b| {
        ((299 * u32::from(r) + 587 * u32::from(g) + 114 * u32::from(b) + 500) / 1000) as u16
    })
}

fn convert_samples<T, F>(
    source: &[T],
    source_color: ColorModel,
    target: ColorModel,
    max: T,
    to_gray: F,
) -> Result<Vec<T>, CodecError>
where
    T: Copy + Into<u32> + TryFrom<u32>,
    F: Fn(T, T, T) -> T,
{
    let source_channels = source_color.channels();
    let target_channels = target.channels();
    let mut output = Vec::with_capacity(source.len() / source_channels * target_channels);
    for pixel in source.chunks_exact(source_channels) {
        let (r, g, b, alpha) = match source_color {
            ColorModel::Gray => (pixel[0], pixel[0], pixel[0], max),
            ColorModel::GrayAlpha => (pixel[0], pixel[0], pixel[0], pixel[1]),
            ColorModel::Rgb => (pixel[0], pixel[1], pixel[2], max),
            ColorModel::Rgba => (pixel[0], pixel[1], pixel[2], pixel[3]),
            ColorModel::Cmyk => {
                let maximum = max.into();
                let c = pixel[0].into();
                let m = pixel[1].into();
                let y = pixel[2].into();
                let k = pixel[3].into();
                (
                    cast_sample(((maximum - c) * (maximum - k) + maximum / 2) / maximum)?,
                    cast_sample(((maximum - m) * (maximum - k) + maximum / 2) / maximum)?,
                    cast_sample(((maximum - y) * (maximum - k) + maximum / 2) / maximum)?,
                    max,
                )
            }
        };
        match target {
            ColorModel::Gray => output.push(to_gray(r, g, b)),
            ColorModel::Rgb => output.extend_from_slice(&[r, g, b]),
            ColorModel::Rgba => output.extend_from_slice(&[r, g, b, alpha]),
            ColorModel::GrayAlpha | ColorModel::Cmyk => {
                return Err(CodecError::new(
                    CodecErrorKind::UnsupportedFormat,
                    "requested color conversion is unsupported",
                ));
            }
        }
    }
    Ok(output)
}

fn cast_sample<T: TryFrom<u32>>(value: u32) -> Result<T, CodecError> {
    T::try_from(value).map_err(|_| {
        CodecError::new(
            CodecErrorKind::Decode,
            "color conversion produced an invalid sample",
        )
    })
}
