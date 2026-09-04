# Image I/O

Variopinta exposes standalone JPEG and PNG helpers. Pipeline `Encoded` and
`Path` carriers use the same native codecs and resource-limit model; pipeline
`Encode` and `Write` outputs use the same format options.

## Read and decode

```text
vp.read_image(
    path,
    *,
    mode="rgb",
    max_pixels=100_000_000,
    max_encoded_bytes=None,
) -> numpy.ndarray

vp.decode_image(
    data,
    *,
    mode="rgb",
    max_pixels=100_000_000,
    max_encoded_bytes=None,
) -> numpy.ndarray
```

`read_image` accepts a local `str` or `os.PathLike[str]`. `decode_image`
accepts `bytes`, `bytearray`, or `memoryview`. Both detect JPEG or static PNG
from the contents and return an owned, C-contiguous NumPy array.

Modes are:

| Mode | Result |
|---|---|
| `"unchanged"` | native decoded channels and sample dtype |
| `"gray"` | HW grayscale |
| `"rgb"` | HWC RGB |
| `"rgba"` | HWC RGBA |

For PNG in `"unchanged"` mode, packed 1-, 2-, 4-, or 8-bit grayscale samples
and palette indices are unpacked into `uint8` without rescaling, palette
expansion, or transparency application. A 16-bit PNG remains `uint16` where
the selected mode supports it.

```python
from pathlib import Path

import variopinta as vp

image = vp.read_image("input.png", mode="rgb")
same_image = vp.decode_image(Path("input.png").read_bytes(), mode="rgb")
```

Format detection does not trust a filename suffix. Paths are local only; URLs,
globs, and `bytes` paths are rejected.

## Encode and write

```text
vp.encode_image(
    image,
    *,
    format,
    quality=None,
    compression=None,
) -> bytes

vp.write_image(
    path,
    image,
    *,
    format=None,
    quality=None,
    compression=None,
) -> None
```

Supported output arrays and options are:

| Format | Array | Option |
|---|---|---|
| JPEG | HW grayscale or HWC RGB `uint8` | `quality=1..100`, default 95 |
| PNG | HW or HWC with 1–4 channels, `uint8` or `uint16` | `compression=0..9`, default 6 |

`format` accepts `"jpeg"`, `"jpg"`, `"png"`, and their dotted forms.
`write_image(format=None)` infers JPEG or PNG from the destination suffix. If a
recognized suffix and explicit format are both present, they must agree.

```python
import numpy as np
import variopinta as vp

image = np.zeros((32, 48, 3), dtype=np.uint8)
payload = vp.encode_image(image, format="png", compression=3)
decoded = vp.decode_image(payload, mode="rgb")

assert decoded.shape == image.shape
assert decoded.dtype == np.uint8

vp.write_image("output.jpg", image, quality=90)
```

Input arrays may be non-contiguous; the boundary normalizes them without
mutating the caller's data.

## Resource limits

`max_pixels` rejects decoded dimensions whose product exceeds the limit.
`max_encoded_bytes` rejects oversized encoded input before decoding. Both must
be positive integers or `None`; `None` disables that limit. The default
pixel limit is 100 million and the encoded-byte limit is disabled by default.

Set limits from the trust boundary of the application rather than from image
metadata:

```python
image = vp.decode_image(
    payload,
    max_pixels=20_000_000,
    max_encoded_bytes=8 * 1024 * 1024,
)
```

The same options are available on `Encoded(...)` and `Path(...)` pipeline
carriers.

## Deliberate omissions

The codec API does not apply EXIF orientation, preserve metadata, or decode
animated PNG. JPEG decoding produces 8-bit samples. PNG behavior is limited to
the channel and dtype contracts above.

Pipeline mask acquisition is intentionally narrower than generic image I/O:
it accepts only static, non-transparent grayscale or indexed PNG with 1-, 2-,
4-, or 8-bit samples. Pipeline mask encoding and writing always produce
lossless 8-bit grayscale PNG. See [Pipelines and targets](pipelines-and-targets.md#carriers).
