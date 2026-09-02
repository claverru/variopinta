# Pipelines and image I/O

## Compose and compilation

`Compose(transforms, seed=None, *, input=ArrayInput(), output=ReturnOutput())`
constructs the semantic reference pipeline. Calling `.compile()` returns an
immutable executor that may elide copies, reuse buffers, select specialized
kernels, or apply observable-equivalent fusion.

```python
import variopinta as vp

reference = vp.Compose(
    [vp.RandomCrop(256, 256), vp.Resize(224, 224), vp.Normalize()],
    seed=42,
)
compiled = reference.compile()

expected = reference(image, key=7)
actual = compiled(image, key=7)
```

Use an unsigned 64-bit `key` when output must be independent of call order or
worker assignment. Without a key, each pipeline advances its seeded sequence.
Determinism applies to the same installed build, environment, pipeline, input,
seed, and key; exact pixels and random streams may change across releases,
builds, or platforms.

`explain()` reports the executable plan without reading an input. It includes
operations, passes, copies, buffers, dtype and layout transitions, selected
kernel forms, fallbacks, source and sink policy, and GIL state.

## Array contract

Array input is a positive-size HWC RGB NumPy array with dtype `uint8`.
Non-contiguous input is normalized at the Python boundary. Returned arrays are
owned and contiguous: normally HWC `uint8`, or HWC `float32` after an applied
`Normalize`. Terminal `ToTorch` returns a contiguous CHW CPU tensor.

## Standalone image I/O

```python
import variopinta as vp

image = vp.read_image("input.jpg")
encoded = vp.encode_image(image, format="jpeg", quality=90)
decoded = vp.decode_image(encoded, mode="rgb")
vp.write_image("output.png", decoded, compression=6)
```

`read_image` and `decode_image` accept JPEG or static PNG. Decode modes are
`unchanged`, `gray`, `rgb`, and `rgba`. Format detection uses the contents, not
the filename. EXIF orientation, metadata preservation, and animated PNG are not
supported.

`encode_image` and `write_image` support:

- JPEG `uint8` grayscale or RGB, with quality 1–100 and default 95;
- PNG with one to four `uint8` or `uint16` channels, with compression 0–9 and
  default 6.

`read_image` and `decode_image` default to a 100,000,000-pixel decoded-image
limit. `max_encoded_bytes` can reject a compressed input before snapshotting a
mutable buffer or reading a complete file. Set either limit to `None` only when
the caller provides an equivalent resource boundary.

## Native sources and sinks

Source and sink policy is fixed when `Compose` is constructed. The three inputs
and three outputs form nine explicit routes:

| Configuration | Call value or result |
|---|---|
| `ArrayInput()` | NumPy HWC RGB `uint8` |
| `EncodedInput(...)` | complete JPEG or static PNG in `bytes`, `bytearray`, or `memoryview` |
| `PathInput(...)` | local `str` or `os.PathLike[str]` |
| `ReturnOutput()` | owned NumPy array or terminal Torch tensor |
| `EncodedOutput(...)` | encoded Python `bytes` |
| `PathOutput(...)` | writes `destination` and returns `None` |

An encoded service route needs no intermediate Python array:

```python
pipeline = vp.Compose(
    [vp.Resize(224, 224)],
    input=vp.EncodedInput(max_encoded_bytes=32 * 1024 * 1024),
    output=vp.EncodedOutput(format="png", compression=6),
).compile()

response_bytes = pipeline(request_bytes, key=7)
```

A local path route can read, augment, and write in one call:

```python
from pathlib import Path

import variopinta as vp

pipeline = vp.Compose(
    [vp.RandomCrop(256, 256), vp.Resize(224, 224)],
    seed=42,
    input=vp.PathInput(max_encoded_bytes=32 * 1024 * 1024),
    output=vp.PathOutput(format="jpeg", quality=90),
).compile()

pipeline(Path("input.png"), destination=Path("output.jpg"), key=7)
```

Encoded and path inputs are decoded to RGB. Mutable encoded carriers are
snapshotted at call entry. Encoded sinks require a statically HWC RGB `uint8`
route, so executable `Normalize` and every `ToTorch` configuration are rejected
when the pipeline is constructed.

`PathOutput` requires `destination`. A recognized JPEG or PNG suffix must agree
with the configured format; extensionless and unrelated suffixes are allowed.
Parent directories must exist. Existing destinations are replaced directly,
writes are not atomic, and concurrent writes to one path are not coordinated.
The source is completely read, decoded, and augmented before the destination is
opened, including when both paths are equal.

Owned encoded and path routes release the GIL during read, decode,
augmentation, encode, and write. Array-backed augmentation retains the GIL
while borrowing NumPy input. Returning Python `bytes` requires a final copy.
