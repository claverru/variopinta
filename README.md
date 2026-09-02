# Variopinta

Variopinta is an experimental CPU image-augmentation compiler with a Python
configuration API and a Rust execution core. It compiles complete pipelines to
reduce Python/native crossings, reuse buffers, select optimized kernels, and
report the resulting execution plan.

> *Variopinta* is the feminine form of the Spanish *variopinto*: “varied in
> color or appearance,” from Italian *variopinto*, “varied” and “painted.”
> — [RAE](https://dle.rae.es/variopinto)

Variopinta is image-only and experimental. The public Python API may change
between `0.y.0` releases; patch releases preserve documented signatures and
data contracts unless a correctness or security fix requires otherwise.

## Installation

Version 0.2 supports CPython 3.10–3.13 on 64-bit x86 Linux with glibc 2.34 or
newer and on macOS 11 or newer running natively on Apple Silicon. AVX2 is
detected at runtime on x86-64 and is not required. Other Python implementations,
operating systems, architectures, and 32-bit environments are not supported.

Install Variopinta from PyPI:

```bash
python -m pip install variopinta
```

To build from a source checkout, install Rust 1.87 or newer and CMake. Linux
x86-64 additionally needs a C/C++ toolchain and NASM:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake nasm
python -m pip install .
```

On Apple Silicon, install Xcode command-line tools and CMake; NASM is not
required. Source and wheel builds compile the locked vendored libjpeg-turbo
statically and do not use Homebrew or MacPorts codec libraries.

The build uses Maturin through Python build isolation. NumPy is installed as
the only required runtime dependency.

`ToTorch` is optional and requires a PyTorch build compatible with the selected
Python and platform:

```bash
python -m pip install torch
```

## Quick start

```python
import numpy as np
import variopinta as vp

pipeline = vp.Compose(
    [
        vp.RandomCrop(256, 256),
        vp.Resize(224, 224),
        vp.HorizontalFlip(p=0.5),
        vp.Normalize(),
    ],
    seed=42,
).compile()

image = np.zeros((320, 320, 3), dtype=np.uint8)
output = pipeline(image, key=0)

print(output.shape, output.dtype)  # (224, 224, 3) float32
print(pipeline.explain())
```

`Compose` provides the semantic reference path; `.compile()` selects the
optimized execution plan. `explain()` reports operations, pixel passes,
buffers, copies, dtype and layout changes, fusion, and portable fallbacks. Its
schema version is `2`: each step has an `always`, `conditional`, or `never`
status, and exact `p=0` routes report only work that can execute.

Use an explicit unsigned 64-bit `key` when a result must be independent of call
order or worker assignment. Omitting it advances the sequence associated with
the pipeline seed.

## Data contract

| Stage | Type | Shape and layout |
|---|---|---|
| Pipeline input | NumPy `uint8` | HWC RGB with positive dimensions |
| Default output | NumPy `uint8` | owned, contiguous HWC RGB |
| After `Normalize` | NumPy `float32` | owned, contiguous HWC RGB |
| After terminal `ToTorch` | CPU tensor | contiguous CHW; preserves the current dtype |

Non-contiguous NumPy input is made contiguous at the Python boundary.
`Normalize` must be terminal or immediately precede `ToTorch`; `ToTorch` must
always be last. Public floating-point configuration is stored at its effective
finite `float32` value. Values that overflow `float32` or leave a documented
open or closed domain after conversion raise `ValueError`.

## Transforms

- Geometry: `Resize`, `RandomCrop`, `RandomResizedCrop`, `CenterCrop`,
  `PadIfNeeded`, `Affine`, `RandomRotation`, `Perspective`, and
  `GridDistortion`.
- Flips: `HorizontalFlip` and `VerticalFlip`.
- Color and filtering: `ColorJitter`, `GaussianBlur`, `GaussianNoise`,
  `Sharpen`, `Grayscale`, `Invert`, `Solarize`, and `Posterize`.
- Dropout: `CoarseDropout`.
- Terminal conversion: `Normalize` and `ToTorch`.

Transforms are immutable configuration objects and accept an application
probability `p` where applicable. Geometric operations support the documented
nearest or bilinear interpolation policies and constant or reflect-101 borders.
Variopinta defines its own rounding, sampling, and border semantics; it does
not promise pixel or random-stream identity with another library. `Affine` and
`RandomRotation` reject an input axis above 16,777,216 before rasterization.

## Image I/O

`read_image` and `decode_image` accept JPEG or static PNG and return owned,
contiguous NumPy arrays. Decode modes are `unchanged`, `gray`, `rgb`, and
`rgba`.

`encode_image` and `write_image` support:

- JPEG: `uint8` grayscale or RGB, quality 1–100;
- PNG: one to four `uint8` or `uint16` channels, compression 0–9.

```python
import variopinta as vp

image = vp.read_image("input.jpg")
encoded = vp.encode_image(image, format="jpeg", quality=90)
decoded = vp.decode_image(encoded)
vp.write_image("output.png", decoded, compression=6)
```

Format detection uses file contents when decoding. EXIF orientation, metadata
preservation, and animated PNG are not supported.

`decode_image` and `read_image` also accept `max_encoded_bytes`; it is checked
before the encoded buffer snapshot or complete file read. Both functions use a
100,000,000-pixel decoded-image limit by default. Set either limit to `None` to
disable that limit.

## Native pipeline I/O

Pipeline source and sink policy can be fixed when `Compose` is built. The
default remains `ArrayInput()` plus `ReturnOutput()`, preserving the NumPy and
optional Torch behavior above. The three inputs and three outputs form nine
explicit routes:

| Configuration | Call value or result |
|---|---|
| `ArrayInput()` | NumPy HWC RGB `uint8` source |
| `EncodedInput(...)` | complete JPEG or static PNG in `bytes`, `bytearray`, or `memoryview` |
| `PathInput(...)` | local `str` or `os.PathLike[str]` source |
| `ReturnOutput()` | owned NumPy array or terminal Torch tensor |
| `EncodedOutput(...)` | encoded Python `bytes` |
| `PathOutput(...)` | writes `destination` and returns `None` |

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

service_pipeline = vp.Compose(
    [vp.Resize(224, 224)],
    input=vp.EncodedInput(max_encoded_bytes=32 * 1024 * 1024),
    output=vp.EncodedOutput(format="png", compression=6),
).compile()
response_bytes = service_pipeline(request_bytes, key=7)
```

Encoded and path inputs are detected from their contents and decoded to RGB
without creating a Python array. Mutable encoded carriers are snapshotted at
call entry. Encoded sinks require a statically HWC RGB `uint8` pipeline, so a
pipeline containing executable `Normalize` or any `ToTorch` is rejected when
constructed. JPEG quality is 1–100 (default 95); PNG compression is 0–9
(default 6).

`PathOutput` requires `destination`. A recognized `.jpg`, `.jpeg`, or `.png`
suffix must agree with its configured format; extensionless and other suffixes
are allowed. Parent directories must already exist and existing files are
replaced directly. The source is fully read, decoded, and augmented before the
destination is opened, including when both paths are the same. File writes are
not atomic and concurrent writes to one destination are not coordinated.

Owned encoded and path routes release the GIL during read, decode,
augmentation, encode, and write. Array-backed augmentation keeps the GIL while
borrowing NumPy input. `explain()` reports the configured source and sink,
limits, codec options, materialized buffers, copies, and stage-specific GIL
state without touching a source or destination.

## Reproducibility and limits

Repeated keyed calls are deterministic for the same installed release,
execution environment, pipeline, input, seed, and key. Exact pixels or random
streams are not guaranteed across releases, builds, or platforms. Pin the full
package build and record those inputs when bit-exact replay matters.

Current scope excludes structured targets such as masks and bounding boxes,
GPU execution, native batches, and Python callbacks inside a pipeline. The
augmentation path retains the GIL while it borrows NumPy input; native codec
and file I/O work releases it.

## Performance evidence

The controlled benchmark compares Variopinta with Torchvision v2,
Albumentations, and AlbumentationsX on one reference machine. It measures
equivalent materialized work and records correctness, copies, buffers, kernel
paths, hardware, and statistical limits. The results support the compiled
pipeline design; they do not establish a universal Rust speed advantage.
The published x86-64 results do not claim performance parity on Apple Silicon;
Variopinta-owned kernels use their portable scalar paths there, while resize
and JPEG dependencies may independently select upstream ARM64 SIMD.

The reproducible harness and committed evidence are available under
[`benchmarks/`](https://github.com/claverru/variopinta/tree/main/benchmarks) and
[`results/`](https://github.com/claverru/variopinta/tree/main/results).

## Project information

- [Changelog](https://github.com/claverru/variopinta/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/claverru/variopinta/blob/main/CONTRIBUTING.md)
- [Security policy](https://github.com/claverru/variopinta/blob/main/SECURITY.md)

## License

Variopinta is licensed under the
[Apache License 2.0](https://github.com/claverru/variopinta/blob/main/LICENSE).
Native wheels contain permissively licensed third-party components; their
required attributions are in
[THIRD_PARTY_NOTICES](https://github.com/claverru/variopinta/blob/main/THIRD_PARTY_NOTICES).
