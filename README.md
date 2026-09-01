# Variopinta

Variopinta is an experimental CPU image-augmentation compiler with a Python
configuration API and a Rust execution core. It compiles complete pipelines to
reduce Python/native crossings, reuse buffers, select optimized kernels, and
report the resulting execution plan.

> *Variopinta* is the feminine form of the Spanish *variopinto*: “varied in
> color or appearance,” from Italian *variopinto*, “varied” and “painted.”
> — [RAE](https://dle.rae.es/variopinto)

Variopinta is image-only and pre-release. The public Python API may change
between `0.y.0` releases; patch releases preserve documented signatures and
data contracts unless a correctness or security fix requires otherwise.

## Installation

Version 0.1 supports CPython 3.10–3.13 on 64-bit x86 Linux with glibc 2.34 or
newer. AVX2 is detected at runtime and is not required. Other Python
implementations, operating systems, architectures, and 32-bit environments are
not supported.

Install Variopinta from PyPI:

```bash
python -m pip install variopinta
```

To build from a source checkout, install Rust 1.87 or newer, a C/C++ toolchain,
CMake, and NASM. On Ubuntu, the native prerequisites are:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake nasm
python -m pip install .
```

The build uses Maturin through Python build isolation. NumPy is installed as
the only required runtime dependency.

`ToTorch` is optional and requires a PyTorch build compatible with the selected
Python and Linux environment:

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

The reproducible harness and committed evidence are available under
[`benchmarks/`](benchmarks/) and [`results/`](results/).

## Project information

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

Variopinta is licensed under the [Apache License 2.0](LICENSE). Native wheels
contain permissively licensed third-party components; their required
attributions are in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
