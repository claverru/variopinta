# Getting started

Install Variopinta and run a compiled pipeline on a NumPy image. The example
creates its own input, so you can run it without an image file or PyTorch.

## Requirements

Published wheels support CPython 3.10–3.13 on:

- 64-bit x86 Linux with glibc 2.34 or newer;
- macOS 11 or newer running natively on Apple Silicon.

AVX2 is detected at runtime on x86-64 and is not required. Other Python
implementations, operating systems, architectures, and 32-bit environments are
not supported.

Install from PyPI:

```bash
python -m pip install variopinta
```

NumPy is the only required runtime dependency. `ReturnTensor` is optional and
requires a PyTorch build compatible with your Python and platform:

```bash
python -m pip install torch
```

## Default image-to-image pipeline

The default API accepts one HWC RGB `uint8` NumPy array and returns one owned,
C-contiguous NumPy array:

```python
import numpy as np
import variopinta as vp

pipeline = vp.Pipeline(
    [
        vp.RandomResizedCrop(224, 224, scale=(0.6, 1.0)),
        vp.HorizontalFlip(p=0.5),
        vp.Normalize(),
    ],
    seed=42,
).compile()

image = np.random.default_rng(0).integers(0, 256, (320, 320, 3), dtype=np.uint8)
output = pipeline(image, key=17)

print(output.shape, output.dtype)  # (224, 224, 3) float32
```

The pipeline crops and resizes the image, may flip it, then normalizes it to
`float32`. `.compile()` selects the optimized execution plan. The seed and
`key` let you replay this result with the same input and environment; see
[control randomness](execution.md#control-randomness) for the complete contract.

## Next steps

- [Connect inputs and outputs](pipelines-and-targets.md): transform images and
  masks together, decode and encode within a pipeline, or return tensors and
  multiple outputs.
- [Compile, reproduce, and inspect](execution.md): compare reference and
  compiled execution, replay samples independently of worker order, and see a
  concrete optimization in `explain()`.
- [Choose transforms](transforms.md): look up constructors, defaults, and
  image/mask behavior.

## Common errors

- Image arrays must be positive-size HWC RGB `uint8`; mask arrays must be
  positive-size HW `uint8`.
- An explicit pipeline accepts only named keyword bindings created from the
  exact target objects used to construct it.
- Every explicit target and output needs a unique public Python-identifier name
  in its scope.
- `Normalize` must be the last transform. Any route that may normalize cannot
  also encode or write its final image.
- `ReturnTensor` raises `ImportError` if PyTorch is not installed.

The [transform reference](transforms.md) lists constructor constraints. The
[pipeline guide](pipelines-and-targets.md) gives the complete input and output
contract.

## Build from source

A source build needs Rust 1.87 or newer and CMake. Linux x86-64 also needs a
C/C++ toolchain and NASM:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake nasm
python -m pip install .
```

On Apple Silicon, install the Xcode command-line tools and CMake; NASM is not
required. Builds use Maturin through Python build isolation and compile the
locked vendored libjpeg-turbo statically. They do not use Homebrew or MacPorts
codec libraries.
