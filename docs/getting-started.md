# Getting started

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

### Build from source

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
)

image = vp.read_image("input.jpg")
reference = pipeline(image, key=17)
compiled = pipeline.compile()
optimized = compiled(image, key=17)

assert reference.shape == (224, 224, 3)
assert reference.dtype == np.float32
assert np.array_equal(reference, optimized)
```

Use the reference `Pipeline` while checking semantics and `.compile()` for the
optimized executor. Both expose `.transforms`, `.seed`, `.targets`, and
`.explain()`.

## Control randomness

`seed` initializes the pipeline's random stream. A call can additionally take
an unsigned 64-bit `key`:

```python
first = compiled(image, key=100)
second = compiled(image, key=100)
assert np.array_equal(first, second)
```

Use explicit keys when results must be independent of request order, retries,
or worker assignment. If `key` is omitted, successful calls advance the
sequence owned by that pipeline instance. Validation, acquisition, execution,
encoding, or delivery failures do not consume a sequence position.

## Add a semantic mask

Explicit targets give each input a role and give each output a name. Geometry
is sampled once and shared by all targets; image-only color and filtering
operations do not change masks.

```python
import numpy as np
import variopinta as vp

image_array = vp.ReturnArray(name="array")
mask_array = vp.ReturnArray(name="array")
image_target = vp.Image(name="image", outputs=image_array)
mask_target = vp.Mask(name="labels", outputs=mask_array, fill=255)

pipeline = vp.Pipeline(
    [
        vp.RandomCrop(256, 256),
        vp.HorizontalFlip(p=0.5),
        vp.ColorJitter(p=0.3),
    ],
    seed=42,
    targets=(image_target, mask_target),
).compile()

image = np.zeros((320, 320, 3), dtype=np.uint8)
mask = np.zeros((320, 320), dtype=np.uint8)
result = pipeline(
    image=image_target.bind(image),
    labels=mask_target.bind(mask),
    key=17,
)

assert result.image.array.shape == (256, 256, 3)
assert result.labels.array.shape == (256, 256)
assert result[image_target][image_array] is result.image.array
```

All target inputs must start with the same height and width. Masks use nearest
interpolation without antialiasing, and constant borders use the mask target's
scalar `fill` rather than an image transform's RGB `fill`.

## Return a tensor

Output choice belongs to the target signature. The following pipeline returns
a contiguous CPU CHW tensor and imports PyTorch only when the output is
presented:

```python
import numpy as np
import variopinta as vp

tensor = vp.ReturnTensor(name="tensor")
image_target = vp.Image(outputs=tensor, name="image")
pipeline = vp.Pipeline(
    [vp.Resize(224, 224), vp.Normalize()],
    targets=image_target,
).compile()

image = np.zeros((320, 480, 3), dtype=np.uint8)
result = pipeline(image=image_target.bind(image), key=0)

assert tuple(result.image.tensor.shape) == (3, 224, 224)
```

For encoded buffers, local paths, file outputs, and output fan-out, continue to
[Pipelines and targets](pipelines-and-targets.md), including its dedicated
[multiple-output examples](pipelines-and-targets.md#multiple-outputs). For standalone codecs, see
[Image I/O](image-io.md).

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
