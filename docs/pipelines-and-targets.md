# Pipelines and targets

Connect input acquisition, augmentation, and output delivery in one pipeline.
Use [encoded inputs and outputs](#encoded-request-and-response) to keep decode,
augmentation, and encode inside one native call, or declare
[multiple outputs](#multiple-outputs) to transform each target once for every
requested output.

## Pipeline executors

```text
vp.Pipeline(transforms, *, seed=None, targets=None)
```

`Pipeline` and its compiled executor share the input and output contracts below.
See [compilation](execution.md#compile-a-pipeline) for execution semantics.

With `targets=None` (the default), the pipeline has an implicit image target.
It accepts exactly one positional HW grayscale, HWC1 grayscale, or HWC3 RGB
`uint8` NumPy array and directly
returns one owned, C-contiguous NumPy array:

```python
import numpy as np
import variopinta as vp

pipeline = vp.Pipeline([vp.Resize(224, 224)], seed=42).compile()
image = np.zeros((320, 480, 3), dtype=np.uint8)
output = pipeline(image, key=7)
```

Use explicit targets when a call needs masks, encoded or path inputs, tensors,
encoding, writes, multiple inputs, or multiple outputs.

## Explicit targets

A target combines three decisions:

1. its semantic role: `Image` or `Mask`;
2. its `input_spec`: `Array`, `Encoded`, or `Path`;
3. one or more `output_specs`.

For segmentation, declare an image and a mask. Geometry is sampled once and
shared by both; image-only color and filtering operations leave the mask alone.

```python
import numpy as np
import variopinta as vp

image_array = vp.ReturnArray(name="array")
labels_array = vp.ReturnArray(name="array")
image_target = vp.Image(name="image", output_specs=image_array)
labels_target = vp.Mask(name="labels", output_specs=labels_array, fill=255)

pipeline = vp.Pipeline(
    [
        vp.LongestMaxSize(256),
        vp.PadIfNeeded(min_height=256, min_width=256),
        vp.HorizontalFlip(p=0.5),
        vp.ColorJitter(p=0.3),
    ],
    seed=42,
    targets=(image_target, labels_target),
).compile()
```

Every publicly constructed target and output must have a name. A name must be a public
Python identifier: it cannot start with `_`, be a keyword, or be `key`. Target
names are unique across the pipeline; output names are unique within their
target. `Image(name="image")` and `Mask(name="mask")` each default to a distinct
`ReturnArray(name="array")` output.

Both `targets` and `output_specs` accept a single port or a non-empty sequence:
`targets=image_target` or `targets=(image_target, labels_target)`, and
`output_specs=image_array` or `output_specs=(image_array, jpeg)`. The attributes
`pipeline.targets` and `target.output_specs` always contain tuples. Repeating the
same target within a pipeline or the same output within a target is rejected.

Explicit calls are keyword-only and must bind every declared target exactly
once. A binding belongs to the target object that created it:

```python
image = np.zeros((320, 320, 3), dtype=np.uint8)
mask = np.zeros((320, 320), dtype=np.uint8)
result = pipeline(
    labels=labels_target.bind(mask),
    image=image_target.bind(image),
    key=7,
)
```

Here the image and labels keep their aspect ratio, receive identical resize
dimensions and centered offsets, and finish at 256 × 256. The image uses the
configured resize interpolation and padding fill. The labels use nearest
resize interpolation and the mask target's `fill=255`.

Keyword order is irrelevant. Missing, extra, positional, or foreign bindings
are rejected before native execution.

## Reading results

Explicit pipelines always return a `PipelineResult` containing one
`TargetResult` per target, even with `targets=image_target` and one output.
Read values using names or the original port objects:

```python
array_by_name = result.image.array
array_by_identity = result[image_target][image_array]
assert array_by_name is array_by_identity
assert result.labels.array.shape == (256, 256)
```

The result containers are immutable; the returned arrays and tensors remain
mutable. Both executors infer `np.ndarray` for positional-array calls and
`PipelineResult` for keyword-binding calls. Identity lookup preserves each
output port's static result type; named output lookup is dynamic. The configured
call mode, binding names, and port identities are validated at runtime.
String indexing is not supported. Result `repr()` values show compact shape
and type facts without raster, source, or destination payloads.

## Input specifications

| Input specification | Accepted source | Options |
|---|---|---|
| `Array()` | image: NumPy HW/HWC1 grayscale or HWC3 RGB `uint8`; mask: NumPy HW `uint8` | none |
| `Encoded(...)` | `bytes`, `bytearray`, or `memoryview` | `max_pixels=100_000_000`, `max_encoded_bytes=None` |
| `Path(...)` | local `str` or `os.PathLike[str]` | `max_pixels=100_000_000`, `max_encoded_bytes=None` |

`Encoded` and `Path` image targets accept JPEG and static PNG. Encoded and path
mask targets accept only static grayscale or indexed PNG without alpha, using
1-, 2-, 4-, or 8-bit samples. Mask inputs reject JPEG, RGB/RGBA, 16-bit,
animated, transparent, and malformed files.

`max_pixels` and `max_encoded_bytes` are checked during acquisition. Pass
`None` to disable a limit deliberately. Paths are local filesystem paths;
URLs, glob syntax, and `bytes` paths are rejected. Mutable encoded inputs are
snapshotted before native work.

`Image(input_spec=Array(), *, name, output_specs=..., decode_mode=None)` infers
channels from arrays; explicit decoder modes are rejected for `Array`. `Mask`
uses the same target arguments and adds keyword-only `fill=0`. For `Encoded` and `Path`,
`None` and `"rgb"` decode RGB, while `"gray"` requests grayscale, including
conversion from color files. Decoded 16-bit pixels are rejected by augmentation.
Mask decoding retains its separate label-preserving rules.

## Grayscale images

Grayscale images run directly through every image transform without RGB
expansion. Supply scalar or one-element normalization parameters:

```python
import numpy as np
import variopinta as vp

image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
pipeline = vp.Pipeline([
    vp.Resize(32, 32),
    vp.Normalize(mean=0.5, std=0.5),
]).compile()
assert pipeline(image).shape == (32, 32)
assert pipeline(image[..., None]).shape == (32, 32, 1)
```

`Normalize()` keeps its RGB defaults and therefore rejects grayscale. Scalar
or one-element image fills and normalization statistics broadcast; three-element
sequences require RGB even if all components are equal. An inactive `p=0`
transform does not restrict the supported channels. Other parameter validation
still applies at construction.

One array pipeline may alternate grayscale and RGB calls. All image targets
must support the active shared parameters, and mixed targets share one sampled
geometry. Unsupported channels are rejected before sampling or writes and do
not consume an implicit key. RGB arrays remain RGB even when their three
channels have identical values.

For native grayscale acquisition use
`vp.Image(vp.Path(), name="image", output_specs=vp.ReturnArray(name="array"), decode_mode="gray")`.
Grayscale `uint8` results may also be encoded or written as grayscale JPEG/PNG.

## Output ports

| Output | Value | Options |
|---|---|---|
| `ReturnArray(*, name)` | owned C-contiguous NumPy array | name |
| `ReturnTensor(*, name)` | contiguous CPU Torch tensor | name |
| `Encode(format, *, name, quality=None, compression_level=None)` | Python `bytes` | JPEG or PNG options |
| `Write(format=None, *, name, quality=None, compression_level=None)` | normalized `pathlib.Path` | JPEG or PNG options |

Image arrays preserve the input rank (HW, HWC1, or HWC3); decoded grayscale
arrays are HW. Image tensors are CHW, including `(1, H, W)` for grayscale.
Both preserve the semantic dtype:
normally `uint8`, or `float32` after `Normalize`. Mask arrays and tensors are HW
`uint8`; tensor output does not add a channel or convert labels to `int64`.
PyTorch is imported only for a declared `ReturnTensor` route.

Codec options and defaults are listed in [Image I/O](image-io.md#encode-and-write).
Mask encoding and writing are always lossless 8-bit grayscale PNG.

An image route with `Normalize(p>0)` cannot declare `Encode` or `Write`, because
the final raster may be `float32`. `Normalize(p=0)` is a never-executed route
and remains compatible.

## Return a tensor

Output choice belongs to the target signature. The following pipeline returns
a contiguous CPU CHW tensor and imports PyTorch only when the output is
presented:

```python
import numpy as np
import variopinta as vp

tensor = vp.ReturnTensor(name="tensor")
image_target = vp.Image(output_specs=tensor, name="image")
pipeline = vp.Pipeline(
    [vp.Resize(224, 224), vp.Normalize()],
    targets=image_target,
).compile()

image = np.zeros((320, 480, 3), dtype=np.uint8)
result = pipeline(image=image_target.bind(image), key=0)

assert tuple(result.image.tensor.shape) == (3, 224, 224)
```

For this tensor-only target, compiled normalization writes `float32` values
directly in CHW layout. Adding an array output requires an HWC raster as well;
inspect the selected route with [`.explain()`](execution.md#inspect-the-execution-plan).

## Encoded request and response

An encoded service route can keep decode, augmentation, and encode inside one
native call:

```python
from pathlib import Path

import variopinta as vp

png = vp.Encode("png", compression_level=3, name="png")
image_target = vp.Image(
    vp.Encoded(max_encoded_bytes=8 * 1024 * 1024),
    output_specs=png,
    name="image",
)
pipeline = vp.Pipeline(
    [vp.Resize(224, 224)],
    targets=image_target,
).compile()

request_bytes = Path("input.jpg").read_bytes()
result = pipeline(image=image_target.bind(request_bytes), key=7)
response_bytes = result.image.png
```

`request_bytes` is a complete JPEG or PNG payload. `response_bytes` is a PNG
`bytes` object.

## Multiple outputs

A target is transformed once before its final raster is delivered to every
declared output. Return an array alongside encoded bytes by declaring both
ports in the target signature:

```python
import numpy as np
import variopinta as vp

array = vp.ReturnArray(name="array")
jpeg = vp.Encode("jpeg", quality=90, name="jpeg")
image_target = vp.Image(name="image", output_specs=(array, jpeg))
pipeline = vp.Pipeline(
    [vp.Resize(224, 224)],
    targets=image_target,
).compile()

image = np.zeros((320, 480, 3), dtype=np.uint8)
result = pipeline(image=image_target.bind(image), key=7)

assert result.image.array.shape == (224, 224, 3)
assert isinstance(result.image.jpeg, bytes)
```

Encoding and writing read the common final raster, never an intermediate
transform state. Binding the same source to two targets still executes both
target routes.

Declaration order controls output introspection, while named and identity
lookups remain independent of that order.

## Write bindings

A `Write` output declares a file output, and `Write.bind()` supplies its
destination for each call. Pass that binding after the source in
`target.bind()`.

This example reads an image from disk into a NumPy array, transforms it once,
and both returns the resulting array and writes it to a PNG file. These are
two outputs of one image target:

```python
from pathlib import Path

import variopinta as vp

array = vp.ReturnArray(name="array")
png = vp.Write("png", compression_level=3, name="png")
image_target = vp.Image(output_specs=(array, png), name="image")

pipeline = vp.Pipeline(
    [vp.Resize(512, 512)],
    targets=image_target,
).compile()

image = vp.read_image("input.png")
result = pipeline(
    image=image_target.bind(
        image,
        png.bind("output.png"),
    ),
    key=7,
)

assert result.image.array.shape == (512, 512, 3)
assert result.image.png == Path("output.png")
```

`result.image.array` contains the transformed pixels; `result.image.png`
contains the written path. The PNG stores the same pixels as the returned
array, and the input array is unchanged. To include reading and decoding in the
native pipeline call as well, declare the image target with `vp.Path()` and
bind the source path instead of a preloaded array.

`Write(format=None)` infers JPEG or PNG from the destination suffix. A known
suffix must agree with an explicit format. Target binding rejects missing,
duplicate, or foreign `Write` bindings.

Before changing a destination, a call validates all bindings and paths,
acquires and decodes every source, checks the common canvas, executes all
targets, and prepares every encoding. Exact duplicate destinations are rejected
globally. Each file is installed through a sibling temporary file and atomic
rename where the platform supports it. Multiple files do not form one
cross-file transaction.

## Image and mask behavior

All inputs in a call must share their initial height and width. The pipeline
samples one plan for the call and applies it to every target.

Crops, flips, fixed or aspect-preserving resize, padding, affine, rotation, perspective, and grid
distortion apply to masks with nearest interpolation and no antialiasing.
Constant borders use each mask target's own scalar `fill`. Color, filtering,
noise, dropout, and normalization are image-only.

## Ownership

Inputs are never mutated. Non-contiguous arrays are copied to contiguous
storage once per target. Returned arrays own their storage and are
C-contiguous. Returned mutable siblings own independent storage.

See [execution](execution.md) for compilation, deterministic keys, worker and
GIL behavior, execution-plan inspection, and performance evidence. Standalone
codec functions and their full data matrix are documented in
[Image I/O](image-io.md).
