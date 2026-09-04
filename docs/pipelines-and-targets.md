# Pipelines and targets

## Pipeline executors

```text
vp.Pipeline(transforms, seed=None, *, targets=None)
```

`Pipeline` is the semantic reference executor. `.compile()` returns an
immutable `CompiledPipeline` with the same transforms, seed, target signature,
call shape, and keyed results. Both executors are callable and expose
`.transforms`, `.seed`, `.targets`, and `.explain()`.

Without `targets`, the pipeline has an implicit image target. It accepts exactly
one positional HWC RGB `uint8` NumPy array and directly returns one owned,
C-contiguous NumPy array:

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
2. its input carrier: `Array`, `Encoded`, or `Path`;
3. one or more output ports.

```python
image_array = vp.ReturnArray(name="array")
labels_array = vp.ReturnArray(name="array")
image_target = vp.Image(name="image", outputs=image_array)
labels_target = vp.Mask(name="labels", outputs=labels_array, fill=255)

pipeline = vp.Pipeline(
    [vp.RandomCrop(256, 256), vp.HorizontalFlip(p=0.5)],
    seed=42,
    targets=(image_target, labels_target),
).compile()
```

Every explicit target and output must have a name. A name must be a public
Python identifier: it cannot start with `_`, be a keyword, or be `key`. Target
names are unique across the pipeline; output names are unique within their
target.

Pass one output port directly. Use a tuple or another sequence when a target
has multiple outputs. `target.outputs` is always stored as a tuple.

Explicit calls are keyword-only and must bind every declared target exactly
once. A binding belongs to the target object that created it:

```python
result = pipeline(
    labels=labels_target.bind(mask),
    image=image_target.bind(image),
    key=7,
)
```

Keyword order is irrelevant. Missing, extra, positional, or foreign bindings
are rejected before native execution.

## Carriers

| Carrier | Accepted source | Options |
|---|---|---|
| `Array()` | image: NumPy HWC RGB `uint8`; mask: NumPy HW `uint8` | none |
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

## Output ports

| Output | Value | Options |
|---|---|---|
| `ReturnArray(name=...)` | owned C-contiguous NumPy array | name |
| `ReturnTensor(name=...)` | contiguous CPU Torch tensor | name |
| `Encode(format, quality=None, compression=None, name=...)` | Python `bytes` | JPEG or PNG options |
| `Write(format=None, quality=None, compression=None, name=...)` | normalized `pathlib.Path` | JPEG or PNG options |

Image arrays are HWC; image tensors are CHW. Both preserve the semantic dtype:
normally `uint8`, or `float32` after `Normalize`. Mask arrays and tensors are HW
`uint8`; tensor output does not add a channel or convert labels to `int64`.
PyTorch is imported only for a declared `ReturnTensor` route.

JPEG image output accepts `quality` from 1 through 100 and defaults to 95. PNG
image output accepts `compression` from 0 through 9 and defaults to 6. Mask
encoding and writing are always lossless 8-bit grayscale PNG.

An image route with `Normalize(p>0)` cannot declare `Encode` or `Write`, because
the final raster may be `float32`. `Normalize(p=0)` is a never-executed route
and remains compatible.

### Encoded request and response

An encoded service route can keep decode, augmentation, and encode inside one
native call:

```python
from pathlib import Path

import variopinta as vp

png = vp.Encode("png", compression=3, name="png")
image_target = vp.Image(
    vp.Encoded(max_encoded_bytes=8 * 1024 * 1024),
    outputs=png,
    name="image",
)
pipeline = vp.Pipeline(
    [vp.Resize(224, 224)],
    targets=(image_target,),
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
image_target = vp.Image(name="image", outputs=(array, jpeg))
pipeline = vp.Pipeline(
    [vp.Resize(224, 224)],
    targets=(image_target,),
).compile()

image = np.zeros((320, 480, 3), dtype=np.uint8)
result = pipeline(image=image_target.bind(image), key=7)

assert result.image.array.shape == (224, 224, 3)
assert isinstance(result.image.jpeg, bytes)
```

Returned mutable siblings own independent storage. Encoding and writing read
the common final raster, never an intermediate transform state. Binding the
same source to two targets still executes both target routes.

Outputs can also combine returned data with a destination bound for each call:

```python
from pathlib import Path

import variopinta as vp

array = vp.ReturnArray(name="array")
png = vp.Write("png", compression=3, name="png")
image_target = vp.Image(vp.Path(), outputs=(array, png), name="image")

pipeline = vp.Pipeline(
    [vp.Resize(512, 512)],
    targets=(image_target,),
).compile()

result = pipeline(
    image=image_target.bind(
        "input.png",
        png.bind("output.png"),
    ),
    key=7,
)

assert result.image.array.shape == (512, 512, 3)
assert result.image.png == Path("output.png")
```

Declaration order controls output introspection, while named and identity
lookups remain independent of that order.

## Write bindings

A `Write` output is part of the static target signature, but its destination is
bound per call as shown above.

`Write(format=None)` infers JPEG or PNG from the destination suffix. A known
suffix must agree with an explicit format. Target binding rejects missing,
duplicate, or foreign `Write` bindings.

Before changing a destination, a call validates all bindings and paths,
acquires and decodes every source, checks the common canvas, executes all
targets, and prepares every encoding. Exact duplicate destinations are rejected
globally. Each file is installed through a sibling temporary file and atomic
rename where the platform supports it. Multiple files do not form one
cross-file transaction.

## Reading results

Explicit pipelines always return an immutable `PipelineResult`, even with one
target and one output. Read values using names or the original port objects:

```python
import numpy as np
import variopinta as vp

image_array = vp.ReturnArray(name="array")
image_target = vp.Image(outputs=image_array, name="image")
pipeline = vp.Pipeline([], targets=(image_target,))
image = np.zeros((8, 8, 3), dtype=np.uint8)
result = pipeline(image=image_target.bind(image), key=0)

array_by_name = result.image.array
array_by_identity = result[image_target][image_array]
assert array_by_name is array_by_identity
```

Identity lookup preserves the output port's static result type. String indexing
is not supported. Result `repr()` values show compact shape and type facts but
do not include raster, source, or destination payloads.

## Image and mask behavior

All inputs in a call must share their initial height and width. The pipeline
samples one plan for the call and applies it to every target.

Crops, flips, resize, padding, affine, rotation, perspective, and grid
distortion apply to masks with nearest interpolation and no antialiasing.
Constant borders use each mask target's own scalar `fill`. Color, filtering,
noise, dropout, and normalization are image-only.

## Ownership and concurrency

Inputs are never mutated. Non-contiguous arrays are copied to contiguous
storage once per target. Returned arrays own their storage and are
C-contiguous.

A call containing any `Array` carrier retains the Python GIL during aggregate
augmentation. Calls whose inputs are entirely `Encoded` or `Path` release the
GIL through acquisition, augmentation, encoding, and delivery. Compiled
pipelines are safe to share across workers.

Use an explicit unsigned 64-bit `key` when output must not depend on call order
or worker scheduling. Without a key, successful calls advance the sequence
associated with the pipeline's unsigned 64-bit seed. Failed calls do not
advance it.

## Inspect the execution plan

```python
plan = pipeline.explain()
```

`explain()` describes transforms, target carriers, ordered outputs, operations,
pixel passes, buffers, copies, dtype and layout changes, fusion, portable
fallbacks, codec options, and delivery. Operations are marked `always`,
`conditional`, or `never`; exact `p=0` routes report only work that can execute.

The report distinguishes semantic transform passes from output fan-out,
including a normalized raster written directly as CHW and a terminal HWC-to-CHW
copy when direct production is unavailable. Declaration order is introspection
order, not call order. Runtime arrays, tensors, encoded contents, source paths,
and destination paths are never inspected or included.

Standalone codec functions and their full data matrix are documented in
[Image I/O](image-io.md).
