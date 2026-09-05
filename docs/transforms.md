# Transform reference

Transforms are immutable configuration objects. Every transform has an
application probability `p` in `[0, 1]`; `p=0` is never applied and `p=1` is
always applied. Public floating-point values are stored at their effective
finite `float32` value.

Examples below assume:

```python
import variopinta as vp
```

## Common options

`interpolation` accepts `vp.Interpolation.NEAREST` or
`vp.Interpolation.BILINEAR`. `antialias=True` enables scale-adaptive filtering
for bilinear downscaling where offered.

`border_mode` accepts `vp.BorderMode.CONSTANT` or
`vp.BorderMode.REFLECT101`. An image `fill` is one integer or an RGB tuple with
values in `[0, 255]`; reflect-101 ignores it. A mask target has its own scalar
`fill` in `[0, 255]`.

A scalar sampling argument usually fixes a value or creates the symmetric
range described below. A two-item tuple gives an explicit inclusive sampling
range and must be ordered.

## Geometry

### `Resize`

```python
vp.Resize(
    height,
    width,
    p=1.0,
    interpolation=vp.Interpolation.BILINEAR,
    antialias=False,
)
```

Resizes to a positive `height` and `width`. Masks always use nearest
interpolation without antialiasing.

### `RandomCrop`

```python
vp.RandomCrop(height, width, p=1.0)
```

Samples a crop origin uniformly. `height` and `width` must be positive; an
applied crop fails if it does not fit the current raster.

### `RandomResizedCrop`

```python
vp.RandomResizedCrop(
    height,
    width,
    scale=(0.08, 1.0),
    ratio=(0.75, 4.0 / 3.0),
    p=1.0,
    interpolation=vp.Interpolation.BILINEAR,
    antialias=False,
)
```

Samples a crop area and aspect ratio, then resizes to the positive output
dimensions. `scale` contains positive fractions no greater than `1.0`; `ratio`
contains positive values. The crop boundary is materialized before resizing so
the filter cannot read discarded pixels.

### `CenterCrop`

```python
vp.CenterCrop(height, width, p=1.0)
```

Returns a centered crop with positive dimensions. An applied crop must fit the
current raster.

### `PadIfNeeded`

```python
vp.PadIfNeeded(
    min_height=None,
    min_width=None,
    pad_height_divisor=None,
    pad_width_divisor=None,
    position=vp.PadPosition.CENTER,
    p=1.0,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
)
```

For each axis, configure exactly one positive minimum or positive divisor.
Padding never shrinks a raster. Positions are `CENTER`, `TOP_LEFT`,
`TOP_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT`, and `RANDOM`.

### `Affine`

```python
vp.Affine(
    degrees=10.0,
    translate=(0.0, 0.0),
    scale=1.0,
    shear=0.0,
    p=1.0,
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
)
```

Applies centered inverse-mapped rotation, relative X/Y translation, isotropic
scale, and X/Y shear without changing the output size.

- A non-negative scalar `degrees=d` samples from `(-d, d)`; a tuple supplies
  the explicit degree range.
- `translate=(x, y)` gives maximum relative displacement for each axis, with
  both values in `[0, 1]`.
- A scalar `scale` fixes a positive value; a tuple gives a positive range.
- A non-negative scalar `shear=s` samples X shear from `(-s, s)`. A two-item
  tuple supplies the X range; a four-item tuple supplies X then Y ranges. Every
  shear angle must be strictly between -90 and 90 degrees.

An input axis above 16,777,216 is rejected before rasterization.

### `RandomRotation`

```python
vp.RandomRotation(
    degrees=10.0,
    p=1.0,
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
)
```

Samples an angle and uses the affine rasterizer with unchanged size, unit
scale, and no translation or shear. A non-negative scalar creates a symmetric
range; a tuple supplies the explicit range. It has the same input-axis limit as
`Affine`.

### `Perspective`

```python
vp.Perspective(
    scale=0.05,
    p=1.0,
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
)
```

Samples inward corner displacements while preserving image size. A scalar
fixes `scale`; a tuple gives a range. Values must be in `[0, 0.5)`. A one-pixel
axis is treated as identity.

### `GridDistortion`

```python
vp.GridDistortion(
    num_steps=5,
    distort_limit=0.3,
    p=1.0,
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
)
```

Builds positive monotonic coordinate maps anchored at both image endpoints.
`num_steps` is positive and is reduced when an axis has fewer intervals. A
non-negative scalar `distort_limit=d` creates `(-d, d)`; a tuple supplies the
range. Every value must be strictly inside `(-1, 1)`.

## Flips and dropout

### `HorizontalFlip`

```python
vp.HorizontalFlip(p=0.5)
```

Reverses the width axis.

### `VerticalFlip`

```python
vp.VerticalFlip(p=0.5)
```

Reverses the height axis.

### `CoarseDropout`

```python
vp.CoarseDropout(
    num_holes_range=(1, 2),
    hole_height_range=(0.1, 0.2),
    hole_width_range=(0.1, 0.2),
    fill=0,
    p=0.5,
)
```

Fills sampled image rectangles, which may overlap. `num_holes_range` contains
ordered positive integers. Each size range must contain either two positive
integers for pixels or two floats in `(0, 1]` for fractions of the current
axis. Dropout does not alter masks.

## Color, noise, and filtering

### `ColorJitter`

```python
vp.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.0,
    p=1.0,
)
```

Applies enabled adjustments in a sampled order. A non-negative scalar
brightness, contrast, or saturation value `v` creates the factor range
`(max(0, 1-v), 1+v)`; a tuple supplies a non-negative range. A hue scalar in
`[0, 0.5]` creates a symmetric range in turns; an explicit range must stay in
`[-0.5, 0.5]`.

### `GaussianNoise`

```python
vp.GaussianNoise(mean=0.0, std=10.0, per_channel=True, p=1.0)
```

Adds Gaussian noise expressed in `uint8` levels. `mean` is a fixed scalar or
finite range; `std` is a fixed non-negative scalar or non-negative range. With
`per_channel=False`, the RGB channels share one draw at each pixel.

### `Sharpen`

```python
vp.Sharpen(alpha=0.5, lightness=1.0, p=1.0)
```

Blends the source with a reflect-101 cross-kernel result. `alpha` is a fixed
value or range in `[0, 1]`; `lightness` is a fixed non-negative value or
non-negative range.

### `GaussianBlur`

```python
vp.GaussianBlur(kernel_size=5, sigma=1.1, p=1.0)
```

Uses a positive odd kernel. `sigma` is a fixed positive value or an explicit
positive range.

### `Grayscale`

```python
vp.Grayscale(p=1.0)
```

Writes luminance into all three RGB channels; the output remains RGB.

### `Invert`

```python
vp.Invert(p=1.0)
```

Maps each channel to `255 - value`.

### `Solarize`

```python
vp.Solarize(threshold=128, p=1.0)
```

Inverts channel values at or above the integer threshold, which must be in
`[0, 255]`.

### `Posterize`

```python
vp.Posterize(bits=4, p=1.0)
```

Retains the requested number of high bits per channel. `bits` is an integer in
`[1, 8]`.

All transforms in this section are image-only and leave masks unchanged.

## Terminal conversion

### `Normalize`

```python
vp.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value=255.0,
    p=1.0,
)
```

Computes `(pixel / max_pixel_value - mean) / std` per RGB channel and produces
`float32`. `mean` contains three finite values, `std` contains three positive
values, and `max_pixel_value` is positive.

`Normalize` must be the final transform. `ReturnArray` presents the result as
HWC and `ReturnTensor` as CHW. A route on which normalization can execute
cannot encode or write its final image. Masks are unchanged.

## Cross-target semantics

Every target in one call shares geometric sampling. Masks use nearest
interpolation without antialiasing, and constant borders use the mask target's
`fill`. Image-only transforms do not consume a separate mask plan.

Variopinta defines its own sampling, interpolation, border, rounding, and
clipping behavior. It does not promise pixel or random-stream identity with
another library. Reference and compiled pipelines do produce the same result
for the same sampled execution.

See [Pipelines and targets](pipelines-and-targets.md) for target signatures and
result layouts, and [execution](execution.md) for deterministic keys and
compiled-plan inspection.
