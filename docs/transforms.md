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
`vp.BorderMode.REFLECT101`. An image `fill` is an integer or a one-/three-element integer sequence in
`[0, 255]`, excluding booleans. Scalars and one-element sequences broadcast;
three-element sequences require RGB for active transforms, including with
reflect-101 borders. Reflect-101 ignores the compatible fill during rasterization. A mask target has its own scalar
`fill` in `[0, 255]`.

Every sampled configuration uses an ordered two-item tuple. Equal endpoints
select a fixed value.

## Geometry

### `Resize`

```python
vp.Resize(
    height,
    width,
    interpolation=vp.Interpolation.BILINEAR,
    antialias=False,
    p=1.0,
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
    area_range=(0.08, 1.0),
    aspect_ratio_range=(0.75, 4.0 / 3.0),
    interpolation=vp.Interpolation.BILINEAR,
    antialias=False,
    p=1.0,
)
```

Samples a crop area and aspect ratio, then resizes to the positive output
dimensions. `area_range` contains positive fractions no greater than `1.0`;
`aspect_ratio_range` contains positive values. The crop boundary is materialized before resizing so
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
    height_divisor=None,
    width_divisor=None,
    position=vp.PadPosition.CENTER,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
    p=1.0,
)
```

For each axis, configure exactly one positive minimum or positive divisor.
Padding never shrinks a raster. Positions are `CENTER`, `TOP_LEFT`,
`TOP_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT`, and `RANDOM`.

### `Affine`

```python
vp.Affine(
    degrees_range=(-10.0, 10.0),
    translate_max_fraction=(0.0, 0.0),
    scale_range=(1.0, 1.0),
    shear_x_range=(0.0, 0.0),
    shear_y_range=(0.0, 0.0),
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
    p=1.0,
)
```

Applies centered inverse-mapped rotation, relative X/Y translation, isotropic
scale, and X/Y shear without changing the output size.

- `degrees_range` is the inclusive degree range.
- `translate_max_fraction=(x, y)` gives maximum relative displacement for each axis, with
  both values in `[0, 1]`.
- `scale_range` contains positive scale values.
- `shear_x_range` and `shear_y_range` are degree ranges. Every shear angle must
  be strictly between -90 and 90 degrees.

An input axis above 16,777,216 is rejected before rasterization.

### `RandomRotation`

```python
vp.RandomRotation(
    degrees_range=(-10.0, 10.0),
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
    p=1.0,
)
```

Samples an angle and uses the affine rasterizer with unchanged size, unit
scale, and no translation or shear. It has the same input-axis limit as `Affine`.

### `Perspective`

```python
vp.Perspective(
    distortion_scale_range=(0.05, 0.05),
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
    p=1.0,
)
```

Samples inward corner displacements while preserving image size.
`distortion_scale_range` values must be in `[0, 0.5)`. A one-pixel axis is
treated as identity.

### `GridDistortion`

```python
vp.GridDistortion(
    num_steps=5,
    distortion_range=(-0.3, 0.3),
    interpolation=vp.Interpolation.BILINEAR,
    border_mode=vp.BorderMode.CONSTANT,
    fill=0,
    p=1.0,
)
```

Builds positive monotonic coordinate maps anchored at both image endpoints.
`num_steps` is positive and is reduced when an axis has fewer intervals. Every
`distortion_range` value must be strictly inside `(-1, 1)`.

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
    hole_height_unit="fraction",
    hole_width_range=(0.1, 0.2),
    hole_width_unit="fraction",
    fill=0,
    p=0.5,
)
```

Fills sampled image rectangles, which may overlap. `num_holes_range` contains
ordered positive integers. Each size unit is `"pixels"` or `"fraction"`.
Pixel endpoints are positive integers or integer-valued floats and are stored
as integers. Fraction endpoints are numeric values in `(0, 1]` and are stored
at their effective float32 values. The height and width units are independent.
Dropout does not alter masks.

## Color, noise, and filtering

### `ColorJitter`

```python
vp.ColorJitter(
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    hue_range=(0.0, 0.0),
    p=1.0,
)
```

Applies enabled adjustments in a sampled order. Brightness, contrast, and
saturation ranges contain non-negative multiplicative factors. Hue is a range
of displacements in turns and must stay in `[-0.5, 0.5]`.

On grayscale, saturation and hue preserve pixels. Their parameters and order
are still sampled, so channel count does not affect later geometric sampling.
Brightness and contrast retain the composed Q14 path when hue is disabled and
the staged rounding/clipping barriers when hue is enabled.

### `GaussianNoise`

```python
vp.GaussianNoise(
    mean_range=(0.0, 0.0),
    std_range=(10.0, 10.0),
    per_channel=True,
    p=1.0,
)
```

Adds Gaussian noise expressed in `uint8` levels. `mean_range` is finite and
`std_range` is non-negative. With
`per_channel=False`, the RGB channels share one draw at each pixel. Both modes
use the same single draw per pixel on grayscale.

### `Sharpen`

```python
vp.Sharpen(
    blend_weight_range=(0.5, 0.5),
    strength_range=(1.0, 1.0),
    p=1.0,
)
```

Blends the source with a reflect-101 cross-kernel result.
`blend_weight_range` stays in `[0, 1]`; `strength_range` is non-negative.

### `GaussianBlur`

```python
vp.GaussianBlur(kernel_size=5, sigma_range=(1.1, 1.1), p=1.0)
```

Uses a positive odd kernel. `sigma_range` contains positive values.

### `Grayscale`

```python
vp.Grayscale(p=1.0)
```

Writes luminance into all three RGB channels; the output remains RGB.
One-channel input is an identity with preserved rank and owned output.

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

Computes `(pixel / max_pixel_value - mean) / std` per channel and produces
`float32`. Each of `mean` and `std` accepts a finite scalar or a one-/three-element
numeric sequence, excluding booleans. Scalars and one-element sequences broadcast;
three-element sequences require RGB. `std` remains positive after float32
canonicalization, and `max_pixel_value` is positive. The unchanged RGB defaults
require grayscale callers to supply compatible parameters, such as
`Normalize(mean=0.5, std=0.5)`.

`Normalize` must be the final transform. `ReturnArray` presents the result as
HW/HWC with preserved array rank and `ReturnTensor` as CHW. A route on which normalization can execute
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
