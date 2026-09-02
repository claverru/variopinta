# Transform reference

Transforms are immutable configuration objects. `p` is the probability that a
transform is applied and must be in `[0, 1]`; `ToTorch` is unconditional. Public
floating-point values are stored at their effective finite `float32` value.
Examples below assume `import variopinta as vp`.

The interpolation values are `Interpolation.NEAREST` and
`Interpolation.BILINEAR`. Border-aware transforms accept
`BorderMode.CONSTANT` or `BorderMode.REFLECT101`. A constant `fill` is either
one integer or an RGB triplet with values in `[0, 255]`.

## Geometry

### `Resize(height, width, p=1.0, interpolation=BILINEAR, antialias=False)`

Resizes to a fixed positive height and width. Bilinear resizing uses a fixed
kernel unless `antialias=True` selects scale-adaptive filtering for downscaling.

### `RandomCrop(height, width, p=1.0)`

Samples a crop origin uniformly. An applied crop fails if it is larger than the
current image.

### `RandomResizedCrop(height, width, scale=(0.08, 1.0), ratio=(0.75, 4/3), p=1.0, interpolation=BILINEAR, antialias=False)`

Samples a crop area and aspect ratio, materializes that crop, and resizes it.
The crop boundary is preserved so the resize filter cannot read discarded
pixels.

### `CenterCrop(height, width, p=1.0)`

Returns the centered crop. An applied crop must fit the current image.

### `PadIfNeeded(...)`

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

For each axis, configure exactly one positive minimum or divisor. Available
positions are `CENTER`, `TOP_LEFT`, `TOP_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT`,
and `RANDOM`. Padding never shrinks an image.

### `Affine(...)`

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
scale, and X/Y shear without changing the output size. A scalar `degrees`
becomes a symmetric range. `Affine` rejects an input axis above 16,777,216.

### `RandomRotation(degrees=10.0, p=1.0, interpolation=BILINEAR, border_mode=CONSTANT, fill=0)`

Samples only an angle and uses the affine rasterizer with unchanged size, unit
scale, and no translation or shear. It has the same input-axis limit as
`Affine`.

### `Perspective(scale=0.05, p=1.0, interpolation=BILINEAR, border_mode=CONSTANT, fill=0)`

Samples inward corner displacements while preserving image size. `scale` must
stay below `0.5`; a one-pixel axis is treated as identity.

### `GridDistortion(num_steps=5, distort_limit=0.3, p=1.0, interpolation=BILINEAR, border_mode=CONSTANT, fill=0)`

Builds positive monotonic coordinate maps anchored at both image endpoints.
The requested step count is reduced when an axis has fewer intervals.

## Flips and dropout

- `HorizontalFlip(p=0.5)` reverses the width axis.
- `VerticalFlip(p=0.5)` reverses the height axis.
- `CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.2),
  hole_width_range=(0.1, 0.2), fill=0, p=0.5)` fills sampled rectangles. Integer
  size ranges are pixels; floating-point ranges are fractions of the current
  axis.

## Color and filtering

- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0)`
  samples non-negative factor ranges and a hue offset in turns. Enabled
  adjustments run in a sampled order.
- `GaussianBlur(kernel_size=5, sigma=1.1, p=1.0)` uses a positive odd kernel and
  a positive fixed value or sampling range for sigma.
- `GaussianNoise(mean=0.0, std=10.0, per_channel=True, p=1.0)` expresses mean
  and standard deviation in `uint8` levels. Set `per_channel=False` to share one
  draw across RGB at each pixel.
- `Sharpen(alpha=0.5, lightness=1.0, p=1.0)` blends the source with a reflect-101
  cross-kernel result. `alpha` is in `[0, 1]`; `lightness` is non-negative.
- `Grayscale(p=1.0)` writes the luminance value into all three RGB channels.
- `Invert(p=1.0)` maps each channel to `255 - value`.
- `Solarize(threshold=128, p=1.0)` inverts values at or above the threshold.
- `Posterize(bits=4, p=1.0)` retains between one and eight high bits per channel.

## Terminal transforms

### `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)`

Produces an owned contiguous HWC `float32` array using
`(pixel / max_pixel_value - mean) / std`. It must be terminal or immediately
precede `ToTorch`.

### `ToTorch()`

Produces a contiguous CHW CPU tensor and preserves the current dtype. It must
be the final transform and requires PyTorch to be installed.

## Semantics

Variopinta defines its own sampling, interpolation, border, rounding, and
clipping behavior. It does not promise pixel or random-stream identity with
another augmentation library. Reference and compiled Variopinta pipelines do
promise the same result for the same sampled execution.
