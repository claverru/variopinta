# Changelog

## Unreleased

## 0.7.0

- Add `LongestMaxSize` for aspect-preserving resize with exact half-up dimension
  rounding, optional antialiasing, and shared image/mask geometry. Compose it
  with `PadIfNeeded` for square output without stretching.
- Accelerate constant padding for RGB and grayscale images and label masks,
  preserving fill values and output pixels.
- Renew benchmark evidence, record participant-specific resize rounding, and
  fix fingerprint coverage for tensor-output benchmarks.

## 0.6.0

- Rename target configuration to `input_spec` and `output_specs`, require names
  for public targets and output ports, and give explicit targets a default
  `ReturnArray(name="array")` output.
- Make secondary pipeline, target, output, I/O, and transform configuration
  keyword-only. Rename PNG `compression` to `compression_level`.
- Replace sampled scalar shorthands with explicit `*_range` tuples, clarify
  affine and crop units, and add explicit pixel/fraction units to
  `CoarseDropout` hole sizes.
- Construct compiled pipelines and result containers through pipeline methods
  only. Pickles remain same-release artifacts; pickles from releases with the
  earlier field names are not supported.

Migration mappings for this breaking pre-1.0 release:

| Earlier spelling | Current spelling |
|---|---|
| `Image(carrier=..., outputs=..., name=...)` | `Image(input_spec=..., name=..., output_specs=...)` |
| `Encode(..., compression=n)` | `Encode(..., name=..., compression_level=n)` |
| `RandomResizedCrop(..., scale=..., ratio=...)` | `RandomResizedCrop(..., area_range=..., aspect_ratio_range=...)` |
| `Affine(degrees=d, translate=..., scale=s, shear=...)` | `Affine(degrees_range=(-d, d), translate_max_fraction=..., scale_range=(s, s), shear_x_range=..., shear_y_range=...)` |
| `GaussianNoise(mean=m, std=s)` | `GaussianNoise(mean_range=(m, m), std_range=(s, s))` |
| `Sharpen(alpha=a, lightness=s)` | `Sharpen(blend_weight_range=(a, a), strength_range=(s, s))` |

Convert every other sampled scalar to an explicit two-endpoint range. For old
ColorJitter scalar amounts, use the already-canonical factor endpoints:
`brightness=0.2` becomes `brightness_range=(0.8, 1.2)`. Add
`hole_height_unit="pixels"` or `hole_width_unit="pixels"` when migrating old
integer dropout ranges; omitted units mean fractions.

## 0.5.0

- Support native grayscale `uint8` images in HW and HWC1 layouts across the
  transform catalog, preserving array rank and returning owned, contiguous output.
- Accept scalar and one-element normalization statistics and image fills that
  broadcast to grayscale or RGB. Active three-element values require RGB;
  grayscale normalization requires compatible values such as `mean=0.5, std=0.5`.
- Add `Image(decode_mode="gray")` for encoded-buffer and path inputs, grayscale
  JPEG/PNG outputs, and contiguous single-channel CHW tensor output.
- Upgrade `explain()` to schema 5 with per-target channel alternatives and
  channel-aware execution reporting. Consumers of schema 4 must update.
- Add a Fashion-MNIST notebook with grayscale augmentation previews and PyTorch
  classifier training on CPU or CUDA.
- Add an Oxford-IIIT Pet notebook with shared image/mask augmentation, keyed
  replay, and segmentation training with PyTorch Lightning.

## 0.4.4

- Support standard-library pickle for `Pipeline` and `CompiledPipeline`,
  preserving execution mode, random sequence position, and shared target/output
  port identities while rebuilding independent native execution state.
- Allow Datasets to own compiled pipelines directly with `spawn` and `forkserver`
  workers, without custom serialization or lazy compilation.
- Add an executed Imagenette notebook with augmentation examples, keyed replay,
  and ten epochs of Lightning GPU training using four persistent workers per loader.

## 0.4.3

- Infer NumPy array and `PipelineResult` return types for implicit and explicit
  calls to both pipeline executors, preserving typed output-port lookup.
- Accept sequences of differently typed output ports in static type checking.

## 0.4.2

- Accept a single `Image` or `Mask` directly in `Pipeline(targets=...)`, with
  the same bindings and result structure as a one-element sequence.

## 0.4.1

- Accept a single output port directly in `Image` and `Mask` target signatures,
  while preserving tuple and sequence declarations for multiple outputs.
- Add focused examples for returning, encoding, and writing multiple outputs
  from one transformed target.

## 0.4.0

- Replace pipeline-global I/O and special-case masks with immutable `Image` and
  `Mask` targets. Each target selects an `Array`, `Encoded`, or `Path` carrier
  and fans its final raster out to named `ReturnArray`, `ReturnTensor`, `Encode`,
  and `Write` ports.
- Replace positional explicit calls and arity-collapsing tuples with keyword
  target bindings plus immutable `PipelineResult`/`TargetResult` values.
- Remove the `Return` output and `ToTorch` transform. `ReturnTensor` now owns
  the optional Torch adapter and the CHW terminal layout.
- Rename the public reference executor to `Pipeline`; `.compile()` returns
  `CompiledPipeline` with the same static target signature.
- Move semantic border fill to each `Mask` port, keep mask rasterization nearest
  and lossless, and make standalone I/O generic. PNG `unchanged` decoding
  preserves grayscale samples and palette indices from static 1/2/4/8-bit PNG.
- Make target writes atomic per file after all sources are acquired and every
  encoded result is prepared. Multiple destinations are not one transaction.

## 0.3.1

- Accelerate the dominant single-image CPU paths for padding, sharpening,
  affine and generic remapping, Gaussian noise, hue jitter, and standalone
  HWC-to-CHW conversion. `GaussianNoise` now uses the pinned `rand_distr`
  ZIGNOR stream, so keyed noise pixels differ from earlier releases while
  later transform sampling remains unchanged.

## 0.3.0

- Add immutable array, encoded-buffer, and path pipeline inputs plus returned,
  encoded-buffer, and path outputs, with native decode/augment/encode routes,
  size limits, GIL-aware execution, and source/sink introspection.
- Add `max_encoded_bytes` to `decode_image` and `read_image`.

## 0.2.0

- Add native Apple Silicon support on macOS 11 or newer, including thin ARM64
  wheels, target-honest execution introspection, and two-platform release
  validation.

## 0.1.0

- Compile immutable image-augmentation pipelines with 22 transforms, explicit
  execution plans, deterministic keyed runs, and reusable native workspaces.
- Accept NumPy HWC RGB input and provide optional terminal PyTorch conversion.
- Decode, encode, read, and write JPEG and PNG images through native codecs.
- Support CPython 3.10–3.13 on 64-bit x86 Linux with glibc 2.34 or newer.

The Python API is experimental. Minor `0.y.0` releases may change documented
interfaces; patch releases preserve them except where correctness or security
requires a change.
