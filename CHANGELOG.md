# Changelog

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
