# Variopinta

Variopinta is a CPU image-augmentation compiler: define a pipeline in Python,
then compile it for optimized native execution in Rust, with reproducible
randomness and an inspectable execution plan.

Variopinta is pre-alpha. The public API may change between `0.y.0` releases;
patch releases preserve documented signatures and data contracts unless a
correctness or security fix requires otherwise.

## Why Variopinta?

- **One pipeline, from input to output.** Decode, augment, and deliver arrays,
  tensors, or encoded images in one native call. Compilation plans buffer reuse,
  copy avoidance, and output layout; each target is transformed once for all
  its outputs. See [pipelines and outputs](https://github.com/claverru/variopinta/blob/main/docs/pipelines-and-targets.md#encoded-request-and-response).
- **Reproduce results regardless of worker order.** Use a pipeline seed and a
  per-call `key` to replay an augmentation independently of call order or worker
  assignment, within the same release and execution environment. See
  [control randomness](https://github.com/claverru/variopinta/blob/main/docs/execution.md#control-randomness).
- **See what the compiler actually does.** `explain()` exposes the execution
  plan, including pixel passes, buffers, copies, and layout changes, so you can
  inspect which optimizations your pipeline uses. See
  [execution-plan inspection](https://github.com/claverru/variopinta/blob/main/docs/execution.md#inspect-the-execution-plan).

## Install

Variopinta supports CPython 3.10–3.13 on 64-bit x86 Linux with glibc 2.34 or
newer and on macOS 11 or newer running natively on Apple Silicon.

```bash
python -m pip install variopinta
```

PyTorch is optional and needed only for `ReturnTensor` outputs. Platform
details and source-build instructions are in
[Getting started](https://github.com/claverru/variopinta/blob/main/docs/getting-started.md).

## Quick start

```python
import numpy as np
import variopinta as vp

pipeline = vp.Pipeline(
    [
        vp.RandomCrop(256, 256),
        vp.Resize(224, 224),
        vp.HorizontalFlip(p=0.5),
        vp.Normalize(),
    ],
    seed=42,
).compile()

image = np.random.default_rng(0).integers(0, 256, (320, 320, 3), dtype=np.uint8)
output = pipeline(image, key=0)
replayed = pipeline(image, key=0)

assert np.array_equal(output, replayed)
print(output.shape, output.dtype)  # (224, 224, 3) float32
print(pipeline.explain())
```

`Pipeline` is the semantic reference executor. `.compile()` selects the
optimized execution plan while keeping the same call signature and keyed
result.

Keep the pipeline, input, seed, and key fixed to replay a result. Exact replay
is not guaranteed across releases, builds, or platforms.

## Documentation

- [Getting started](https://github.com/claverru/variopinta/blob/main/docs/getting-started.md)
- [Pipelines and targets](https://github.com/claverru/variopinta/blob/main/docs/pipelines-and-targets.md)
- [Compile, reproduce, and inspect](https://github.com/claverru/variopinta/blob/main/docs/execution.md)
- [Image I/O](https://github.com/claverru/variopinta/blob/main/docs/image-io.md)
- [Transform reference](https://github.com/claverru/variopinta/blob/main/docs/transforms.md)

Current scope includes images and semantic masks with NumPy, encoded-buffer,
and local-path inputs. It does not include boxes, keypoints, GPU execution,
native batches, or Python callbacks inside a pipeline.

## Project information

Performance depends on the pipeline, image sizes, output formats, and hardware.
The [benchmark harness](https://github.com/claverru/variopinta/tree/main/benchmarks)
and [recorded evidence](https://github.com/claverru/variopinta/tree/main/benchmarks/evidence)
support comparisons for specific tested configurations.

- [Changelog](https://github.com/claverru/variopinta/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/claverru/variopinta/blob/main/CONTRIBUTING.md)
- [Security policy](https://github.com/claverru/variopinta/blob/main/SECURITY.md)

Variopinta is licensed under the
[Apache License 2.0](https://github.com/claverru/variopinta/blob/main/LICENSE).
Native-wheel attributions are in
[THIRD_PARTY_NOTICES](https://github.com/claverru/variopinta/blob/main/THIRD_PARTY_NOTICES).

*Variopinta* is the feminine form of the Spanish *variopinto*: “varied in
color or appearance,” from Italian *variopinto*, “varied” and “painted.”
— [RAE](https://dle.rae.es/variopinto)
