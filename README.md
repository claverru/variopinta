# Variopinta

Variopinta is an experimental CPU image-augmentation compiler. Pipelines are
configured in Python and executed by optimized Rust kernels, with whole-pipeline
planning for buffer reuse, kernel selection, and fewer Python/native crossings.

Variopinta is pre-alpha. The public API may change between `0.y.0` releases;
patch releases preserve documented signatures and data contracts unless a
correctness or security fix requires otherwise.

## Install

Variopinta supports CPython 3.10–3.13 on supported 64-bit Linux and Apple
Silicon macOS systems.

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

image = np.zeros((320, 320, 3), dtype=np.uint8)
output = pipeline(image, key=0)

print(output.shape, output.dtype)  # (224, 224, 3) float32
```

`Pipeline` is the semantic reference executor. `.compile()` selects the
optimized execution plan while keeping the same call signature and keyed
result. Use `pipeline.explain()` to inspect the operations, buffers, copies,
fusion, dtypes, layouts, and target routes in that plan.

## Documentation

- [Getting started](https://github.com/claverru/variopinta/blob/main/docs/getting-started.md)
- [Pipelines and targets](https://github.com/claverru/variopinta/blob/main/docs/pipelines-and-targets.md)
- [Image I/O](https://github.com/claverru/variopinta/blob/main/docs/image-io.md)
- [Transform reference](https://github.com/claverru/variopinta/blob/main/docs/transforms.md)

Current scope includes images and semantic masks with NumPy, encoded-buffer,
and local-path inputs. It does not include boxes, keypoints, GPU execution,
native batches, or Python callbacks inside a pipeline.

## Project information

- [Changelog](https://github.com/claverru/variopinta/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/claverru/variopinta/blob/main/CONTRIBUTING.md)
- [Security policy](https://github.com/claverru/variopinta/blob/main/SECURITY.md)
- [Benchmark harness](https://github.com/claverru/variopinta/tree/main/benchmarks)

Variopinta is licensed under the
[Apache License 2.0](https://github.com/claverru/variopinta/blob/main/LICENSE).
Native-wheel attributions are in
[THIRD_PARTY_NOTICES](https://github.com/claverru/variopinta/blob/main/THIRD_PARTY_NOTICES).
