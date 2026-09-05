# Compile, reproduce, and inspect

Compile a pipeline once, reuse it across calls, and inspect how its transforms
and outputs will execute. This guide covers execution; [pipelines and
targets](pipelines-and-targets.md) covers inputs and outputs.

## Compile a pipeline

`Pipeline` is the semantic reference executor. `.compile()` returns an
immutable `CompiledPipeline` with the same transforms, seed, target signature,
call shape, and keyed results. Both expose `.transforms`, `.seed`, `.targets`,
and `.explain()`.

```python
import numpy as np
import variopinta as vp

pipeline = vp.Pipeline(
    [vp.RandomCrop(224, 224), vp.HorizontalFlip(p=0.5), vp.Normalize()],
    seed=42,
)
compiled = pipeline.compile()
image = np.random.default_rng(0).integers(0, 256, (320, 320, 3), dtype=np.uint8)

reference = pipeline(image, key=17)
optimized = compiled(image, key=17)
assert np.array_equal(reference, optimized)
```

Compilation selects kernel forms, reusable buffers, and output layouts for the
complete augmentation pipeline. For example, supported leading operations can
write directly from the input into owned output, avoiding an initial input
copy. A tensor-only image target ending in `Normalize` can produce normalized
CHW data directly, avoiding a separate layout conversion. See
[tensor output](pipelines-and-targets.md#return-a-tensor).

Input and output choices also determine the work around augmentation.
[Encoded routes](pipelines-and-targets.md#encoded-request-and-response) keep
decode, augmentation, and encode in one native call. With
[multiple outputs](pipelines-and-targets.md#multiple-outputs), each target is
transformed once before its final raster is delivered to every output.

## Control randomness

A pipeline has an unsigned 64-bit seed. An explicit unsigned 64-bit `key`
identifies a run independently of call order or worker assignment. The
following example continues with `compiled` and `image` from above:

```python
keys = [100, 101, 102]
forward = {key: compiled(image, key=key) for key in keys}
reverse = {key: compiled(image, key=key) for key in reversed(keys)}

assert all(np.array_equal(forward[key], reverse[key]) for key in keys)
```

Assign keys from stable sample identifiers when results must survive reordered
requests or retries. Reuse a key to replay a sample; choose a different key to
request another random draw. Distinct keys need not produce distinct pixels,
for example when a transform is skipped or the input is uniform.

Repeated keyed calls are deterministic for the same pipeline, input, seed,
installed release, and execution environment. Exact replay is not guaranteed
across releases, builds, or platforms; pin the full environment when bit-exact
results matter. Reference and compiled execution share Variopinta's sampling
and numeric semantics, which do not promise pixel or random-stream identity
with another library.

Without a key, successful calls advance the sequence associated with the
pipeline's seed. Validation, acquisition, execution, encoding, or delivery
failures do not consume a sequence position. Use explicit keys when scheduling
must not affect which augmentation a sample receives.

## Share a compiled pipeline

Compiled pipelines are safe to share across workers. Each run owns its random
state and working buffers; explicit keys make results independent of worker
assignment.

A call containing any `Array` carrier retains the Python GIL during aggregate
augmentation. Calls whose inputs are entirely `Encoded` or `Path` release it
through acquisition, augmentation, encoding, and delivery. Calls without a key
serialize their sequence commit so a failed call does not consume a key.

## Inspect the execution plan

`explain()` returns a structured report from the execution plan used at runtime.
Compare a reference pipeline with its compiled version to see a concrete
optimization:

```python
import variopinta as vp

pipeline = vp.Pipeline([vp.VerticalFlip(p=1)], seed=42)
for executor in (pipeline, pipeline.compile()):
    plan = executor.explain()
    step = plan["steps"][0]
    print(plan["mode"], step["input_materialization"], step["kernel_form"])
```

```text
reference owned-input-copy owned-in-place
compiled borrowed-input borrowed-to-owned
```

The reference executor copies the input before flipping it in place. The
compiled executor writes the flipped pixels directly into owned output. Both
leave the input unchanged and produce the same result; compilation removes the
initial copy in this example.

| Field | What to inspect |
|---|---|
| `steps` | Each transform's execution status, kernel form, pixel passes, and selection reason |
| `optimizations`, `fusions`, `unit_specializations` | Selected pipeline optimizations, cross-transform fusions, and individual kernel specializations |
| `targets` | Input carriers, outputs, buffers, copies, dtypes, layouts, and delivery |
| `python_boundary` | Native crossings and whether augmentation retains the GIL |
| `fallbacks` | Portable and architecture-dependent kernel routes |

Operations are marked `always`, `conditional`, or `never`; exact `p=0` routes
report only work that can execute. The report distinguishes transform passes
from output delivery, including direct normalized CHW production and a
terminal HWC-to-CHW copy when direct production is unavailable.

This is a static plan, not a timing profile or a trace of one random draw.
Read copy counts together with their conditions. Declaration order controls
target and output introspection; it does not constrain keyword call order.
Runtime arrays, tensors, encoded contents, source paths, and destination paths
are never inspected or included.

## Evaluate performance

Use the plan to identify copies and conversions, then measure the complete
route you intend to run. Latency depends on image sizes, transforms, input and
output formats, platform, and competing library configuration. Fewer copies
alone do not establish a speedup for every pipeline.

The controlled [benchmark harness](../benchmarks/) and
[recorded evidence](../benchmarks/evidence/) include case definitions,
environment fingerprints, validation results, and raw timings. Comparisons
apply to the tested configurations.
