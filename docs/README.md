# Variopinta documentation

Variopinta builds CPU image-augmentation pipelines in Python and runs them in
Rust. Start with the common NumPy workflow, then add explicit targets only when
you need masks, encoded data, files, tensors, or more than one output.

## Read in this order

1. [Getting started](getting-started.md) — install Variopinta, run a pipeline,
   add a mask, and choose deterministic keys.
2. [Pipelines and targets](pipelines-and-targets.md) — define input carriers and
   output ports, read results, reason about ownership and concurrency, and
   inspect compiled plans.
3. [Image I/O](image-io.md) — decode, encode, read, and write JPEG and PNG data,
   including resource limits and format restrictions.
4. [Transform reference](transforms.md) — look up every constructor, default,
   range, and image/mask behavior.

## Choose the smallest API that fits

| Task | Start with |
|---|---|
| NumPy image in, NumPy image out | The [default pipeline](getting-started.md#default-image-to-image-pipeline) |
| Image and semantic mask together | [Image and mask](getting-started.md#add-a-semantic-mask) |
| Tensor, bytes, path, or multiple outputs | [Explicit targets](pipelines-and-targets.md#explicit-targets) |
| Standalone JPEG or PNG conversion | [Image I/O](image-io.md) |
| Constructor or parameter lookup | [Transform reference](transforms.md) |

## Guarantees and scope

The reference and compiled executors produce the same result for the same
pipeline, input, seed, and key. Variopinta defines its own random sampling,
interpolation, border, rounding, and clipping semantics; it does not promise
pixel or random-stream identity with another library.

Repeated keyed calls are deterministic for the same installed release,
execution environment, pipeline, input, seed, and key. Exact replay is not
guaranteed across releases, builds, or platforms, so pin the full environment
when bit-exact results matter.

Performance depends on the complete route, image sizes, transforms, output
materialization, platform, and competing library configuration. The controlled
[benchmark harness](../benchmarks/) records equivalent work,
correctness checks, copies, buffer plans, hardware, and statistical limits. Its
results support specific tested configurations, not a universal Rust speed
claim.

This versioned documentation describes the current public API. Contributor and
architecture notes live under `docs_internal/` and are not part of the public
contract.
