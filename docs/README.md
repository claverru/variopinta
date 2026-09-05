# Variopinta documentation

Start with a working CPU augmentation pipeline, then learn how to integrate
its inputs and outputs, reproduce results, and inspect its execution plan.

## Read in this order

1. [Getting started](getting-started.md) — install and run your first compiled
   pipeline.
2. [Pipelines and targets](pipelines-and-targets.md) — connect images, masks,
   files, encoded data, and tensor or multiple outputs in one pipeline.
3. [Compile, reproduce, and inspect](execution.md) — understand compilation,
   assign deterministic keys, share a pipeline, and interpret `explain()`.

Use [Image I/O](image-io.md) for standalone JPEG and PNG helpers and
[Transform reference](transforms.md) for constructors, defaults, and image/mask
behavior.

## Choose the smallest API that fits

| Task | Start with |
|---|---|
| NumPy image in, NumPy image out | The [default pipeline](getting-started.md#default-image-to-image-pipeline) |
| Image and semantic mask together | [Explicit targets](pipelines-and-targets.md#explicit-targets) |
| Return a PyTorch tensor | [Return a tensor](pipelines-and-targets.md#return-a-tensor) |
| Multiple outputs from one target | [Multiple outputs](pipelines-and-targets.md#multiple-outputs) |
| Decode, augment, and encode in one native call | [Encoded request and response](pipelines-and-targets.md#encoded-request-and-response) |
| Read, transform, and write local files | [Write bindings](pipelines-and-targets.md#write-bindings) |
| Replay an augmentation regardless of worker order | [Control randomness](execution.md#control-randomness) |
| Inspect copies, buffers, and selected optimizations | [Inspect the execution plan](execution.md#inspect-the-execution-plan) |
| Standalone JPEG or PNG conversion | [Image I/O](image-io.md) |
| Constructor or parameter lookup | [Transform reference](transforms.md) |

This versioned documentation describes the current public API. Contributor and
architecture notes live under `docs_internal/` and are not part of the public
contract.
