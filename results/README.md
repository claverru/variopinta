# Benchmark evidence

This directory keeps only canonical, reviewable evidence. Intermediate worker files, quick
diagnostics, CSV exports, plots, and rendered reports are generated locally and ignored.

## Reusing evidence across releases

Every newly generated canonical JSON artifact records a benchmark fingerprint, the Git revision,
and whether the measured source set was dirty. The fingerprint covers the implementation, bindings,
benchmark harness, locked dependencies, and benchmark-environment setup. It deliberately
normalizes only the Variopinta package version fields in `pyproject.toml`, the Rust workspace
manifest, and local workspace entries in `Cargo.lock`.

Consequently, a release-only version bump does not make otherwise identical measurements stale.
Use `just evidence-status` to compare stored fingerprints with the current tree:

- `current` means the evidence may be reused while retaining its original measured package and
  environment metadata.
- `stale` means relevant code or benchmark configuration changed and the artifact must be rerun.
- `legacy` means the artifact predates fingerprints and must be regenerated before claiming that
  it represents the current implementation.
- `dirty` means the run was produced from modified measured sources and should not be committed as
  canonical evidence.
- `missing` means the corresponding benchmark has not produced a canonical artifact.

Run `just evidence` from the isolated benchmark environments to regenerate the full set. Focused
recipes remain available for individual artifacts. In particular, `just io-performance` measures
standalone decode/read/encode/write operations and the supported pipeline source/sink routes; no
separate pipeline-I/O benchmark is needed.

Raw observations are the source of truth. CSV files, Markdown reports, and plots are derived and
must be regenerated rather than edited or copied forward. Quick runs are diagnostic only and are
never committed.
