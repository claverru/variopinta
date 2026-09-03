# Canonical benchmark evidence

Canonical results are stored as one JSON shard per benchmark case. Each shard contains the case definition, source and environment fingerprints, execution metadata, validation results, and raw timings.

For a transform change, renew only the cases that include that transform with `just evidence --case CASE_ID`. The recorded revision identifies where a result came from; it does not make unrelated results invalid.

`just evidence --stale` renews every result reported as stale, while `just evidence-full` rebuilds the complete evidence set for a release.

Diagnostic runs and derived reports are written to the ignored `benchmarks/.runs/` directory.
