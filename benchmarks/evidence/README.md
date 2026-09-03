# Canonical benchmark evidence

This tree stores one raw JSON shard for every case in the benchmark registry.
The registry defines the expected paths and completeness matrix; no separate
artifact manifest is maintained.

Use `just evidence --case CASE_ID` to renew complete cases, `just evidence
--stale` to renew missing or invalidated cases, and `just evidence-full` before
publishing release performance evidence. Diagnostic selections and all derived
reports are written below the ignored `benchmarks/.runs/` directory.

Each shard retains the normalized case definition, scoped fingerprints,
environment metadata, execution order, validation results, and every timing
observation. Reports and CSV files must be regenerated from these shards.
