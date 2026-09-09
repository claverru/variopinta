# Canonical benchmark evidence

Canonical results are stored as one JSON shard per benchmark case. Each shard contains the case definition, source and environment fingerprints, execution metadata, validation results, and raw timings.

Run benchmarks before committing, then commit the code and results together. Evidence freshness depends on fingerprints of measured file contents, case definitions, and environment configuration, regardless of Git state. Collection checks fingerprints before and after execution and refuses to save results if measured inputs changed during the run.

After changing a transform, run only its affected benchmark cases, including catalog cases, pipelines, and contracts that use it:

```bash
just evidence --tag transform:GaussianBlur
```

Use `just evidence --case CASE_ID` for one exact case, or `just evidence --stale` to renew every missing, invalid, or stale case. Transform tags include dependent pipelines and contracts. `just evidence-full` rebuilds the complete evidence set for a release.

Fingerprints track native kernel files by their transform dependencies. A shared kernel change affects every transform that uses that file. Shared compiler, binding, API, operation, or harness files require broader reruns; unknown dependencies retain the full source scope. Keep the case dependencies in `benchmarks/registry.py` and kernel mapping in `benchmarks/fingerprints.py` current when adding or changing execution paths.

Git tracks result history; use `git log -1 -- benchmarks/evidence/PATH.json` to find a shard's latest commit. Shards omit the Git revision and dirty flag; those fields in older shards are ignored.

Bookkeeping changes alone do not require new measurements. The transition to scoped kernel fingerprints uses `scripts/migrate_benchmark_evidence.py` to verify the old fingerprints against the previous code, confirm that measured inputs still match, and update metadata while preserving observations. Changed measured inputs require rerunning the affected cases.

Diagnostic runs and derived reports are written to the ignored `benchmarks/.runs/` directory.
