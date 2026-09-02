set shell := ["bash", "-cu"]

python := env_var_or_default("PYTHON", "python")
maturin := env_var_or_default("MATURIN", "maturin")
ruff := env_var_or_default("RUFF", "ruff")
cargo_manifest := "rust/Cargo.toml"
python_sources := "python benchmarks scripts tests"

# List available recipes.
default:
    @just --list

# Install development tools and build the extension in the active environment.
setup:
    {{ python }} -m pip install --upgrade pip
    {{ python }} -m pip install -r requirements/dev.txt
    {{ maturin }} develop --release --locked --manifest-path rust/pyext/Cargo.toml

# Build the extension in the active environment.
develop:
    {{ maturin }} develop --release --locked --manifest-path rust/pyext/Cargo.toml

# Format Rust and Python sources.
format:
    cargo fmt --manifest-path {{ cargo_manifest }} --all
    {{ ruff }} format {{ python_sources }}

# Apply safe Python lint fixes, then format all sources.
fix:
    {{ ruff }} check --fix {{ python_sources }}
    just format

# Check Rust and Python formatting.
format-check:
    cargo fmt --manifest-path {{ cargo_manifest }} --all -- --check
    {{ ruff }} format --check {{ python_sources }}

# Run static analysis.
lint:
    {{ ruff }} check {{ python_sources }}
    cargo clippy --manifest-path {{ cargo_manifest }} --workspace --all-targets --locked -- -D warnings

# Audit native unsafe blocks. Extra arguments are passed to the audit script.
audit-unsafe *args:
    {{ python }} scripts/audit_unsafe.py {{ args }}

# Audit native unsafe blocks and retain the CI inventory.
audit-unsafe-evidence:
    {{ python }} scripts/audit_unsafe.py --output unsafe-evidence/inventory.json

# Run the locked Rust workspace tests.
test-rust:
    cargo test --manifest-path {{ cargo_manifest }} --workspace --all-targets --locked

# Check Python syntax without importing the extension.
python-syntax:
    {{ python }} -m compileall -q {{ python_sources }}

# Build the extension, run Python tests, and audit catalog correctness.
python-check: python-syntax develop
    {{ python }} -m unittest discover -s tests
    {{ python }} benchmarks/run_catalog_audit.py --output results/raw/catalog-audit-local.json

# Run all ordinary tests.
test: test-rust python-check

# Run every local gate used before completion.
check: format-check lint audit-unsafe test

# Run focused catalog registration and boundary conformance checks.
catalog-conformance: develop
    cargo test --manifest-path {{ cargo_manifest }} -p augment-core --locked catalog
    {{ python }} -m unittest tests.test_catalog.CatalogTests.test_transform_catalog_sets_and_binding_support_are_exact
    {{ python }} -m unittest tests.test_catalog.CatalogTests.test_transform_catalog_conformance_matrix

# Run the x86-64 core suite under AddressSanitizer.
asan:
    RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zsanitizer=address" cargo test --manifest-path {{ cargo_manifest }} -p augment-core --target x86_64-unknown-linux-gnu --locked

# Create the isolated benchmark environments. Extra arguments are forwarded.
benchmark-setup *args:
    {{ python }} scripts/setup_benchmark_envs.py {{ args }}

# Run the complete comparative benchmark.
benchmark:
    {{ python }} benchmarks/run_benchmark.py

# Run the diagnostic comparative benchmark.
benchmark-quick:
    {{ python }} benchmarks/run_benchmark.py --quick

# Regenerate the report and plots from committed observations.
benchmark-render:
    {{ python }} benchmarks/run_benchmark.py --render-existing

# Run codec interoperability checks.
io-parity:
    {{ python }} benchmarks/run_io_parity.py

# Measure focused codec performance.
io-performance:
    {{ python }} benchmarks/run_io_performance.py

# Run backend and optimization attribution experiments.
layer-experiments:
    {{ python }} benchmarks/run_layer_experiments.py

# Run the exhaustive transform-catalog correctness audit.
catalog-audit: develop
    {{ python }} benchmarks/run_catalog_audit.py

# Run the full adaptive catalog benchmark.
catalog-benchmark:
    {{ python }} benchmarks/run_catalog_benchmark.py

# Run the diagnostic adaptive catalog benchmark.
catalog-benchmark-quick:
    {{ python }} benchmarks/run_catalog_benchmark.py --quick

# Regenerate all full benchmark evidence in dependency order.
evidence:
    just io-parity
    just io-performance
    just catalog-audit
    just catalog-benchmark
    just layer-experiments
    just benchmark

# Check whether canonical benchmark evidence matches the current measured code.
evidence-status:
    {{ python }} benchmarks/check_evidence.py

# Build one host-platform wheel and one sdist into a new release-dist directory.
release-build:
    test ! -e release-dist
    mkdir release-dist
    {{ maturin }} sdist --manifest-path rust/pyext/Cargo.toml --out release-dist
    TURBOJPEG_SOURCE=vendor TURBOJPEG_STATIC=1 TURBOJPEG_BINDING=pregenerated {{ maturin }} build --release --locked --compatibility pypi --out release-dist
