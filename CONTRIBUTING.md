# Contributing

Variopinta is experimental and its public API is still being defined. Open an
issue before substantial API, dependency, architecture, or benchmark work so
the intended contract can be agreed before implementation.

## Development workflow

Use CPython 3.10–3.13, Rust 1.87 or newer, `just`, a C/C++ toolchain, and CMake.
Linux x86-64 also requires NASM; native Apple Silicon builds use Xcode
command-line tools and do not require NASM. Create an ordinary virtual
environment and install the development dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
just setup
```

Run the complete local gate before submitting a change:

```bash
just check
```

Keep changes focused. Update tests and the canonical documentation with the
implementation. Do not commit virtual environments, build outputs, local
wheels, quick benchmark runs, or intermediate result files.

Performance claims require the controlled benchmark harness and committed
evidence under [`benchmarks/`](benchmarks/) and [`results/`](results/). Ordinary
changes do not need to regenerate performance data unless they affect measured
work or published results.

By submitting a contribution, you agree that it is licensed under the
[Apache License 2.0](LICENSE).
