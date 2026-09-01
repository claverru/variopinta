from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = Path(
    os.environ.get("VARIOPINTA_BENCH_ENV_ROOT", ROOT / ".venvs" / "benchmarks")
).expanduser()


def environment_executable(environment: str, executable: str) -> Path:
    return ENV_ROOT / environment / "bin" / executable


def python_for(backend: str) -> Path:
    return environment_executable(backend, "python")


def require_environments(backends: Iterable[str]) -> None:
    missing = sorted(backend for backend in set(backends) if not python_for(backend).is_file())
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"Missing benchmark environments: {names}. Run `just benchmark-setup` first."
        )


def rebuild_variopinta() -> None:
    require_environments(("rust",))
    rust_environment = ENV_ROOT / "rust"
    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = str(rust_environment)
    environment["PATH"] = os.pathsep.join(
        (str(rust_environment / "bin"), environment.get("PATH", ""))
    )
    maturin = environment_executable("rust", "maturin")
    if not maturin.is_file():
        inherited = shutil.which("maturin", path=environment["PATH"])
        if inherited is None:
            raise SystemExit("Maturin is missing from the rust benchmark environment")
        maturin = Path(inherited)
    subprocess.run(
        [
            str(maturin),
            "develop",
            "--release",
            "--locked",
            "--manifest-path",
            str(ROOT / "rust" / "pyext" / "Cargo.toml"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
