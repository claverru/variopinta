from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("torchvision", "albumentations", "albumentationsx", "rust")


def run(*command: str, environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def install(python: Path, requirements: Path, wheelhouse: Path | None) -> None:
    command = [str(python), "-m", "pip", "install"]
    if wheelhouse is not None:
        command.extend(("--find-links", str(wheelhouse)))
    command.extend(("-r", str(requirements)))
    run(*command)


def create_environment(
    name: str,
    root: Path,
    wheelhouse: Path | None,
    *,
    recreate: bool,
    system_site_packages: bool,
) -> None:
    path = root / name
    if recreate and path.exists():
        shutil.rmtree(path)
    if not (path / "bin" / "python").is_file():
        venv.EnvBuilder(with_pip=True, system_site_packages=system_site_packages).create(path)
    python = path / "bin" / "python"
    install(python, ROOT / "requirements" / "benchmarks.txt", wheelhouse)
    if name in {"albumentations", "albumentationsx"}:
        install(python, ROOT / "requirements" / f"{name}.txt", wheelhouse)
    if name == "rust":
        install(python, ROOT / "requirements" / "dev.txt", wheelhouse)
        environment = os.environ.copy()
        environment["VIRTUAL_ENV"] = str(path)
        environment["PATH"] = os.pathsep.join((str(path / "bin"), environment.get("PATH", "")))
        maturin = path / "bin" / "maturin"
        if not maturin.is_file():
            inherited = shutil.which("maturin", path=environment["PATH"])
            if inherited is None:
                raise SystemExit("Maturin is missing from the rust benchmark environment")
            maturin = Path(inherited)
        run(
            str(maturin),
            "develop",
            "--release",
            "--locked",
            "--manifest-path",
            "rust/pyext/Cargo.toml",
            environment=environment,
        )
    run(str(python), "-m", "pip", "check")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create isolated benchmark environments")
    parser.add_argument(
        "--env-root",
        type=Path,
        default=Path(os.environ.get("VARIOPINTA_BENCH_ENV_ROOT", ROOT / ".venvs" / "benchmarks")),
    )
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "wheelhouse")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--system-site-packages", action="store_true")
    args = parser.parse_args()

    wheelhouse = args.wheelhouse.resolve() if args.wheelhouse.is_dir() else None
    environment_root = args.env_root.resolve()
    environment_root.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        create_environment(
            backend,
            environment_root,
            wheelhouse,
            recreate=args.recreate,
            system_site_packages=args.system_site_packages,
        )
    print(f"Benchmark environments ready in {environment_root}")


if __name__ == "__main__":
    main()
