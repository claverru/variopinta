from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.common import ROOT, normalized_evidence_input
from benchmarks.model import CaseSpec
from benchmarks.registry import case_transforms

FRAMEWORK_PATHS = (
    "benchmarks/common.py",
    "benchmarks/model.py",
    "benchmarks/selection.py",
    "benchmarks/controller.py",
    "benchmarks/worker.py",
)

SCOPE_PATTERNS = {
    "grayscale": ("benchmarks/grayscale_suite.py", "rust/core/examples/grayscale_memory.rs"),
    "transforms": (
        "benchmarks/adapters.py",
        "benchmarks/layer_worker.py",
    ),
    "pipelines": (
        "benchmarks/adapters.py",
        "benchmarks/layer_worker.py",
    ),
    "outputs": (
        "benchmarks/adapters.py",
        "benchmarks/layer_worker.py",
    ),
    "catalog": ("benchmarks/catalog_suite.py",),
    "io": (
        "benchmarks/io_performance_worker.py",
        "benchmarks/io_parity_worker.py",
    ),
    "contracts": (
        "benchmarks/contract_suite.py",
        "benchmarks/correctness.py",
        "benchmarks/adapters.py",
    ),
    "variopinta": (
        "pyproject.toml",
        "python/variopinta/*.py",
        "rust/Cargo.lock",
        "rust/Cargo.toml",
        "rust/core/Cargo.toml",
        "rust/core/src/**/*.rs",
        "rust/pyext/Cargo.toml",
        "rust/pyext/src/**/*.rs",
    ),
    "variopinta-io": (
        "rust/io/Cargo.toml",
        "rust/io/src/**/*.rs",
    ),
}

NON_MEASUREMENT_BENCHMARK_FILES = {
    Path("benchmarks/__init__.py"),
    Path("benchmarks/__main__.py"),
    Path("benchmarks/cli.py"),
    Path("benchmarks/evidence.py"),
    Path("benchmarks/fingerprints.py"),
    Path("benchmarks/registry.py"),
    Path("benchmarks/views.py"),
}

ENVIRONMENT_PATTERNS = {
    "torchvision": ("requirements/benchmarks.txt",),
    "albumentationsx": (
        "requirements/benchmarks.txt",
        "requirements/albumentationsx.txt",
    ),
    "rust": (
        "requirements/benchmarks.txt",
        "requirements/dev.txt",
    ),
    "io": (
        "requirements/io.txt",
        "requirements/dev.txt",
    ),
}


TRANSFORM_KERNELS = {
    "Resize": (),
    "LongestMaxSize": (),
    "RandomCrop": (),
    "RandomResizedCrop": (),
    "CenterCrop": (),
    "CoarseDropout": (),
    "HorizontalFlip": ("point",),
    "VerticalFlip": ("point",),
    "Grayscale": ("point",),
    "Invert": ("point",),
    "Solarize": ("point",),
    "Posterize": ("point",),
    "ColorJitter": ("color",),
    "Affine": ("affine",),
    "RandomRotation": ("affine",),
    "GaussianBlur": ("blur",),
    "GaussianNoise": ("noise",),
    "PadIfNeeded": ("pad",),
    "Sharpen": ("sharpen",),
    "Perspective": ("remap", "affine"),
    "GridDistortion": ("remap", "affine"),
    "Normalize": ("layout",),
    "ReturnTensor": ("layout",),
}


def _unused_kernel_paths(case: CaseSpec) -> set[Path]:
    transforms = case_transforms(case)
    # Unknown dependencies retain the full source scope.
    if transforms is None or not transforms <= TRANSFORM_KERNELS.keys():
        return set()
    all_kernels = {kernel for kernels in TRANSFORM_KERNELS.values() for kernel in kernels}
    used = {kernel for name in transforms for kernel in TRANSFORM_KERNELS[name]}
    return {Path(f"rust/core/src/kernels/{kernel}.rs") for kernel in all_kernels - used}


def _paths_for_patterns(root: Path, patterns: tuple[str, ...]) -> set[Path]:
    return {
        path.relative_to(root)
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    }


def _digest_paths(root: Path, paths: set[Path]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(normalized_evidence_input(relative, (root / relative).read_bytes()))
        digest.update(b"\0")
    return digest.hexdigest()


def unclassified_measured_paths(root: Path = ROOT) -> tuple[Path, ...]:
    classified = _paths_for_patterns(
        root,
        (
            *FRAMEWORK_PATHS,
            *(pattern for patterns in SCOPE_PATTERNS.values() for pattern in patterns),
            "benchmarks/environments.py",
            "scripts/setup_benchmark_envs.py",
            *(pattern for patterns in ENVIRONMENT_PATTERNS.values() for pattern in patterns),
        ),
    )
    candidates = {
        *(
            path.relative_to(root)
            for path in (root / "benchmarks").glob("**/*.py")
            if path.is_file()
        ),
        *(
            path.relative_to(root)
            for path in (root / "requirements").glob("*.txt")
            if path.is_file()
        ),
    }
    return tuple(sorted(candidates - classified - NON_MEASUREMENT_BENCHMARK_FILES))


def environment_fingerprint(environment: str, root: Path = ROOT) -> str:
    patterns = (
        "scripts/setup_benchmark_envs.py",
        "benchmarks/environments.py",
        *ENVIRONMENT_PATTERNS[environment],
    )
    return _digest_paths(root, _paths_for_patterns(root, patterns))


def case_fingerprint(case: CaseSpec, root: Path = ROOT) -> dict[str, Any]:
    scope_patterns = tuple(pattern for scope in case.scopes for pattern in SCOPE_PATTERNS[scope])
    unclassified = unclassified_measured_paths(root)
    source_paths = _paths_for_patterns(root, (*FRAMEWORK_PATHS, *scope_patterns)) | set(
        unclassified
    )
    source_paths -= _unused_kernel_paths(case)
    environments = sorted({route.environment for route in case.routes})
    components = {
        "definition": hashlib.sha256(
            json.dumps(case.normalized(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source": _digest_paths(root, source_paths),
        "environments": {
            environment: environment_fingerprint(environment, root) for environment in environments
        },
        "unclassified": [path.as_posix() for path in unclassified],
    }
    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"digest": digest, "components": components}


def compatibility_signature(metadata: dict[str, Any]) -> dict[str, Any]:
    environments = {}
    for name, values in sorted(metadata.items()):
        environments[name] = {
            key: values.get(key)
            for key in (
                "python",
                "platform",
                "architecture",
                "processor",
                "cpu_count",
                "packages",
                "thread_control",
            )
        }
    digest = hashlib.sha256(
        json.dumps(environments, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"digest": digest, "environments": environments}
