from __future__ import annotations

import re

from benchmarks.common import FOCUSED_CASES, FOCUSED_LABELS, PIPELINES, TRANSFORMS
from benchmarks.model import CaseSpec, RouteSpec, TimingPolicy

TV = RouteSpec("torchvision.stock", "torchvision", "stock", "torchvision")
AX = RouteSpec("albumentationsx.stock", "albumentationsx", "stock", "albumentationsx")
RUST_FRESH = RouteSpec(
    "variopinta.staged-fresh", "variopinta", "staged-fresh", "rust", "attribution"
)
RUST_REUSE = RouteSpec(
    "variopinta.staged-reuse", "variopinta", "staged-reuse", "rust", "attribution"
)
RUST_COMPILED = RouteSpec("variopinta.compiled", "variopinta", "compiled", "rust")
RUST_REFERENCE = RouteSpec("variopinta.reference", "variopinta", "reference", "rust", "control")
PILLOW = RouteSpec("pillow.stock", "pillow", "stock", "io")
OPENCV = RouteSpec("opencv.stock", "opencv", "stock", "io")
OPENCV_CONTROL = RouteSpec("opencv.cross-control", "opencv", "cross-control", "io", "control")
IO_RUST = RouteSpec("variopinta.stock", "variopinta", "stock", "io")
IO_CONTRACT = RouteSpec(
    "variopinta.interoperability", "variopinta", "interoperability", "io", "control"
)

DEFAULT_TIMING = TimingPolicy(100.0, 3, 7, 512)
PIPELINE_TIMING = TimingPolicy(150.0, 8, 7, 1024, block_size=16)
IO_TIMING = TimingPolicy(100.0, 3, 7, 256)


def _slug(value: str) -> str:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value).replace("_", "-")
    return words.lower()


FOCUSED_IDS = {
    "affine-reflect101": "transforms.affine.bilinear-reflect101",
    "rotation-reflect101": "transforms.random-rotation.bilinear-reflect101",
    "gaussian-noise-independent": "transforms.gaussian-noise.independent-rgb",
    "gaussian-noise-shared": "transforms.gaussian-noise.shared-rgb",
    "color-jitter-hue": "transforms.color-jitter.hue",
    "sharpen-cross": "transforms.sharpen.cross-3x3",
    "perspective-bilinear": "transforms.perspective.bilinear-constant",
    "perspective-bilinear-reflect101": "transforms.perspective.bilinear-reflect101",
    "perspective-nearest-constant": "transforms.perspective.nearest-constant",
    "perspective-nearest-reflect101": "transforms.perspective.nearest-reflect101",
    "grid-distortion-bilinear": "transforms.grid-distortion.bilinear-constant",
    "grid-distortion-bilinear-reflect101": "transforms.grid-distortion.bilinear-reflect101",
    "grid-distortion-nearest-constant": "transforms.grid-distortion.nearest-constant",
    "grid-distortion-nearest-reflect101": "transforms.grid-distortion.nearest-reflect101",
    "pad-constant": "transforms.pad-if-needed.constant",
    "pad-reflect101": "transforms.pad-if-needed.reflect101",
    "return-tensor": "outputs.return-tensor.contiguous-uint8-chw",
}

FOCUSED_PARTICIPANTS = {
    "affine-reflect101": (AX,),
    "rotation-reflect101": (AX,),
    "gaussian-noise-independent": (TV, AX),
    "gaussian-noise-shared": (AX,),
    "color-jitter-hue": (TV, AX),
    "sharpen-cross": (AX, OPENCV_CONTROL),
    "perspective-bilinear": (TV, AX),
    "perspective-bilinear-reflect101": (AX,),
    "perspective-nearest-constant": (TV, AX),
    "perspective-nearest-reflect101": (AX,),
    "grid-distortion-bilinear": (AX,),
    "grid-distortion-bilinear-reflect101": (AX,),
    "grid-distortion-nearest-constant": (AX,),
    "grid-distortion-nearest-reflect101": (AX,),
    "pad-constant": (TV, AX),
    "pad-reflect101": (TV, AX),
    "return-tensor": (TV, AX),
}

CATALOG_POLICIES = (
    ("Resize", "bilinear"),
    ("Resize", "bilinear-antialias"),
    ("RandomCrop", "default"),
    ("RandomResizedCrop", "bilinear"),
    ("RandomResizedCrop", "bilinear-antialias"),
    ("CenterCrop", "default"),
    ("PadIfNeeded", "constant"),
    ("PadIfNeeded", "reflect101"),
    ("CoarseDropout", "eight-holes"),
    ("HorizontalFlip", "default"),
    ("VerticalFlip", "default"),
    ("ColorJitter", "matrix"),
    ("ColorJitter", "hue"),
    ("Affine", "constant"),
    ("Affine", "reflect101"),
    ("RandomRotation", "constant"),
    ("RandomRotation", "reflect101"),
    ("GaussianNoise", "independent-rgb"),
    ("GaussianNoise", "shared-rgb"),
    ("Sharpen", "cross-3x3"),
    ("Perspective", "bilinear"),
    ("Perspective", "nearest-reflect101"),
    ("GridDistortion", "bilinear"),
    ("GridDistortion", "nearest-reflect101"),
    ("GaussianBlur", "fixed-sigma"),
    ("GaussianBlur", "sampled-sigma"),
    ("Grayscale", "default"),
    ("Invert", "default"),
    ("Solarize", "default"),
    ("Posterize", "default"),
    ("Normalize", "default"),
)

IO_OPERATIONS = (
    "decode",
    "read",
    "encode",
    "write",
    "pipeline-three-call-encoded",
    "pipeline-three-call-path",
    "pipeline-encoded-return",
    "pipeline-array-encoded",
    "pipeline-encoded-encoded",
    "pipeline-path-path",
)


def _transform_cases() -> list[CaseSpec]:
    cases = []
    for name in TRANSFORMS:
        cases.append(
            CaseSpec(
                id=f"transforms.{_slug(name)}.default",
                suite="transforms",
                label=name,
                tags=("transform", f"transform:{name}", "comparison"),
                routes=(TV, AX, RUST_FRESH, RUST_REUSE, RUST_COMPILED),
                sizes=(224, 512, 1024),
                executor="layers",
                factory=f"transform:{name}",
                comparability="policy",
                scopes=("transforms", "variopinta"),
                timing=DEFAULT_TIMING,
            )
        )
    cases.append(
        CaseSpec(
            id="transforms.resize.bilinear-antialias",
            suite="transforms",
            label="Resize (bilinear, antialias)",
            tags=("transform", "transform:Resize", "policy:antialias"),
            routes=(TV, RUST_COMPILED),
            sizes=(224, 512, 1024),
            executor="layers",
            factory="transform-antialias:Resize",
            comparability="policy",
            scopes=("transforms", "variopinta"),
            timing=DEFAULT_TIMING,
        )
    )
    for focused in FOCUSED_CASES:
        case_id = FOCUSED_IDS[focused]
        suite = case_id.partition(".")[0]
        category = "output" if suite == "outputs" else "transform"
        cases.append(
            CaseSpec(
                id=case_id,
                suite=suite,
                label=FOCUSED_LABELS[focused],
                tags=(
                    category,
                    f"{category}:{FOCUSED_LABELS[focused].split(' ')[0]}",
                    "focused",
                ),
                routes=(
                    *FOCUSED_PARTICIPANTS[focused],
                    RUST_FRESH,
                    RUST_REUSE,
                    RUST_COMPILED,
                ),
                sizes=(224, 512, 1024),
                executor="layers",
                factory=f"focused:{focused}",
                comparability="operational",
                scopes=(suite, "variopinta"),
                timing=DEFAULT_TIMING,
            )
        )
    return cases


def _pipeline_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            id=f"pipelines.{name.replace('_', '-')}",
            suite="pipelines",
            label=name.replace("_", " ").title(),
            tags=("pipeline", f"pipeline:{name}"),
            routes=(TV, AX, RUST_FRESH, RUST_REUSE, RUST_COMPILED),
            sizes=(224, 512, 1024),
            executor="layers",
            factory=f"pipeline:{name}",
            comparability="policy",
            scopes=("pipelines", "transforms", "variopinta"),
            timing=PIPELINE_TIMING,
        )
        for name in PIPELINES
    ]


def _catalog_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            id=f"catalog.{_slug(name.replace('+', '-'))}.{_slug(policy)}",
            suite="catalog",
            label=f"{name} ({policy})",
            tags=("catalog", f"transform:{name.split('+')[0]}", f"policy:{policy}"),
            routes=(RUST_REFERENCE, RUST_COMPILED),
            sizes=(224, 512, 1024),
            executor="catalog",
            factory=f"{name}|{policy}",
            comparability="exact",
            scopes=("catalog", "variopinta"),
            timing=DEFAULT_TIMING,
        )
        for name, policy in CATALOG_POLICIES
    ]


def _io_cases() -> list[CaseSpec]:
    cases = [
        CaseSpec(
            id="io.interoperability",
            suite="io",
            label="Codec interoperability",
            tags=("io", "validation", "contract"),
            routes=(IO_CONTRACT,),
            sizes=(),
            executor="io-parity",
            factory="interoperability",
            comparability="control",
            scopes=("io", "variopinta-io", "variopinta"),
            timing=None,
        )
    ]
    for format_name in ("jpeg", "png"):
        for operation in IO_OPERATIONS:
            routes = (
                (IO_RUST, PILLOW, OPENCV)
                if operation in {"decode", "read", "encode", "write"}
                else (IO_RUST,)
            )
            cases.append(
                CaseSpec(
                    id=f"io.{format_name}.{operation}",
                    suite="io",
                    label=f"{format_name.upper()} {operation}",
                    tags=("io", f"format:{format_name}", f"operation:{operation}"),
                    routes=routes,
                    sizes=(512,),
                    executor="io-performance",
                    factory=f"{format_name}|{operation}",
                    comparability="operational",
                    scopes=("io", "variopinta-io", "variopinta"),
                    timing=IO_TIMING,
                )
            )
    return cases


def _contract_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            id="contracts.augmentation-boundaries",
            suite="contracts",
            label="Augmentation boundary contracts",
            tags=("contract", "layout", "dtype", "ownership", "dimensions"),
            routes=(TV, AX, RUST_COMPILED),
            sizes=(),
            executor="contracts",
            factory="augmentation-boundaries",
            comparability="control",
            scopes=("contracts", "transforms", "variopinta"),
            timing=None,
        )
    ]


CASES = tuple(
    sorted(
        [
            *_transform_cases(),
            *_pipeline_cases(),
            *_catalog_cases(),
            *_io_cases(),
            *_contract_cases(),
        ],
        key=lambda case: case.id,
    )
)
CASE_BY_ID = {case.id: case for case in CASES}

if len(CASE_BY_ID) != len(CASES):
    raise RuntimeError("duplicate benchmark case identifier")
