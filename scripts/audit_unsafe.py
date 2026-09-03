from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "rust/core/src/kernels/affine.rs": (
        4,
        [
            "optimized_kernel_matches_safe_oracle_at_boundaries",
            "reflected_rgb_interpolation_matches_scalar",
        ],
    ),
    "rust/core/src/kernels/blur.rs": (4, ["optimized_passes_match_scalar_at_vector_boundaries"]),
    "rust/core/src/kernels/color.rs": (
        4,
        [
            "optimized_kernel_matches_scalar_at_vector_boundaries",
            "hue_dispatch_matches_scalar_at_vector_and_sector_boundaries",
        ],
    ),
    "rust/core/src/kernels/layout.rs": (
        21,
        [
            "avx2_normalize_matches_scalar",
            "u8_chw_dispatch_matches_scalar_for_unaligned_sources_and_tails",
        ],
    ),
    "rust/core/src/kernels/noise.rs": (
        2,
        ["application_dispatch_matches_scalar_at_vector_boundaries"],
    ),
    "rust/core/src/kernels/point.rs": (10, ["point_kernels_handle_scalar_tails"]),
    "rust/core/src/kernels/remap.rs": (2, ["perspective_descriptor_dispatch_matches_scalar"]),
    "rust/core/src/kernels/sharpen.rs": (
        2,
        ["dispatch_matches_scalar_for_arbitrary_rectangles_and_coefficients"],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory local unsafe blocks and their oracles")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    inventory = []
    discovered: dict[str, int] = {}
    for base in (ROOT / "rust/core", ROOT / "rust/io", ROOT / "rust/pyext"):
        for path in sorted(base.rglob("*.rs")):
            relative = str(path.relative_to(ROOT))
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if not re.search(r"\bunsafe\s*\{", line):
                    continue
                context = " ".join(lines[max(0, index - 3) : index + 1])
                if "SAFETY:" not in context:
                    raise SystemExit(f"{relative}:{index + 1} has no adjacent SAFETY invariant")
                discovered[relative] = discovered.get(relative, 0) + 1
                inventory.append(
                    {
                        "file": relative,
                        "line": index + 1,
                        "safety": context.split("SAFETY:", 1)[1].strip(),
                    }
                )

    expected_counts = {path: count for path, (count, _oracles) in EXPECTED.items()}
    if discovered != expected_counts:
        raise SystemExit(
            f"unsafe inventory changed: expected {expected_counts}, found {discovered}"
        )
    for relative, (_count, oracles) in EXPECTED.items():
        source = (ROOT / relative).read_text()
        for oracle in oracles:
            if oracle not in source:
                raise SystemExit(f"{relative} is missing scalar oracle {oracle}")

    payload = {
        "blocks": inventory,
        "files": len(discovered),
        "status": "pass",
        "total_blocks": len(inventory),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
