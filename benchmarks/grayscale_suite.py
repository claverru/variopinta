from __future__ import annotations

import json
import subprocess
from typing import Any

import numpy as np
from common import ROOT, SEED, output_facts, time_calls_adaptive


def run_planned(
    items: list[dict[str, Any]], quick: bool, repetition: int, *, validate_only: bool = False
) -> list[dict[str, Any]]:
    import variopinta as vp

    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(ROOT / "rust/Cargo.toml"),
            "-p",
            "augment-core",
            "--example",
            "grayscale_memory",
        ],
        check=True,
        capture_output=True,
    )
    rows = []
    for order, item in enumerate(items, 1):
        for size in item["sizes"]:
            case = item["factory"]
            out = size * 3 // 4
            transforms = {
                "geometry": [vp.CenterCrop(size - 2, size - 2), vp.Resize(out, out)],
                "aspect-resize-pad": [
                    vp.LongestMaxSize(out),
                    vp.PadIfNeeded(min_height=out, min_width=out),
                ],
                "filtering": [vp.GaussianBlur(), vp.Sharpen()],
                "normalized-tensor": [vp.Resize(out, out), vp.Normalize(mean=0.5, std=0.5)],
            }[case]
            tensor = case == "normalized-tensor"
            port = vp.ReturnTensor(name="value") if tensor else vp.ReturnArray(name="value")
            target = vp.Image(name="image", output_specs=port)
            reference = vp.Pipeline(transforms, targets=target, seed=SEED)
            compiled = reference.compile()
            input_height = size * 2 // 3 if case == "aspect-resize-pad" else size
            images = [
                np.random.default_rng(SEED + i).integers(
                    0, 256, (input_height, size), dtype=np.uint8
                )
                for i in range(8)
            ]

            def run(image, *, expanded=False, pipeline=compiled, target=target, tensor=tensor):
                source = np.repeat(image[..., None], 3, axis=2) if expanded else image
                output = pipeline(image=target.bind(source), key=SEED).image.value
                if expanded:
                    return output[:1].clone() if tensor else output[..., 0].copy()
                return output

            expected = run(images[0], pipeline=reference)
            native = run(images[0])
            control = run(images[0], expanded=True)
            exact = np.array_equal(np.asarray(expected), np.asarray(native)) and np.array_equal(
                np.asarray(native), np.asarray(control)
            )
            route = item["route"]
            expanded = route["variant"] == "rgb-expansion"
            output = control if expanded else native
            measured = subprocess.run(
                [
                    str(ROOT / "rust/target/release/examples/grayscale_memory"),
                    str(size),
                    case,
                    route["variant"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            memory = [json.loads(line) for line in measured.stdout.splitlines()]
            copies = int(case == "filtering") + 2 * int(expanded)
            row = {
                "case_id": item["case_id"],
                "route_id": route["id"],
                "participant": route["participant"],
                "variant": route["variant"],
                "role": route["role"],
                "size": size,
                "repetition": repetition,
                "case_order": order,
                "reference_exact": exact,
                "validation": output_facts(output),
                "valid": bool(exact),
                "explanation": compiled.explain(),
                "native_memory": memory,
                "memory_scope": "requested live Rust heap bytes in instrumented core call; excludes Python/Torch, allocator overhead, transient realloc storage, and preexisting input/plan; warm peak excludes retained workspace",
                "full_raster_copy_count": copies,
                "copy_count_method": "selected native entry policy plus explicit RGB expansion and owned channel extraction; excludes geometry/filter writes",
            }
            if not validate_only:
                policy = dict(item["timing"])
                if quick:
                    policy.update(budget_ms=10.0, warmup_calls=2, min_samples=3, max_calls=64)
                timing, _ = time_calls_adaptive(
                    lambda image, expanded=expanded: run(image, expanded=expanded), images, **policy
                )
                row.update(timing)
            rows.append(row)
    return rows
