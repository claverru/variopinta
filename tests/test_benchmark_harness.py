from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmarks import environments
from benchmarks.common import metadata, summarize_observations, time_calls_adaptive
from benchmarks.controller import validate_complete
from benchmarks.evidence import shard_path, status_for, write_json_atomic
from benchmarks.fingerprints import case_fingerprint, unclassified_measured_paths
from benchmarks.model import CaseSpec, PlannedCase, RouteSpec, TimingPolicy
from benchmarks.registry import CASES
from benchmarks.selection import Selectors, select_cases, validate_selector_values


class LayerWorkerTests(unittest.TestCase):
    def test_variopinta_routes_use_the_target_aware_native_api(self) -> None:
        script = """
import numpy as np
from layer_worker import _rust_apply

source = np.zeros((5, 7, 3), dtype=np.uint8)
for mode in ("reference", "compiled"):
    output = _rust_apply([{"type": "Invert", "p": 1.0}], mode)(source)
    assert output.shape == source.shape
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[1] / "benchmarks",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class AdaptiveTimingTests(unittest.TestCase):
    def test_call_count_is_bounded_and_reported(self) -> None:
        calls = 0

        def measured(value: int) -> int:
            nonlocal calls
            calls += 1
            return value

        timing, output = time_calls_adaptive(
            measured,
            [3, 7],
            budget_ms=1_000.0,
            warmup_calls=2,
            min_samples=3,
            max_calls=12,
            target_sample_ms=0.001,
        )

        self.assertIn(output, (3, 7))
        self.assertLessEqual(timing["iterations"], 12)
        self.assertGreaterEqual(timing["samples"], 3)
        self.assertEqual(timing["iterations"], timing["samples"] * timing["block_size"])
        self.assertEqual(timing["samples"], len(timing["observations_ms"]))
        self.assertEqual(
            timing["median_ms"], summarize_observations(timing["observations_ms"])["median_ms"]
        )
        self.assertEqual(calls, 2 + 1 + timing["iterations"])

    def test_minimum_samples_survive_a_short_budget(self) -> None:
        def measured(value: int) -> int:
            time.sleep(0.001)
            return value

        timing, _ = time_calls_adaptive(
            measured,
            [1],
            budget_ms=0.1,
            warmup_calls=0,
            min_samples=3,
            max_calls=10,
            target_sample_ms=0.001,
        )
        self.assertEqual(timing["samples"], 3)
        self.assertEqual(timing["iterations"], 3)

    def test_fixed_block_size_is_validated(self) -> None:
        timing, _ = time_calls_adaptive(
            lambda value: value,
            [1],
            budget_ms=0.1,
            warmup_calls=0,
            min_samples=3,
            max_calls=12,
            block_size=4,
        )
        self.assertEqual(timing["block_size"], 4)
        with self.assertRaises(ValueError):
            time_calls_adaptive(
                lambda value: value,
                [1],
                budget_ms=1.0,
                warmup_calls=0,
                min_samples=3,
                max_calls=10,
                block_size=4,
            )


class RegistryTests(unittest.TestCase):
    def test_case_and_route_identifiers_are_unique(self) -> None:
        self.assertEqual(len(CASES), len({case.id for case in CASES}))
        for case in CASES:
            self.assertEqual(len(case.routes), len({route.id for route in case.routes}))
            self.assertTrue(case.id.startswith(f"{case.suite}."))

    def test_atomic_transform_selection_reduces_dimensions(self) -> None:
        selectors = Selectors(
            cases=("transforms.affine.bilinear-reflect101",),
            sizes=(512,),
            participants=("variopinta",),
            variants=("compiled",),
        )
        validate_selector_values(CASES, selectors)
        plan = select_cases(CASES, selectors)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].sizes, (512,))
        self.assertEqual([route.id for route in plan[0].routes], ["variopinta.compiled"])

    def test_complete_selection_expands_the_case_matrix(self) -> None:
        selectors = Selectors(cases=("transforms.invert.default",), participants=("variopinta",))
        plan = select_cases(CASES, selectors, complete=True)
        self.assertGreater(len(plan[0].routes), 1)
        self.assertEqual(plan[0].sizes, (224, 512, 1024))

    def test_unknown_values_fail_before_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown benchmark participants"):
            validate_selector_values(CASES, Selectors(participants=("missing",)))


class EvidenceTests(unittest.TestCase):
    def test_shard_path_follows_case_identifier(self) -> None:
        case = next(case for case in CASES if case.id == "transforms.invert.default")
        root = Path("/tmp/evidence")
        self.assertEqual(shard_path(case, root), root / "transforms" / "invert" / "default.json")

    def test_atomic_write_preserves_previous_file_on_serialization_failure(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json_atomic(path, {"value": 1})
            with self.assertRaises(TypeError):
                write_json_atomic(path, {"value": object()})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_completeness_requires_every_route_size_and_repetition(self) -> None:
        route = RouteSpec("variopinta.compiled", "variopinta", "compiled", "rust")
        case = CaseSpec(
            "transforms.test.default",
            "transforms",
            "Test",
            (),
            (route,),
            (224,),
            "layers",
            "transform:Invert",
            "policy",
            ("transforms",),
            TimingPolicy(1.0, 0, 1, 1),
        )
        planned = PlannedCase(case, case.routes, case.sizes)
        row = {
            "route_id": route.id,
            "size": 224,
            "repetition": 1,
            "valid": True,
            "samples": 1,
            "observations_ms": [1.0],
        }
        validate_complete(planned, [row], 1)
        with self.assertRaisesRegex(RuntimeError, "incomplete evidence"):
            validate_complete(planned, [row, row], 1)

    def test_status_rejects_an_incomplete_canonical_matrix(self) -> None:
        route = RouteSpec("variopinta.compiled", "variopinta", "compiled", "rust")
        case = CaseSpec(
            "transforms.test.default",
            "transforms",
            "Test",
            (),
            (route,),
            (224,),
            "layers",
            "transform:Invert",
            "policy",
            ("transforms",),
            TimingPolicy(1.0, 0, 1, 1),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(
                shard_path(case, root),
                {
                    "schema_version": 1,
                    "case_id": case.id,
                    "case": case.normalized(),
                    "fingerprint": {"digest": "current"},
                    "rows": [
                        {
                            "route_id": route.id,
                            "size": 224,
                            "repetition": 1,
                            "valid": True,
                            "samples": 1,
                            "observations_ms": [1.0],
                        }
                    ],
                },
            )
            with (
                patch("benchmarks.evidence.unclassified_measured_paths", return_value=()),
                patch(
                    "benchmarks.evidence.case_fingerprint",
                    return_value={"digest": "current"},
                ),
            ):
                status = status_for(case, root)
        self.assertEqual(status.state, "invalid")


class FingerprintTests(unittest.TestCase):
    def _root(self, directory: str) -> tuple[Path, CaseSpec]:
        root = Path(directory)
        for relative in (
            "benchmarks/common.py",
            "benchmarks/model.py",
            "benchmarks/selection.py",
            "benchmarks/controller.py",
            "benchmarks/worker.py",
            "benchmarks/adapters.py",
            "benchmarks/layer_worker.py",
            "benchmarks/environments.py",
            "scripts/setup_benchmark_envs.py",
            "requirements/benchmarks.txt",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        route = RouteSpec("torchvision.stock", "torchvision", "stock", "torchvision")
        case = CaseSpec(
            "transforms.test.default",
            "transforms",
            "Test",
            (),
            (route,),
            (224,),
            "layers",
            "transform:Invert",
            "policy",
            ("transforms",),
            TimingPolicy(1.0, 0, 1, 1),
        )
        return root, case

    def test_view_change_does_not_invalidate_measurements(self) -> None:
        with TemporaryDirectory() as directory:
            root, case = self._root(directory)
            view = root / "benchmarks" / "views.py"
            view.write_text("one")
            original = case_fingerprint(case, root)["digest"]
            view.write_text("two")
            self.assertEqual(case_fingerprint(case, root)["digest"], original)
            (root / "benchmarks" / "layer_worker.py").write_text("changed")
            self.assertNotEqual(case_fingerprint(case, root)["digest"], original)

    def test_unknown_benchmark_source_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root, _ = self._root(directory)
            unknown = root / "benchmarks" / "new_worker.py"
            unknown.write_text("value = 1")
            self.assertEqual(unclassified_measured_paths(root), (Path("benchmarks/new_worker.py"),))


class BenchmarkMetadataTests(unittest.TestCase):
    def test_optional_packages_do_not_need_to_be_importable(self) -> None:
        versions = {"numpy": "2.2.6", "torch": "2.13.0", "variopinta": "0.3.0"}

        def package_version(name: str) -> str:
            try:
                return versions[name]
            except KeyError as error:
                raise PackageNotFoundError(name) from error

        with patch("importlib.metadata.version", side_effect=package_version):
            result = metadata("rust", {"logical_cpu": 0})
        self.assertEqual(result["torch"], "2.13.0")
        self.assertIsNone(result["torchvision"])


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_missing_environment_reports_names(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.object(environments, "ENV_ROOT", Path(directory)):
                with self.assertRaises(SystemExit) as caught:
                    environments.require_environments(("rust", "io"))
        message = str(caught.exception)
        self.assertIn("io", message)
        self.assertIn("rust", message)

    def test_environment_python_uses_configured_root(self) -> None:
        root = Path("/tmp/variopinta-benchmark-test")
        with patch.object(environments, "ENV_ROOT", root):
            self.assertEqual(
                environments.python_for("torchvision"), root / "torchvision" / "bin" / "python"
            )


if __name__ == "__main__":
    unittest.main()
