from __future__ import annotations

import time
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmarks import environments
from benchmarks.common import (
    aggregate_runs,
    benchmark_fingerprint,
    metadata,
    summarize_observations,
    time_calls_adaptive,
)


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

    def test_invalid_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            time_calls_adaptive(
                lambda value: value,
                [1],
                budget_ms=0.0,
                warmup_calls=0,
                min_samples=1,
                max_calls=1,
            )

    def test_aggregation_requires_recalculable_observations(self) -> None:
        row = {
            "kind": "micro",
            "backend": "rust",
            "transform": "Invert",
            "pipeline": None,
            "policy": None,
            "size": 224,
            "median_ms": 2.0,
            "p95_ms": 3.0,
            "images_per_sec": 500.0,
            "observations_ms": [1.0, 2.0, 3.0],
            "repetition": 1,
            "worker": "rust-1",
            "backend_position": 1,
        }
        aggregated = aggregate_runs([row])
        self.assertEqual(aggregated[0]["median_ms"], 2.0)
        self.assertEqual(aggregated[0]["p95_ms"], 3.0)
        self.assertEqual(len(aggregated[0]["worker_observations"]), 1)

        missing = dict(row)
        missing.pop("observations_ms")
        with self.assertRaisesRegex(ValueError, "missing observations"):
            aggregate_runs([missing])

    def test_aggregation_rejects_inconsistent_sample_counts(self) -> None:
        row = {
            "kind": "micro",
            "backend": "rust",
            "transform": "Invert",
            "pipeline": None,
            "policy": None,
            "size": 224,
            "median_ms": 1.0,
            "p95_ms": 1.0,
            "observations_ms": [1.0],
            "samples": 2,
            "repetition": 1,
            "worker": "rust-1",
            "backend_position": 1,
        }
        with self.assertRaisesRegex(ValueError, "sample count"):
            aggregate_runs([row])


class BenchmarkMetadataTests(unittest.TestCase):
    def test_optional_packages_do_not_need_to_be_importable(self) -> None:
        versions = {"numpy": "2.2.6", "torch": "2.7.0", "variopinta": "0.1.0"}

        def package_version(name: str) -> str:
            try:
                return versions[name]
            except KeyError as error:
                raise PackageNotFoundError(name) from error

        with patch("importlib.metadata.version", side_effect=package_version):
            result = metadata("rust-catalog-audit", {"logical_cpu": 0})

        self.assertEqual(result["torch"], "2.7.0")
        self.assertIsNone(result["torchvision"])
        self.assertNotIn("torchvision", result["packages"])

    def test_fingerprint_ignores_only_local_package_version_fields(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmarks").mkdir()
            (root / "python" / "variopinta").mkdir(parents=True)
            (root / "requirements").mkdir()
            (root / "rust").mkdir()
            (root / "scripts").mkdir()
            (root / "benchmarks" / "worker.py").write_text("POLICY = 3\n")
            implementation = root / "python" / "variopinta" / "api.py"
            implementation.write_text("def measured(): return 1\n")
            (root / "requirements" / "benchmark.txt").write_text("numpy==2.2.6\n")
            pyproject = root / "pyproject.toml"
            pyproject.write_text('[project]\nname = "variopinta"\nversion = "0.2.0"\n')
            cargo = root / "rust" / "Cargo.toml"
            cargo.write_text('[workspace.package]\nversion = "0.2.0"\nedition = "2024"\n')
            lock = root / "rust" / "Cargo.lock"
            lock.write_text('[[package]]\nname = "augment-core"\nversion = "0.2.0"\n')
            (root / "scripts" / "setup_benchmark_envs.py").write_text("REPETITIONS = 3\n")

            original = benchmark_fingerprint(root)
            pyproject.write_text('[project]\nname = "variopinta"\nversion = "0.3.0"\n')
            cargo.write_text('[workspace.package]\nversion = "0.3.0"\nedition = "2024"\n')
            lock.write_text('[[package]]\nname = "augment-core"\nversion = "0.3.0"\n')
            self.assertEqual(benchmark_fingerprint(root), original)

            implementation.write_text("def measured(): return 2\n")
            self.assertNotEqual(benchmark_fingerprint(root), original)


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_missing_environment_reports_missing_backends(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.object(environments, "ENV_ROOT", Path(directory)):
                with self.assertRaises(SystemExit) as caught:
                    environments.require_environments(("rust", "albumentations"))

        message = str(caught.exception)
        self.assertIn("Missing benchmark environments", message)
        self.assertIn("albumentations", message)
        self.assertIn("rust", message)

    def test_backend_python_uses_configured_root(self) -> None:
        root = Path("/tmp/variopinta-benchmark-test")
        with patch.object(environments, "ENV_ROOT", root):
            self.assertEqual(
                environments.python_for("torchvision"), root / "torchvision" / "bin" / "python"
            )


if __name__ == "__main__":
    unittest.main()
