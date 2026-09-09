from __future__ import annotations

import copy
import json
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmarks.evidence import shard_path, write_json_atomic
from benchmarks.fingerprints import compatibility_signature
from benchmarks.registry import CASE_BY_ID
from scripts.migrate_benchmark_evidence import migrate_shard, prepare_migration


class BenchmarkMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = replace(
            CASE_BY_ID["catalog.invert.default"], sizes=(224,), tags=("catalog", "transform:Invert")
        )
        self.previous = {
            "case": {**self.case.normalized(), "tags": ["catalog"]},
            "fingerprint": {"digest": "previous"},
        }
        self.current = {"digest": "current"}
        self.payload = {
            "schema_version": 1,
            "case_id": self.case.id,
            **copy.deepcopy(self.previous),
            "provenance": {"source_revision": "old", "source_dirty": True},
            "metadata": {"rust": {"python": "3.12"}},
            "compatibility": compatibility_signature({"rust": {"python": "3.12"}}),
            "execution_order": [{"worker": "original-worker"}],
            "rows": [
                {
                    "route_id": route.id,
                    "size": 224,
                    "repetition": repetition,
                    "valid": True,
                    "samples": 2,
                    "observations_ms": [1.25, 1.5],
                }
                for route in self.case.routes
                for repetition in (1, 2, 3)
            ],
        }

    def test_migration_preserves_all_measurement_fields_and_the_original_payload(self) -> None:
        original = copy.deepcopy(self.payload)
        result = migrate_shard(self.case, self.payload, self.previous, self.current, self.current)
        self.assertEqual(self.payload, original)
        self.assertNotIn("provenance", result)
        self.assertEqual(result["case"], self.case.normalized())
        self.assertEqual(result["fingerprint"], self.current)
        for key in original.keys() - {"case", "fingerprint", "provenance"}:
            self.assertEqual(result[key], original[key], key)

    def test_migration_rejects_unverified_evidence(self) -> None:
        for field, value in (
            ("fingerprint", {"digest": "different"}),
            ("case", {}),
            ("rows", []),
            ("compatibility", {}),
        ):
            with self.subTest(field=field):
                with self.assertRaises((ValueError, RuntimeError)):
                    migrate_shard(
                        self.case,
                        {**self.payload, field: value},
                        self.previous,
                        self.current,
                        self.current,
                    )

    def test_migration_rejects_changes_to_the_measured_case_definition(self) -> None:
        with self.assertRaisesRegex(ValueError, "measured case definition changed"):
            migrate_shard(
                replace(self.case, sizes=(512,)),
                self.payload,
                self.previous,
                self.current,
                self.current,
            )

    def test_preparation_checks_sources_and_environment_before_writing(self) -> None:
        old_controller = (
            "from benchmarks.fingerprints import case_fingerprint, source_provenance\n"
            "def write_evidence():\n"
            "    provenance = source_provenance()\n"
            "    return {\n"
            '            "provenance": provenance,\n'
            "    }\n"
        )
        new_controller = (
            "from benchmarks.fingerprints import case_fingerprint\n"
            "def write_evidence():\n"
            "    return {\n"
            "    }\n"
        )
        for changed_path in (
            None,
            "rust/core/src/kernels/point.rs",
            "requirements/benchmarks.txt",
            "benchmarks/controller.py",
        ):
            with self.subTest(changed_path=changed_path), TemporaryDirectory() as directory:
                root, snapshot = Path(directory) / "current", Path(directory) / "previous"
                for tree, controller in ((root, new_controller), (snapshot, old_controller)):
                    for relative, source in (
                        ("benchmarks/controller.py", controller),
                        ("rust/core/src/kernels/point.rs", "original kernel"),
                        ("requirements/benchmarks.txt", "original environment"),
                    ):
                        path = tree / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(source)
                path = shard_path(self.case, root / "benchmarks/evidence")
                write_json_atomic(path, self.payload)
                original_bytes = path.read_bytes()
                if changed_path is not None:
                    (root / changed_path).write_text("changed")
                export = subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps({self.case.id: self.previous})
                )
                with (
                    patch("scripts.migrate_benchmark_evidence.CASES", (self.case,)),
                    patch("scripts.migrate_benchmark_evidence.subprocess.run", return_value=export),
                ):
                    if changed_path is None:
                        updates = prepare_migration(snapshot, root)
                        self.assertEqual(len(updates), 1)
                        self.assertEqual(updates[0][1]["rows"], self.payload["rows"])
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "measured sources or environment changed"
                        ):
                            prepare_migration(snapshot, root)
                self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
