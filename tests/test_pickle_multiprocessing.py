from __future__ import annotations

import importlib.util
import multiprocessing as mp
import subprocess
import sys
import unittest


class MultiprocessingPickleTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch is not installed")
    def test_dataloaders_two_epochs(self):
        for context in ("spawn", "forkserver"):
            if context not in mp.get_all_start_methods():
                continue
            for persistence in ("persistent", "nonpersistent"):
                with self.subTest(context=context, persistence=persistence):
                    subprocess.run(
                        [sys.executable, "-m", "tests._pickle_workers", context, persistence],
                        check=True,
                        timeout=120,
                    )

    @unittest.skipUnless(sys.platform == "linux", "controlled Linux fork regression")
    def test_idle_previously_used_pipeline_with_fork(self):
        subprocess.run(
            [sys.executable, "-m", "tests._pickle_workers", "fork"], check=True, timeout=30
        )
