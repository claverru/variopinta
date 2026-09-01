from __future__ import annotations

import os
import subprocess

from environments import ROOT, python_for, rebuild_variopinta, require_environments

require_environments(("albumentations", "rust"))
rebuild_variopinta()
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join(
    (str(ROOT / "python"), str(ROOT / "benchmarks"), environment.get("PYTHONPATH", ""))
)
subprocess.run(
    [str(python_for("albumentations")), str(ROOT / "benchmarks" / "io_performance_worker.py")],
    cwd=ROOT,
    env=environment,
    check=True,
)
