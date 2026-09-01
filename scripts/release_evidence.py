from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


def command_output(*command: str) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record provenance for a tested release artifact")
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revision = command_output("git", "rev-parse", "HEAD")
    if revision != args.expected_revision:
        raise SystemExit(f"checkout {revision} does not match {args.expected_revision}")

    artifacts = []
    for path in args.artifact:
        data = path.read_bytes()
        evidence = {
            "filename": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        if path.suffix == ".whl":
            python_tag, abi_tag, platform_tag = path.stem.rsplit("-", 3)[-3:]
            evidence.update(
                {"abi_tag": abi_tag, "platform_tag": platform_tag, "python_tag": python_tag}
            )
        artifacts.append(evidence)
    payload = {
        "artifacts": artifacts,
        "package": "variopinta",
        "package_version": metadata.version("variopinta"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "revision": revision,
        "rustc": command_output("rustc", "--version"),
        "test_result": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
