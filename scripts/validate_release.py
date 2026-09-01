from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

PACKAGE = "variopinta"
PYTHON_FILES = {
    "__init__.py",
    "_validation.py",
    "io.py",
    "pipeline.py",
    "py.typed",
    "transforms.py",
}
FORBIDDEN_PARTS = {
    ".devcontainer",
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "__pycache__",
    "docs",
    "release-dist",
    "target",
    "wheelhouse",
}
FORBIDDEN_NAMES = {
    ".env",
    ".pypirc",
    "AGENTS.md",
    "credentials",
    "credentials.json",
    "ongoing.md",
}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".pyc", ".pyo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate release versions and artifacts")
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--tag")
    return parser.parse_args()


def toml_value(path: Path, section: str, key: str) -> str:
    current_section = ""
    assignment = re.compile(rf'{re.escape(key)}\s*=\s*"([^"]+)"\s*')
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue
        if current_section == section and (match := assignment.fullmatch(line)):
            return match.group(1)
    raise ValueError(f"missing [{section}] {key} in {path}")


def source_version(tag: str | None) -> str:
    python_version = toml_value(Path("pyproject.toml"), "project", "version")
    rust_version = toml_value(Path("rust/Cargo.toml"), "workspace.package", "version")
    if python_version != rust_version:
        raise ValueError(f"Python version {python_version} != Rust version {rust_version}")
    if tag is not None and tag != f"v{python_version}":
        raise ValueError(f"tag {tag!r} != v{python_version}")
    return python_version


def validated_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    if FORBIDDEN_PARTS.intersection(path.parts):
        raise ValueError(f"forbidden archive path: {name}")
    if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden archive file: {name}")
    return path


def parse_metadata(data: bytes) -> Message:
    return BytesParser(policy=policy.default).parsebytes(data)


def validate_metadata(metadata: Message, version: str) -> None:
    expected_classifiers = {
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    }
    classifiers = set(metadata.get_all("Classifier", []))
    requirements = {value.replace(" ", "") for value in metadata.get_all("Requires-Dist", [])}
    if metadata["Name"] != PACKAGE:
        raise ValueError(f"unexpected package name: {metadata['Name']}")
    if metadata["Version"] != version:
        raise ValueError(f"unexpected package version: {metadata['Version']}")
    if metadata["Requires-Python"].replace(" ", "") != ">=3.10,<3.14":
        raise ValueError(f"unexpected Python requirement: {metadata['Requires-Python']}")
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError(f"unexpected license: {metadata['License-Expression']}")
    if "numpy>=2.2.6" not in requirements:
        raise ValueError(f"unexpected runtime requirements: {sorted(requirements)}")
    if not expected_classifiers.issubset(classifiers):
        raise ValueError("supported Python classifiers are incomplete")


def validate_wheel(path: Path, version: str) -> dict[str, object]:
    expected_name = f"{PACKAGE}-{version}-cp310-abi3-manylinux_2_34_x86_64.whl"
    if path.name != expected_name:
        raise ValueError(f"unexpected wheel filename: {path.name}")
    dist_info = f"{PACKAGE}-{version}.dist-info"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        parsed = [validated_path(name) for name in names]
        top_levels = {item.parts[0] for item in parsed if item.parts}
        if top_levels != {PACKAGE, dist_info}:
            raise ValueError(f"unexpected wheel roots: {sorted(top_levels)}")
        package_files = {
            item.name for item in parsed if len(item.parts) == 2 and item.parts[0] == PACKAGE
        }
        extension_files = {
            name
            for name in package_files
            if name.startswith("_variopinta") and name.endswith(".so")
        }
        if package_files - extension_files != PYTHON_FILES or len(extension_files) != 1:
            raise ValueError(f"unexpected package files: {sorted(package_files)}")
        required = {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/THIRD_PARTY_NOTICES",
        }
        if not required.issubset(names):
            raise ValueError(f"wheel is missing: {sorted(required - set(names))}")
        validate_metadata(parse_metadata(archive.read(f"{dist_info}/METADATA")), version)
        wheel_metadata = parse_metadata(archive.read(f"{dist_info}/WHEEL"))
        if "cp310-abi3-manylinux_2_34_x86_64" not in wheel_metadata.get_all("Tag", []):
            raise ValueError("wheel metadata does not contain the supported ABI tag")
    return {"filename": path.name, "files": len(names), "kind": "wheel"}


def validate_sdist(path: Path, version: str) -> dict[str, object]:
    expected_name = f"{PACKAGE}-{version}.tar.gz"
    root = f"{PACKAGE}-{version}"
    if path.name != expected_name:
        raise ValueError(f"unexpected sdist filename: {path.name}")
    allowed_roots = {
        "CHANGELOG.md",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "THIRD_PARTY_NOTICES",
        "pyproject.toml",
        "python",
        "rust",
    }
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        parsed = [validated_path(member.name) for member in members]
        for member, member_path in zip(members, parsed, strict=True):
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"unsupported sdist member type: {member.name}")
            if not member_path.parts or member_path.parts[0] != root:
                raise ValueError(f"unexpected sdist root: {member.name}")
            if len(member_path.parts) > 1 and member_path.parts[1] not in allowed_roots:
                raise ValueError(f"unexpected sdist content: {member.name}")
        names = {member.name for member in members}
        required = {
            f"{root}/LICENSE",
            f"{root}/PKG-INFO",
            f"{root}/README.md",
            f"{root}/THIRD_PARTY_NOTICES",
            f"{root}/pyproject.toml",
            f"{root}/rust/Cargo.lock",
            f"{root}/python/{PACKAGE}/py.typed",
        }
        if not required.issubset(names):
            raise ValueError(f"sdist is missing: {sorted(required - names)}")
        package_files = {
            item.name
            for item in parsed
            if len(item.parts) == 4 and item.parts[:3] == (root, "python", PACKAGE)
        }
        if package_files != PYTHON_FILES:
            raise ValueError(f"unexpected sdist package files: {sorted(package_files)}")
        metadata_file = archive.extractfile(f"{root}/PKG-INFO")
        if metadata_file is None:
            raise ValueError("sdist PKG-INFO is not a regular file")
        validate_metadata(parse_metadata(metadata_file.read()), version)
    return {"filename": path.name, "files": len(members), "kind": "sdist"}


def validate_artifacts(dist_dir: Path, version: str) -> list[dict[str, object]]:
    artifacts = sorted(path for path in dist_dir.iterdir() if path.is_file())
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    if len(artifacts) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            f"expected one wheel and one sdist, found {[path.name for path in artifacts]}"
        )
    return [validate_wheel(wheels[0], version), validate_sdist(sdists[0], version)]


def main() -> None:
    args = parse_args()
    version = source_version(args.tag)
    artifacts = validate_artifacts(args.dist_dir, version) if args.dist_dir else []
    print(json.dumps({"artifacts": artifacts, "package": PACKAGE, "version": version}))


if __name__ == "__main__":
    main()
