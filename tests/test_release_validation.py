from __future__ import annotations

import tarfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.inspect_macos_wheel import deployment_targets, validate_native_metadata
from scripts.validate_release import (
    PACKAGE,
    PYTHON_FILES,
    validate_artifacts,
    validate_sdist,
    validate_selected_artifacts,
    validate_wheel,
)

VERSION = "0.2.0"


def metadata() -> bytes:
    classifiers = "\n".join(
        f"Classifier: {classifier}"
        for classifier in (
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
        )
    )
    return (
        "Metadata-Version: 2.4\n"
        f"Name: {PACKAGE}\n"
        f"Version: {VERSION}\n"
        "Requires-Python: >=3.10,<3.14\n"
        "License-Expression: Apache-2.0\n"
        "Requires-Dist: numpy>=2.2.6\n"
        f"{classifiers}\n\n"
    ).encode()


def build_wheel(directory: Path, platform_tag: str, *, extra: str | None = None) -> Path:
    path = directory / f"{PACKAGE}-{VERSION}-cp310-abi3-{platform_tag}.whl"
    dist_info = f"{PACKAGE}-{VERSION}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        for name in PYTHON_FILES:
            archive.writestr(f"{PACKAGE}/{name}", b"")
        archive.writestr(f"{PACKAGE}/_variopinta.abi3.so", b"native")
        archive.writestr(f"{dist_info}/METADATA", metadata())
        archive.writestr(f"{dist_info}/RECORD", b"")
        archive.writestr(
            f"{dist_info}/WHEEL", f"Wheel-Version: 1.0\nTag: cp310-abi3-{platform_tag}\n"
        )
        archive.writestr(f"{dist_info}/licenses/LICENSE", b"license")
        archive.writestr(f"{dist_info}/licenses/THIRD_PARTY_NOTICES", b"notices")
        if extra is not None:
            archive.writestr(extra, b"unexpected")
    return path


def build_sdist(directory: Path, *, extra: str | None = None) -> Path:
    path = directory / f"{PACKAGE}-{VERSION}.tar.gz"
    root = f"{PACKAGE}-{VERSION}"
    files = {
        "LICENSE": b"license",
        "PKG-INFO": metadata(),
        "README.md": b"readme",
        "THIRD_PARTY_NOTICES": b"notices",
        "pyproject.toml": b"project",
        "rust/Cargo.lock": b"lock",
        **{f"python/{PACKAGE}/{name}": b"" for name in PYTHON_FILES},
    }
    if extra is not None:
        files[extra] = b"unexpected"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    return path


class ReleaseValidationTests(unittest.TestCase):
    def test_complete_two_platform_artifact_set_is_valid(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_wheel(root, "manylinux_2_34_x86_64")
            build_wheel(root, "macosx_11_0_arm64")
            build_sdist(root)
            validated = validate_artifacts(root, VERSION)
            self.assertEqual(len(validated), 3)
            self.assertEqual(
                {item.get("platform") for item in validated if item["kind"] == "wheel"},
                {"linux-x86-64", "macos-arm64"},
            )

    def test_missing_and_unexpected_platforms_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_wheel(root, "manylinux_2_34_x86_64")
            build_sdist(root)
            with self.assertRaisesRegex(ValueError, "missing=.*macosx_11_0_arm64"):
                validate_artifacts(root, VERSION)
            build_wheel(root, "win_amd64")
            with self.assertRaisesRegex(ValueError, "unexpected=.*win_amd64"):
                validate_artifacts(root, VERSION)

    def test_duplicate_artifact_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            wheel = build_wheel(Path(directory), "macosx_11_0_arm64")
            with self.assertRaisesRegex(ValueError, "duplicate artifacts"):
                validate_selected_artifacts([wheel, wheel], VERSION)

    def test_wrong_abi_and_archive_contents_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_abi = build_wheel(root, "macosx_11_0_arm64")
            renamed = root / wrong_abi.name.replace("cp310-abi3", "cp311-cp311")
            wrong_abi.rename(renamed)
            with self.assertRaisesRegex(ValueError, "unexpected wheel filename"):
                validate_wheel(renamed, VERSION)
            wheel = build_wheel(root, "macosx_11_0_arm64", extra=f"{PACKAGE}/debug.py")
            with self.assertRaisesRegex(ValueError, "unexpected package files"):
                validate_wheel(wheel, VERSION)
            sdist = build_sdist(root, extra="docs/internal.md")
            with self.assertRaisesRegex(ValueError, "forbidden archive path"):
                validate_sdist(sdist, VERSION)


class MacosInspectionTests(unittest.TestCase):
    LOAD_COMMANDS = """
Load command 8
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 11.0
      sdk 15.5
"""
    LIBRARIES = """
/tmp/_variopinta.abi3.so:
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1345.120.2)
\t/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 3500.0.0)
"""

    def test_thin_arm64_system_only_binary_is_valid(self) -> None:
        self.assertEqual(deployment_targets(self.LOAD_COMMANDS), {(11, 0)})
        validate_native_metadata(
            "Mach-O 64-bit dynamically linked shared library arm64",
            "arm64",
            self.LOAD_COMMANDS,
            self.LIBRARIES,
        )

    def test_universal_wrong_target_and_non_system_dependency_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not thin arm64"):
            validate_native_metadata(
                "Mach-O 64-bit dynamically linked shared library arm64",
                "x86_64 arm64",
                self.LOAD_COMMANDS,
                self.LIBRARIES,
            )
        with self.assertRaisesRegex(ValueError, "deployment target"):
            validate_native_metadata(
                "Mach-O 64-bit dynamically linked shared library arm64",
                "arm64",
                self.LOAD_COMMANDS.replace("11.0", "12.0"),
                self.LIBRARIES,
            )
        with self.assertRaisesRegex(ValueError, "non-system dynamic"):
            validate_native_metadata(
                "Mach-O 64-bit dynamically linked shared library arm64",
                "arm64",
                self.LOAD_COMMANDS,
                self.LIBRARIES
                + "\t/opt/homebrew/lib/libturbojpeg.dylib (compatibility version 0.0.0)\n",
            )


if __name__ == "__main__":
    unittest.main()
