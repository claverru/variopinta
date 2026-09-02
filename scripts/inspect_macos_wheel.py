from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


def command_output(*command: str) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def deployment_targets(load_commands: str) -> set[tuple[int, ...]]:
    targets = set()
    patterns = (
        r"cmd LC_BUILD_VERSION\b.*?\n\s*minos\s+([0-9.]+)",
        r"cmd LC_VERSION_MIN_MACOSX\b.*?\n\s*version\s+([0-9.]+)",
    )
    for pattern in patterns:
        for value in re.findall(pattern, load_commands, flags=re.DOTALL):
            target = tuple(int(part) for part in value.split("."))
            while len(target) > 2 and target[-1] == 0:
                target = target[:-1]
            targets.add(target)
    return targets


def linked_libraries(output: str) -> list[str]:
    return [
        line.strip().split(" ", 1)[0]
        for line in output.splitlines()
        if line.strip() and not line.rstrip().endswith(":")
    ]


def dylib_identifiers(load_commands: str) -> set[str]:
    return set(
        re.findall(
            r"cmd LC_ID_DYLIB\b.*?\n\s*name\s+(\S+)",
            load_commands,
            flags=re.DOTALL,
        )
    )


def validate_native_metadata(
    file_output: str, lipo_output: str, load_commands: str, libraries_output: str
) -> None:
    if "Mach-O 64-bit" not in file_output or "arm64" not in file_output:
        raise ValueError(f"extension is not a 64-bit ARM Mach-O binary: {file_output}")
    architectures = lipo_output.split()
    if architectures != ["arm64"]:
        raise ValueError(f"extension is not thin arm64: {architectures}")
    targets = deployment_targets(load_commands)
    if targets != {(11, 0)}:
        raise ValueError(f"unexpected macOS deployment target: {sorted(targets)}")
    libraries = linked_libraries(libraries_output)
    identifiers = dylib_identifiers(load_commands)
    forbidden = [
        library
        for library in libraries
        if library not in identifiers and not library.startswith(("/usr/lib/", "/System/Library/"))
    ]
    if forbidden:
        raise ValueError(f"non-system dynamic libraries: {forbidden}")
    if any("turbojpeg" in library.lower() for library in libraries):
        raise ValueError("libturbojpeg must be linked statically")


def inspect_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        extensions = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).parent == PurePosixPath("variopinta")
            and PurePosixPath(name).name.startswith("_variopinta")
            and name.endswith(".so")
        ]
        if len(extensions) != 1:
            raise ValueError(f"expected one native extension, found {extensions}")
        with TemporaryDirectory() as directory:
            extension = Path(directory) / PurePosixPath(extensions[0]).name
            extension.write_bytes(archive.read(extensions[0]))
            validate_native_metadata(
                command_output("file", str(extension)),
                command_output("lipo", "-archs", str(extension)),
                command_output("otool", "-l", str(extension)),
                command_output("otool", "-L", str(extension)),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a macOS ARM64 release wheel")
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    inspect_wheel(args.wheel)
    print(f"validated thin arm64 Mach-O extension in {args.wheel.name}")


if __name__ == "__main__":
    main()
