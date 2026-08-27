from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the pinned falcor2 MDL oracle")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mdl-archive", type=Path, required=True)
    parser.add_argument("--configuration", choices=("Debug", "Release"), default="Release")
    return parser


def _msvc_environment(source: Path) -> dict[str, str]:
    sys.path.insert(0, str(source / "external/slangpy/tools"))
    import msvc  # type: ignore[import-not-found]

    # MSBuild treats environment keys case-insensitively. The setuptools helper can
    # return lower-case keys while conda exposes their upper-case equivalents; keeping
    # both makes old TBB projects fail before CL starts.
    environment = {
        key.upper(): value for key, value in dict(msvc.msvc14_get_vc_env("x64")).items()
    }
    for key, value in os.environ.items():
        normalized = key.upper()
        if normalized in {"SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE"}:
            environment[normalized] = value
    return environment


def main() -> int:
    args = _parser().parse_args()
    source = args.source.resolve()
    archive = args.mdl_archive.resolve()
    mdl_source = archive.with_suffix("")
    if (
        not (source / "CMakePresets.json").is_file()
        or not archive.is_file()
        or not (mdl_source / "bin/libmdl_sdk.dll").is_file()
    ):
        raise FileNotFoundError("falcor2 source or MDL SDK archive is missing")

    build_dir = source / "build/windows-vs2022-oracle"
    environment = _msvc_environment(source)
    subprocess.run(
        [
            "cmake",
            "--preset",
            "windows-vs2022",
            "-B",
            str(build_dir),
            f"-DPython_ROOT_DIR:PATH={sys.prefix}",
            "-DPython_FIND_REGISTRY:STRING=NEVER",
            "-DFALCOR_MDL_URL:STRING=https://github.com/NVIDIA/MDL-SDK/releases/download/2025/MDL-SDK-2025.0.0-387700.1252-nt-x86-64.zip",
            f"-DFETCHCONTENT_SOURCE_DIR_MDL-SDK:PATH={mdl_source}",
            "-DFALCOR_BUILD_TESTS:BOOL=OFF",
            "-DFALCOR_ENABLE_NGX:BOOL=OFF",
            "-DFALCOR_ENABLE_CRASHPAD:BOOL=OFF",
        ],
        cwd=source,
        env=environment,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", args.configuration, "--parallel", "12"],
        cwd=source,
        env=environment,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
