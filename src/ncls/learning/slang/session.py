from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any


_UNIFIED_CORE_PATH = (
    Path(__file__).resolve().parents[4]
    / "shaders"
    / "ncls"
    / "backends"
    / "unified_neural"
    / "unified_neural_core.slang"
)
_NVIDIA_CORE_PATH = (
    Path(__file__).resolve().parents[4]
    / "shaders"
    / "ncls"
    / "backends"
    / "nvidia_neural_appearance"
    / "nvidia_neural_appearance_core.slang"
)
_NVIDIA_MATCHED_LTC_PATH = (
    Path(__file__).resolve().parents[4]
    / "shaders"
    / "ncls"
    / "backends"
    / "nvidia_neural_appearance"
    / "nvidia_matched_ltc_sampler.slang"
)
_INCLUDE_PATTERN = re.compile(rb'^\s*#include\s+"([^"]+)"', re.MULTILINE)


def _slang_implementation_files(core_path: Path) -> tuple[Path, ...]:
    discovered: set[Path] = set()
    pending = [core_path.resolve()]
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        if not path.is_file():
            raise ValueError(f"method Slang include is missing: {path}")
        discovered.add(path)
        for match in _INCLUDE_PATTERN.finditer(path.read_bytes()):
            included = (path.parent / match.group(1).decode("utf-8")).resolve()
            pending.append(included)
    return tuple(sorted(discovered, key=lambda path: path.as_posix()))


def _slang_implementation_sha256(core_path: Path) -> str:
    digest = hashlib.sha256()
    for path in _slang_implementation_files(core_path):
        relative = path.relative_to(core_path.parents[4])
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def unified_slang_implementation_files() -> tuple[Path, ...]:
    """递归展开 exact-core 候选实际编译到的项目 Slang。"""

    return _slang_implementation_files(_UNIFIED_CORE_PATH)


def unified_slang_implementation_sha256() -> str:
    return _slang_implementation_sha256(_UNIFIED_CORE_PATH)


def nvidia_neural_appearance_implementation_files() -> tuple[Path, ...]:
    """递归展开 NVIDIA baseline 实际编译到的项目 Slang。"""

    return _slang_implementation_files(_NVIDIA_CORE_PATH)


def nvidia_neural_appearance_implementation_sha256() -> str:
    return _slang_implementation_sha256(_NVIDIA_CORE_PATH)


def nvidia_matched_ltc_implementation_files() -> tuple[Path, ...]:
    """递归展开冻结 baseline 上的 LTC matched adaptation。"""

    return _slang_implementation_files(_NVIDIA_MATCHED_LTC_PATH)


def nvidia_matched_ltc_implementation_sha256() -> str:
    return _slang_implementation_sha256(_NVIDIA_MATCHED_LTC_PATH)


class UnifiedSlangSession:
    """加载唯一 Falcor-free method core；SlangPy 与 Falcor 测试包含同一文件。"""

    def __init__(self) -> None:
        import slangpy as spy

        self.spy = spy
        self.device = spy.create_device(type=spy.DeviceType.cuda)
        self.module = spy.Module.load_from_file(self.device, str(_UNIFIED_CORE_PATH))

    @property
    def implementation_sha256(self) -> str:
        return unified_slang_implementation_sha256()

    def call(self, name: str, *args: Any) -> Any:
        try:
            function = getattr(self.module, name)
        except AttributeError as error:
            raise ValueError(f"unknown unified Slang function {name!r}") from error
        return function(*args)


class NvidiaNeuralAppearanceSlangSession:
    """加载 NVIDIA 原规模 baseline 的唯一 Falcor-free method core。"""

    def __init__(self) -> None:
        import slangpy as spy

        self.spy = spy
        self.device = spy.create_device(type=spy.DeviceType.cuda)
        self.module = spy.Module.load_from_file(self.device, str(_NVIDIA_CORE_PATH))

    @property
    def implementation_sha256(self) -> str:
        return nvidia_neural_appearance_implementation_sha256()

    def call(self, name: str, *args: Any) -> Any:
        try:
            function = getattr(self.module, name)
        except AttributeError as error:
            raise ValueError(f"unknown NVIDIA baseline Slang function {name!r}") from error
        return function(*args)


class NvidiaMatchedLtcSlangSession:
    """加载独立 matched LTC head；不改变 NVIDIA reproduction core。"""

    def __init__(self) -> None:
        import slangpy as spy

        self.spy = spy
        self.device = spy.create_device(type=spy.DeviceType.cuda)
        self.module = spy.Module.load_from_file(
            self.device, str(_NVIDIA_MATCHED_LTC_PATH)
        )

    @property
    def implementation_sha256(self) -> str:
        return nvidia_matched_ltc_implementation_sha256()

    def call(self, name: str, *args: Any) -> Any:
        try:
            function = getattr(self.module, name)
        except AttributeError as error:
            raise ValueError(f"unknown NVIDIA matched LTC function {name!r}") from error
        return function(*args)
