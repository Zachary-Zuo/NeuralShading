"""锁定 Falcor reference 在 Windows/D3D12 与 Linux/Vulkan 间的显式平台选择。"""

from __future__ import annotations

import pytest

import ncls.references.falcor as falcor_support


class _FakeFalcor:
    class DeviceType:
        D3D12 = "d3d12"
        Vulkan = "vulkan"

    class Device:
        def __init__(self, *, type: str) -> None:
            self.type = type


@pytest.mark.parametrize(
    ("platform", "expected"),
    (("win32", "d3d12"), ("linux", "vulkan"), ("linux2", "vulkan")),
)
def test_create_falcor_device_selects_platform_api(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    expected: str,
) -> None:
    monkeypatch.setattr(falcor_support.sys, "platform", platform)
    assert falcor_support.create_falcor_device(_FakeFalcor).type == expected


def test_create_falcor_device_rejects_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(falcor_support.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="does not support platform"):
        falcor_support.create_falcor_device(_FakeFalcor)
