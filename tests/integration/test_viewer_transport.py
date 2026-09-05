"""用同一 diffuse BSDF 验证共享 renderer、未选材质和四种 slot 组合。"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyexr
import pytest

from tests.fixtures.viewer_control import make_geometry, make_package

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "external/Falcor/build/windows-vs2022/bin/Release"
VIEWER = BIN / "NclsViewer.exe"
pytestmark = pytest.mark.skipif(sys.platform != "win32" or not VIEWER.is_file(), reason="Windows viewer required")


@pytest.fixture(scope="module")
def control():
    # 固定 fixture 路径使 Falcor 的 shader cache 可跨测试运行复用。
    root = ROOT / "artifacts/tests/viewer-transport"
    root.mkdir(parents=True, exist_ok=True)
    geometry = root / "scene.glb"
    make_geometry(BIN / "data/ncls-viewer/shaderball.glb", geometry)
    material, package_root, manifest = make_package(root / "control")
    return root, geometry, material, package_root, manifest


def run_viewer(root, geometry, material, package_root, manifest, mode, reverse=False):
    output = root / (mode + ("-swapped" if reverse else ""))
    slots = ["source-reference", manifest.package_id]
    if reverse:
        slots.reverse()
    args = [str(VIEWER), "--material", str(material), "--reference-geometry", str(geometry),
        "--bundle-root", str(package_root), "--slot0-package", slots[0], "--slot1-package", slots[1],
        "--slot0-mode", mode, "--slot1-mode", mode, "--width", "320", "--height", "240",
        "--headless", "--capture", str(output), "--reference-spp", "16"]
    with (root / (output.name + ".log")).open("w", encoding="utf-8") as log:
        result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    assert result.returncode == 0, root / (output.name + ".log")
    document = json.loads((output / "capture.json").read_text(encoding="utf-8"))
    return output, document


@pytest.mark.parametrize("mode", ["path-tracing", "deferred"])
def test_same_bsdf_transport_and_swapped_slots(control, mode):
    root, geometry, material, package_root, manifest = control
    original = None
    for reverse in (False, True):
        output, document = run_viewer(root, geometry, material, package_root, manifest, mode, reverse)
        assert len(document["scene_material_bindings"]) == 3
        assert sum(item["active"] for item in document["scene_material_bindings"]) == 1
        assert all(item["source_family_id"] == "ncls.layer-stack@1" for item in document["scene_material_bindings"])
        assert [slot["package_id"] for slot in document["slots"]] == (
            [manifest.package_id, "source-reference"] if reverse else ["source-reference", manifest.package_id])
        for slot in document["slots"]:
            assert slot["status"] == "ready" and slot["mode"] == mode
            assert slot["spp"] == (16 if mode == "path-tracing" else 0)
            assert slot["gpu_timing_samples"] == (16 if mode == "path-tracing" else 1)
            assert slot["gpu_ms"] > 0
        images = [pyexr.read(str(output / slot["linear_output"])) for slot in document["slots"]]
        assert images[0].shape[:2] == (240, 160)
        assert all(np.isfinite(image).all() for image in images)
        # 同 BSDF、相同随机流，包含底座/地面遮挡与多 bounce，不能换材质或放宽容差。
        np.testing.assert_allclose(images[0], images[1], rtol=2e-5, atol=2e-6)
        if original is not None:
            # 交换后的每个 renderer 与自身逐位一致；source/package 间另用上面的严格浮点容差。
            np.testing.assert_array_equal(original[0], images[1])
            np.testing.assert_array_equal(original[1], images[0])
        original = images


def test_missing_pt_capability_is_rejected(control):
    root, geometry, _, _, _ = control
    material, package_root, manifest = make_package(root / "evaluate-only", capabilities=3)
    output = root / "unsupported"
    args = [str(VIEWER), "--material", str(material), "--reference-geometry", str(geometry),
        "--bundle-root", str(package_root), "--slot1-package", manifest.package_id,
        "--slot1-mode", "path-tracing", "--headless", "--capture", str(output),
        "--reference-spp", "1", "--width", "320", "--height", "240"]
    with (root / "unsupported.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0
    document = json.loads((output / "capture.json").read_text(encoding="utf-8"))
    slot = document["slots"][1]
    assert slot["status"] == "unsupported" and slot["mode"] == "path-tracing"
    assert "capabilities" in slot["diagnostic"]
    assert slot["spp"] == 0 and not (output / "capture-slot-1.exr").exists()
