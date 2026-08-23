from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ncls.core.material import MaterialProgram, validate_material_program


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_studio_preset_locks_available_shaderball_and_valid_material() -> None:
    preset = _read_json(PROJECT_ROOT / "configs" / "viewer-studio-v1.json")
    assert preset["format_name"] == "ncls.viewer-studio"
    assert preset["format_version"] == 1

    geometry_source = PROJECT_ROOT / str(preset["reference_geometry_source"])
    assert geometry_source.is_file()
    assert _sha256(geometry_source) == preset["reference_geometry_sha256"]

    material_path = PROJECT_ROOT / "configs" / str(preset["source_material"])
    program = MaterialProgram.from_json(material_path.read_text(encoding="utf-8"))
    validate_material_program(program)
    assert len(program.nodes) == 2


def test_benchmark_uses_the_same_studio_assets_and_transport_limits() -> None:
    studio = _read_json(PROJECT_ROOT / "configs" / "viewer-studio-v1.json")
    benchmark = _read_json(PROJECT_ROOT / "configs" / "viewer-benchmark-v1.json")
    assert benchmark["format_name"] == "ncls.viewer-benchmark"
    assert benchmark["format_version"] == 2
    assert benchmark["reference_geometry_sha256"] == studio["reference_geometry_sha256"]
    assert benchmark["environment_sha256"] == studio["environment_sha256"]
    assert benchmark["reference_samples_per_frame"] == studio["reference_samples_per_frame"] == 4
    assert benchmark["reference_scene_max_bounces"] == studio["reference_scene_max_bounces"]
    assert benchmark["reference_layer_walk_max_depth"] == studio["reference_layer_walk_max_depth"]
    assert benchmark["source_material"] == f"configs/{studio['source_material']}"
    assert benchmark["display"]["denoised_preview"] is True

    target = studio["camera"]["target"]
    assert all(camera["target"] == target for camera in benchmark["camera_path"])
