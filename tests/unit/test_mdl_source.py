from __future__ import annotations

import json
from pathlib import Path

import pytest

from ncls.core.identity import sha256_file
from ncls.core.source import SourceEditOperation, SourceEditPatch, SourceSnapshot
from ncls.references.mdl import (
    ARTIFACT_SCHEMA,
    CODEGEN_OPTIONS,
    MDL_SDK_BUILD,
    STB_COMMIT,
    STB_IMAGE_SHA256,
    MdlCompiledArtifact,
)
from ncls.source_materials.families.mdl import MdlFamilyDefinition
from ncls.source_materials.mdl import MDL_FAMILY_ID, MDL_NATIVE_SCHEMA, MdlMaterialSource


def _snapshot() -> SourceSnapshot:
    module_root = Path(__file__).parents[1] / "fixtures/mdl"
    module = module_root / "constant_diffuse.mdl"
    source = MdlMaterialSource(
        module_root,
        "project.fixtures",
        "1",
        "::constant_diffuse",
        "::constant_diffuse::constant_diffuse(color)",
        {"tint": {"mdl_type": "color", "value": [0.8, 0.2, 0.1], "editable": True}},
        "1.7",
    )
    return SourceSnapshot(
        MDL_FAMILY_ID,
        1,
        MDL_NATIVE_SCHEMA,
        sha256_file(module),
        source.to_payload(),
        {"constant_diffuse.mdl": sha256_file(module)},
        {"module_root": str(module_root.resolve())},
        source,
    )


def _rich_snapshot() -> SourceSnapshot:
    module_root = Path(__file__).parents[1] / "fixtures/mdl"
    module = module_root / "constant_diffuse.mdl"
    checker = module_root / "checker.ppm"
    source = MdlMaterialSource(
        module_root,
        "project.fixtures",
        "1",
        "::constant_diffuse",
        "::constant_diffuse::constant_diffuse(color)",
        {
            "mode": {
                "mdl_type": "enum",
                "value": {"name": "repeat", "value": 0},
                "editable": True,
                "choices": [
                    {"name": "repeat", "value": 0},
                    {"name": "clamp", "value": 1},
                ],
            },
            "roughness": {
                "mdl_type": "float",
                "value": 0.4,
                "editable": True,
                "minimum": 0.0,
                "maximum": 1.0,
                "soft_minimum": 0.1,
                "soft_maximum": 0.8,
            },
            "image": {
                "mdl_type": "texture_2d",
                "value": {"path": "checker.ppm", "effective_gamma": 1.0},
                "editable": True,
            },
        },
        "1.7",
    )
    return SourceSnapshot(
        MDL_FAMILY_ID,
        1,
        MDL_NATIVE_SCHEMA,
        sha256_file(module),
        source.to_payload(),
        {
            "constant_diffuse.mdl": sha256_file(module),
            "checker.ppm": sha256_file(checker),
        },
        {"module_root": str(module_root.resolve())},
        source,
    )


def test_mdl_typed_edit_changes_snapshot_identity_and_rejects_stale_patch() -> None:
    snapshot = _snapshot()
    family = MdlFamilyDefinition()
    view = family.describe_parameters(snapshot)
    tint = view.root.children[0].children[0]
    assert tint.path == "/arguments/tint"
    assert tint.value_type == "color3"
    assert tint.editable

    patch = SourceEditPatch(
        snapshot.snapshot_id,
        (SourceEditOperation("set", "/arguments/tint", [0.1, 0.4, 0.7]),),
    )
    result = family.apply_edit(snapshot, patch)
    assert result.snapshot.snapshot_id != snapshot.snapshot_id
    payload = json.loads(result.snapshot.native_payload.decode("utf-8"))
    assert payload["arguments"]["tint"]["value"] == [0.1, 0.4, 0.7]
    assert result.changed_paths == ("/arguments/tint",)
    with pytest.raises(ValueError, match="stale"):
        family.apply_edit(result.snapshot, patch)


def test_mdl_typed_edit_rejects_wrong_shape_and_nonfinite_values() -> None:
    snapshot = _snapshot()
    family = MdlFamilyDefinition()
    for value in ([0.1, 0.2], [0.1, float("nan"), 0.3]):
        patch = SourceEditPatch(
            snapshot.snapshot_id,
            (SourceEditOperation("set", "/arguments/tint", value),),
        )
        with pytest.raises(ValueError, match="component|finite"):
            family.apply_edit(snapshot, patch)


def test_mdl_enum_range_and_texture_edits_preserve_native_contract() -> None:
    snapshot = _rich_snapshot()
    family = MdlFamilyDefinition()
    nodes = {node.path: node for node in family.describe_parameters(snapshot).root.children[0].children}
    assert nodes["/arguments/mode"].choices == ("repeat", "clamp")
    assert nodes["/arguments/mode"].value == "repeat"
    assert nodes["/arguments/roughness"].minimum == 0.0
    assert nodes["/arguments/roughness"].maximum == 1.0
    assert nodes["/arguments/roughness"].metadata["soft_minimum"] == 0.1
    assert nodes["/arguments/image"].kind == "resource"
    assert nodes["/arguments/image"].binding == "texture"

    result = family.apply_edit(
        snapshot,
        SourceEditPatch(
            snapshot.snapshot_id,
            (
                SourceEditOperation("set", "/arguments/mode", "clamp"),
                SourceEditOperation("set", "/arguments/roughness", 0.75),
                SourceEditOperation(
                    "set",
                    "/arguments/image",
                    {"path": "checker_alt.ppm", "effective_gamma": 1.0},
                ),
            ),
        ),
    )
    payload = json.loads(result.snapshot.native_payload.decode("utf-8"))
    assert payload["arguments"]["mode"]["value"] == {"name": "clamp", "value": 1}
    assert payload["arguments"]["roughness"]["value"] == 0.75
    assert payload["arguments"]["image"]["value"]["path"] == "checker_alt.ppm"
    assert "checker.ppm" not in result.snapshot.resource_hashes
    assert result.snapshot.resource_hashes["checker_alt.ppm"] == sha256_file(
        Path(__file__).parents[1] / "fixtures/mdl/checker_alt.ppm"
    )


@pytest.mark.parametrize(
    ("target", "value", "message"),
    (
        ("/arguments/mode", "mirror", "declared choices"),
        ("/arguments/roughness", 1.1, "hard maximum"),
        ("/arguments/image", "../checker.ppm", "module root"),
        ("/arguments/image", "missing.ppm", "missing"),
    ),
)
def test_mdl_native_editor_rejects_invalid_enum_range_and_resource(
    target: str, value: object, message: str
) -> None:
    snapshot = _rich_snapshot()
    patch = SourceEditPatch(snapshot.snapshot_id, (SourceEditOperation("set", target, value),))
    with pytest.raises(ValueError, match=message):
        MdlFamilyDefinition().apply_edit(snapshot, patch)


def test_mdl_compiled_artifact_validates_schema_and_argument_block(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "generated.hlsl").write_text("// generated", encoding="utf-8")
    (root / "argument-block.bin").write_bytes(bytes(16))
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "mdl_sdk": MDL_SDK_BUILD,
        "module": "::constant_diffuse",
        "material": "::constant_diffuse::constant_diffuse(color)",
        "code": "generated.hlsl",
        "capability_audit": {
            "surface_bsdf_evaluate": True,
            "emission": False,
            "volume": False,
            "displacement": False,
        },
        "compiler_identity": {
            "mdl_sdk": MDL_SDK_BUILD,
            "bridge_executable_sha256": "a" * 64,
            "stb_commit": STB_COMMIT,
            "stb_image_sha256": STB_IMAGE_SHA256,
            "codegen_options": CODEGEN_OPTIONS,
        },
        "argument_block": {"path": "argument-block.bin", "size": 16},
        "parameters": [],
        "ro_data": [],
        "textures": [],
        "diagnostics": "",
        "files_sha256": {
            "argument-block.bin": sha256_file(root / "argument-block.bin"),
            "generated.hlsl": sha256_file(root / "generated.hlsl"),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifact = MdlCompiledArtifact.load(root)
    assert len(artifact.argument_block) == 16
    assert len(artifact.artifact_sha256) == 64

    (root / "generated.hlsl").write_text("// tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        MdlCompiledArtifact.load(root)
    (root / "generated.hlsl").write_text("// generated", encoding="utf-8")

    manifest["argument_block"]["size"] = 12
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong size"):
        MdlCompiledArtifact.load(root)

    manifest["argument_block"]["size"] = 16
    del manifest["capability_audit"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="capability audit"):
        MdlCompiledArtifact.load(root)


def test_mdl_reference_protocol_schemas_are_versioned_json_schema() -> None:
    schema_root = Path(__file__).parents[2] / "references/mdl-vmaterials2-v1/schemas"
    expected = {
        "mdl-source.schema.json": "ncls.mdl-source@1",
        "mdl-compiled-artifact.schema.json": "ncls.mdl-compiled-artifact@1",
        "mdl-native-protocol.schema.json": "ncls.mdl-native-protocol@1",
        "mdl-oracle-request.schema.json": "ncls.mdl-oracle-request@1",
        "mdl-oracle-result.schema.json": "ncls.mdl-oracle-result@1",
    }
    assert {path.name for path in schema_root.glob("*.json")} == set(expected)
    for name, schema_id in expected.items():
        value = json.loads((schema_root / name).read_text(encoding="utf-8"))
        assert value["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert value["$id"] == schema_id
