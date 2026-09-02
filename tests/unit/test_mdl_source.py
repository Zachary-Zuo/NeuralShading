from __future__ import annotations

import json
import errno
from pathlib import Path

import numpy as np
import pytest

import ncls.references.mdl as mdl_module
import ncls.source_materials.mdl as source_mdl_module
from ncls.core.identity import sha256_file
from ncls.core.source import SourceEditOperation, SourceEditPatch, SourceSnapshot
from ncls.references.mdl import (
    ARTIFACT_SCHEMA,
    CODEGEN_OPTIONS,
    MDL_SDK_BUILD,
    MdlSdkProgramProvider,
    STB_COMMIT,
    STB_IMAGE_SHA256,
    MdlCompiledArtifact,
    MdlModuleDiscovery,
)
from ncls.source_materials.families.mdl import MdlFamilyDefinition
from ncls.source_materials.mdl import MDL_FAMILY_ID, MDL_NATIVE_SCHEMA, MdlMaterialSource
from ncls.references.programs.mdl import _decoded_texture_binding


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
        "texture_payloads": "decoded",
        "capability_audit": {
            "surface_bsdf_evaluate": True,
            "emission": False,
            "volume": False,
            "displacement": False,
            "cutout_opacity": False,
        },
        "compiled_material_hash": "1" * 32,
        "sub_expression_hashes": {
            "surface.scattering": "2" * 32,
            "geometry.normal": "3" * 32,
            "geometry.cutout_opacity": "4" * 32,
        },
        "compiler_identity": {
            "mdl_sdk": MDL_SDK_BUILD,
            "platform_id": "windows-x86_64@1",
            "semantic_identity": "b" * 64,
            "build_identity": "c" * 64,
            "backend_manifest_sha256": "d" * 64,
            "sdk_archive_sha256": "e" * 64,
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
    artifact.require_runtime_supported()

    manifest["capability_audit"]["cutout_opacity"] = True
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    unsupported = MdlCompiledArtifact.load(root)
    assert not unsupported.runtime_supported
    with pytest.raises(ValueError, match="cutout_opacity"):
        unsupported.require_runtime_supported()
    manifest["capability_audit"]["cutout_opacity"] = False

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


def test_mdl_compiled_artifact_can_defer_decoded_payload_hash(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "generated.hlsl").write_text("// generated", encoding="utf-8")
    (root / "argument-block.bin").write_bytes(bytes(16))
    texture = root / "texture.bin"
    texture.write_bytes(bytes(4))
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "mdl_sdk": MDL_SDK_BUILD,
        "module": "::constant_diffuse",
        "material": "::constant_diffuse::constant_diffuse(color)",
        "code": "generated.hlsl",
        "texture_payloads": "decoded",
        "capability_audit": {
            "surface_bsdf_evaluate": True,
            "emission": False,
            "volume": False,
            "displacement": False,
            "cutout_opacity": False,
        },
        "compiled_material_hash": "1" * 32,
        "sub_expression_hashes": {
            "surface.scattering": "2" * 32,
            "geometry.normal": "3" * 32,
            "geometry.cutout_opacity": "4" * 32,
        },
        "compiler_identity": {
            "mdl_sdk": MDL_SDK_BUILD,
            "platform_id": "linux-x86_64@1",
            "semantic_identity": "b" * 64,
            "build_identity": "c" * 64,
            "backend_manifest_sha256": "d" * 64,
            "sdk_archive_sha256": "e" * 64,
            "bridge_executable_sha256": "a" * 64,
            "stb_commit": STB_COMMIT,
            "stb_image_sha256": STB_IMAGE_SHA256,
            "codegen_options": CODEGEN_OPTIONS,
        },
        "argument_block": {"path": "argument-block.bin", "size": 16},
        "parameters": [],
        "ro_data": [],
        "textures": [
            {
                "index": 1,
                "shape": "bsdf_data",
                "data": "texture.bin",
                "width": 1,
                "height": 1,
                "depth": 1,
                "pixel_type": "Float32",
            }
        ],
        "diagnostics": "",
        "files_sha256": {
            "argument-block.bin": sha256_file(root / "argument-block.bin"),
            "generated.hlsl": sha256_file(root / "generated.hlsl"),
            "texture.bin": sha256_file(texture),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    texture.write_bytes(b"bad!")
    artifact = MdlCompiledArtifact.load(root, verify_texture_payloads=False)
    with pytest.raises(ValueError, match="payload hash mismatch"):
        artifact.verify_texture_payloads()


def test_mdl_family_defers_cached_locator_texture_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_root = tmp_path / "materials"
    module_root.mkdir()
    cached = tmp_path / "cache" / "source-locators" / "locator"
    cached.mkdir(parents=True)
    compiler = type(
        "Compiler",
        (),
        {
            "cache_root": tmp_path / "cache",
            "descriptor": type(
                "Descriptor",
                (),
                {"semantic_identity": "a" * 64, "build_identity": "b" * 64},
            )(),
        },
    )()
    loaded: dict[str, object] = {}

    def fake_load(path: Path, **kwargs: object) -> object:
        loaded.update(path=path, **kwargs)
        return object()

    monkeypatch.setattr(mdl_module, "MdlCompiledArtifact", type("Artifact", (), {"load": fake_load}))
    monkeypatch.setattr(mdl_module, "create_mdl_program_provider", lambda root: compiler)
    monkeypatch.setattr("ncls.core.identity.sha256_json", lambda value: "locator")
    monkeypatch.setattr(
        "ncls.source_materials.mdl.snapshot_from_mdl_artifact", lambda *args, **kwargs: _snapshot()
    )
    family = MdlFamilyDefinition()
    family.load_snapshot(
        {
            "kind": "mdl-export",
            "module_root": str(module_root),
            "module": "::constant_diffuse",
            "export": "::constant_diffuse::constant_diffuse(color)",
        }
    )
    assert loaded["path"] == cached
    assert loaded["verify_texture_payloads"] is False


def test_mdl_decoded_texture_payloads_are_content_addressed(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    payload_a = output / "texture-1.bin"
    payload_b = output / "texture-2.bin"
    payload_a.write_bytes(b"same decoded payload")
    payload_b.write_bytes(payload_a.read_bytes())
    provider = MdlSdkProgramProvider.__new__(MdlSdkProgramProvider)
    provider.cache_root = tmp_path / "cache"
    provider._deduplicate_texture_payloads(
        output,
        {
            "texture_payloads": "decoded",
            "textures": [
                {"data": payload_a.name},
                {"data": payload_b.name},
            ],
        },
    )
    shared = next(path for path in provider.cache_root.rglob("*") if path.is_file())
    assert shared.is_file()
    assert payload_a.read_bytes() == payload_b.read_bytes() == shared.read_bytes()
    assert payload_a.stat().st_ino == payload_b.stat().st_ino == shared.stat().st_ino

    payload_a.unlink()
    payload_b.unlink()
    with pytest.raises(ValueError, match="missing"):
        provider._deduplicate_texture_payloads(
            output,
            {
                "texture_payloads": "decoded",
                "textures": [{"data": "missing.bin"}],
            },
        )


def test_mdl_decoded_texture_payloads_support_cross_device_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    payload = output / "texture.bin"
    payload.write_bytes(b"cross-device decoded payload")
    provider = MdlSdkProgramProvider.__new__(MdlSdkProgramProvider)
    provider.cache_root = tmp_path / "cache"
    original_replace = mdl_module.os.replace

    def cross_device_replace(source: Path, target: Path) -> None:
        if source == payload:
            raise OSError(errno.EXDEV, "cross-device link")
        original_replace(source, target)

    monkeypatch.setattr(mdl_module.os, "replace", cross_device_replace)
    provider._deduplicate_texture_payloads(
        output,
        {
            "texture_payloads": "decoded",
            "textures": [{"data": payload.name}],
        },
    )
    shared = next(path for path in provider.cache_root.rglob("*") if path.is_file())
    assert shared.read_bytes() == payload.read_bytes()
    assert shared.stat().st_ino != payload.stat().st_ino


def test_mdl_file_hash_cache_reuses_content_addressed_hardlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"decoded texture payload")
    alias = tmp_path / "alias.bin"
    alias.hardlink_to(source)

    calls = 0
    original = mdl_module.sha256_file

    def counting_hash(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(mdl_module, "sha256_file", counting_hash)
    mdl_module._FILE_HASH_CACHE.clear()
    assert mdl_module._cached_sha256_file(source) == mdl_module._cached_sha256_file(alias)
    assert calls == 1

    source.write_bytes(b"changed payload")
    assert mdl_module._cached_sha256_file(alias) == original(alias)
    assert calls == 2


def test_mdl_source_snapshot_hash_cache_reuses_hardlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source payload")
    alias = tmp_path / "alias.bin"
    alias.hardlink_to(source)
    calls = 0
    original = mdl_module.sha256_file

    def counting_hash(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(source_mdl_module, "sha256_file", counting_hash)
    source_mdl_module._SOURCE_HASH_CACHE.clear()
    assert source_mdl_module._cached_source_sha256(source) == source_mdl_module._cached_source_sha256(alias)
    assert calls == 1


def test_mdl_rgba16_texture_binding_preserves_all_uint16_bits(tmp_path: Path) -> None:
    path = tmp_path / "rgba16.bin"
    values = np.asarray(
        [[[0, 1, 257, 65535], [65535, 32768, 1024, 9]]], dtype=np.uint16
    )
    path.write_bytes(values.tobytes())
    suffix, payload, descriptor = _decoded_texture_binding(
        path,
        {
            "pixel_type": "Rgba_16",
            "width": 2,
            "height": 1,
            "data_origin": "top_left",
            "gamma": "linear",
        },
    )
    assert suffix == "rgba16"
    assert descriptor == {
        "kind": "texture2d",
        "dtype": "uint16",
        "shape": [1, 2, 4],
        "stride": 8,
        "alignment": 2,
        "format": "rgba16-unorm",
        "color_space": "linear",
    }
    assert len(payload) == 2 * 4 * 2
    np.testing.assert_array_equal(np.frombuffer(payload, dtype=np.uint16).reshape(1, 2, 4), values)


def test_mdl_module_discovery_requires_sorted_exact_exports_and_bridge_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "discovery"
    root.mkdir()
    document = {
        "schema": "ncls.mdl-module-discovery@1",
        "mdl_sdk": MDL_SDK_BUILD,
        "module": "::fixture",
        "materials": ["::fixture::a()", "::fixture::b(float)"],
        "diagnostics": "informational module load trace",
        "bridge_executable_sha256": "a" * 64,
    }
    (root / "discovery.json").write_text(json.dumps(document), encoding="utf-8")
    discovery = MdlModuleDiscovery.load(
        root,
        expected_module="::fixture",
        expected_bridge_sha256="a" * 64,
    )
    assert discovery.materials == ("::fixture::a()", "::fixture::b(float)")

    document["materials"].reverse()
    (root / "discovery.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted and unique"):
        MdlModuleDiscovery.load(root)


def test_mdl_reference_protocol_schemas_are_versioned_json_schema() -> None:
    schema_root = Path(__file__).parents[2] / "references/mdl-vmaterials2-v1/schemas"
    expected = {
        "mdl-source.schema.json": "ncls.mdl-source@1",
        "mdl-compiled-artifact.schema.json": "ncls.mdl-compiled-artifact@1",
        "mdl-vmaterials-family-catalog.schema.json": "ncls.mdl-vmaterials-family-catalog@1",
        "mdl-native-protocol.schema.json": "ncls.mdl-native-protocol@1",
            "mdl-oracle-request.schema.json": "ncls.mdl-oracle-request@1",
            "mdl-oracle-result.schema.json": "ncls.mdl-oracle-result@1",
            "mdl-metal-opaque-v1.schema.json": "ncls.mdl-metal-opaque-registry@1",
        }
    assert {path.name for path in schema_root.glob("*.json")} == set(expected)
    for name, schema_id in expected.items():
        value = json.loads((schema_root / name).read_text(encoding="utf-8"))
        assert value["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert value["$id"] == schema_id
