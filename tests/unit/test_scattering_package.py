import copy
import json

import pytest

from ncls.bundle import ScatteringPackage, write_scattering_package
from ncls.core.scattering import InstancePayload
from ncls.core.source import SourceSnapshot
from tests.fixtures.method_definition import METHOD_DEFINITION


def test_scattering_package_roundtrip_has_program_asset_instance_identities(tmp_path):
    source = SourceSnapshot("ncls.layer-stack@1", 1, "fixture", "a" * 64, b"{}")
    package_root = tmp_path / "package"
    manifest = write_scattering_package(
        package_root, program_kind="method", program_key=METHOD_DEFINITION.descriptor.method_key,
        program_version=1, program_descriptor_sha256=METHOD_DEFINITION.descriptor.descriptor_sha256,
        runtime_abi=METHOD_DEFINITION.descriptor.runtime_abi, source=source,
        program_payload=METHOD_DEFINITION.compile_program({}),
        asset_payload=METHOD_DEFINITION.compile_asset(source, {}),
        validation={"status": "passed"}, provenance={"test": True},
    )
    assert len({manifest.package_id, manifest.program_id, manifest.asset_id, manifest.instance_id}) == 4
    binding = ScatteringPackage.open(package_root).create_binding()
    assert binding.program.module.is_file()
    assert binding.asset.source_snapshot_id == source.snapshot_id
    assert binding.instance.program_id == binding.program.program_id
    assert binding.instance.asset_id == binding.asset.asset_id
    assert binding.program.samplers["program-sampler"]["usage"] == "gFixtureProgramSampler"
    assert binding.asset.samplers["asset-sampler"]["address_mode"] == "wrap"
    binding.program.module.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ScatteringPackage.open(package_root)


def test_scattering_package_rejects_removed_format(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"format_name": "ncls.scattering-package", "format_version": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields|unsupported"):
        ScatteringPackage.open(root)


def _editable_parameter_view(snapshot_id: str) -> dict:
    return {
        "schema_name": "ncls.source-parameter-view",
        "schema_version": 1,
        "family_id": "ncls.layer-stack@1",
        "source_contract_version": 1,
        "snapshot_id": snapshot_id,
        "root": {
            "path": "/",
            "kind": "group",
            "label": "Material",
            "children": [
                {
                    "path": "/roughness",
                    "kind": "value",
                    "label": "Roughness",
                    "children": [],
                    "editable": True,
                    "allowed_operations": ["set"],
                    "value_type": "float",
                    "value": 0.25,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "metadata": {
                        "runtime": {
                            "token_index": 0,
                            "continuous_word": 0,
                            "discrete_word": 4,
                            "type_word": 5,
                            "normalization": {
                                "default": 0.25,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "derived_writes": [
                                {"word": 6, "operation": "copy", "component": 0}
                            ],
                        }
                    },
                }
            ],
            "editable": False,
            "allowed_operations": [],
        },
        "runtime_layout": {
            "schema": "ncls.fixture-raw@1",
            "word_count": 16,
            "offsets": {"continuous": 0},
        },
    }


def _editable_instance(snapshot_id: str) -> InstancePayload:
    return InstancePayload(
        {"compiled_material_index": 0},
        {"raw": bytes(64), "compiled": bytes(64)},
        {
            "raw": {
                "dtype": "uint32",
                "shape": [1],
                "stride": 64,
                "alignment": 16,
                "usage": "gFixtureRaw",
                "kind": "mutable-structured-buffer",
            },
            "compiled": {
                "dtype": "uint32",
                "shape": [1],
                "stride": 64,
                "alignment": 16,
                "usage": "gNclsCompiledMaterials",
                "kind": "mutable-structured-buffer",
            },
        },
        {
            "schema": "ncls.typed-material-editor@1",
            "parameter_view": _editable_parameter_view(snapshot_id),
            "raw_usage": "gFixtureRaw",
            "compiled_usage": "gNclsCompiledMaterials",
        },
        {"entry_point": "nclsCompileMaterial", "thread_group_size": [32, 1, 1]},
    )


def test_editable_instance_roundtrip_preserves_typed_blobs_and_compiler(tmp_path):
    source = SourceSnapshot("ncls.layer-stack@1", 1, "fixture", "a" * 64, b"{}")
    root = tmp_path / "editable"
    write_scattering_package(
        root,
        program_kind="method",
        program_key=METHOD_DEFINITION.descriptor.method_key,
        program_version=1,
        program_descriptor_sha256=METHOD_DEFINITION.descriptor.descriptor_sha256,
        runtime_abi=METHOD_DEFINITION.descriptor.runtime_abi,
        source=source,
        program_payload=METHOD_DEFINITION.compile_program({}),
        asset_payload=METHOD_DEFINITION.compile_asset(source, {}),
        instance_payload=_editable_instance(source.snapshot_id),
        validation={"status": "passed"},
        provenance={"test": True},
    )
    binding = ScatteringPackage.open(root).create_binding()
    assert set(binding.instance.files) == {
        "instance/blob/compiled",
        "instance/blob/raw",
    }
    assert binding.instance.descriptor["editor"]["raw_usage"] == "gFixtureRaw"
    assert binding.instance.descriptor["compiler"]["thread_group_size"] == [32, 1, 1]


def test_editable_instance_rejects_out_of_bounds_runtime_write():
    view = _editable_parameter_view("a" * 64)
    malformed = copy.deepcopy(view)
    malformed["root"]["children"][0]["metadata"]["runtime"]["derived_writes"][0]["word"] = 16
    with pytest.raises(ValueError, match="derived write"):
        InstancePayload(
            {"compiled_material_index": 0},
            {"raw": bytes(64), "compiled": bytes(64)},
            {
                "raw": {
                    "dtype": "uint32", "shape": [1], "stride": 64,
                    "alignment": 16, "usage": "gRaw", "kind": "mutable-structured-buffer",
                },
                "compiled": {
                    "dtype": "uint32", "shape": [1], "stride": 64,
                    "alignment": 16, "usage": "gCompiled", "kind": "mutable-structured-buffer",
                },
            },
            {
                "schema": "ncls.typed-material-editor@1",
                "parameter_view": malformed,
                "raw_usage": "gRaw",
                "compiled_usage": "gCompiled",
            },
            {"entry_point": "compile", "thread_group_size": [1, 1, 1]},
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda view: view["runtime_layout"]["offsets"].update(
                {"continuous": 16}
            ),
            "runtime_layout",
        ),
        (
            lambda view: view["root"]["children"][0].update(
                {"allowed_operations": ["delete"]}
            ),
            "operations",
        ),
        (
            lambda view: view["root"]["children"][0]["metadata"]["runtime"][
                "normalization"
            ].update({"default": "not-a-float"}),
            "scalar value",
        ),
    ),
)
def test_editable_instance_rejects_runtime_metadata_not_supported_by_viewer(
    mutate, message: str
) -> None:
    view = _editable_parameter_view("a" * 64)
    mutate(view)
    with pytest.raises(ValueError, match=message):
        InstancePayload(
            {"compiled_material_index": 0},
            {"raw": bytes(64), "compiled": bytes(64)},
            {
                "raw": {
                    "dtype": "uint32", "shape": [1], "stride": 64,
                    "alignment": 16, "usage": "gRaw", "kind": "mutable-structured-buffer",
                },
                "compiled": {
                    "dtype": "uint32", "shape": [1], "stride": 64,
                    "alignment": 16, "usage": "gCompiled", "kind": "mutable-structured-buffer",
                },
            },
            {
                "schema": "ncls.typed-material-editor@1",
                "parameter_view": view,
                "raw_usage": "gRaw",
                "compiled_usage": "gCompiled",
            },
            {"entry_point": "compile", "thread_group_size": [1, 1, 1]},
        )
