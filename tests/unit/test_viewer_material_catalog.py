from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ncls.core.identity import sha256_json
from ncls.viewer import (
    ViewerMaterialCatalog,
    finalize_catalog_document,
    link_parameter_view,
)


def _view(snapshot_id: str) -> dict[str, object]:
    leaf = {
        "path": "/arguments/texture_scale",
        "kind": "value",
        "label": "Texture Scale",
        "children": [],
        "value_type": "vector2",
        "value": [1.0, 1.0],
        "editable": True,
        "allowed_operations": ["set"],
        "metadata": {
            "responsibility": "coordinates",
            "reference_write": {"offset": 8, "size": 8, "mdl_type": "float2"},
            "runtime": {
                "token_index": 0,
                "continuous_word": 0,
                "discrete_word": 4,
                "type_word": 5,
                "normalization": {"default": [1.0, 1.0]},
                "derived_writes": [],
            },
        },
    }
    return {
        "schema_name": "ncls.source-parameter-view",
        "schema_version": 1,
        "family_id": "mdl.program@1",
        "source_contract_version": 1,
        "snapshot_id": snapshot_id,
        "runtime_layout": {
            "schema": "ncls.metal-fused-runtime-layout@1",
            "word_count": 64,
            "offsets": {
                "continuous": 0,
                "discrete": 4,
                "type": 5,
            },
        },
        "root": {
            "path": "/",
            "kind": "group",
            "label": "MDL Program",
            "children": [
                {
                    "path": "/responsibilities/coordinates",
                    "kind": "group",
                    "label": "Coordinates",
                    "children": [leaf],
                    "editable": False,
                    "allowed_operations": [],
                }
            ],
            "editable": False,
            "allowed_operations": [],
        },
    }


def _document() -> dict[str, object]:
    snapshot = "8" * 64
    entry = {
        "export_id": "1" * 64,
        "display_name": "Brass Sheet",
        "metal": "brass",
        "finish": "sheet",
        "graph_id": "2" * 64,
        "texture_set_id": "3" * 64,
        "parameter_schema_id": "4" * 64,
        "source_snapshot_id": snapshot,
        "artifact_sha256": "5" * 64,
        "artifact_root": "reference/one",
        "package_id": "6" * 64,
        "package_root": "packages/one",
        "program_id": "7" * 64,
        "asset_id": "9" * 64,
        "instance_id": "a" * 64,
        "parameter_view": _view(snapshot),
    }
    return finalize_catalog_document(
        {
            "schema_name": "ncls.viewer-material-catalog",
            "schema_version": 1,
            "registry": {
                "identity": "b" * 64,
                "sha256": "c" * 64,
                "opaque_entry_count": 1,
                "rejected_cutout_count": 145,
            },
            "checkpoint": {
                "sha256": "d" * 64,
                "checkpoint_descriptor_sha256": "0" * 64,
                "runtime_descriptor_sha256": "f" * 64,
                "compatibility": "exact-diagnostic-evaluator-preview",
                "method_key": "metal-fused-neural-material",
                "step": 20_000,
                "phase": "joint-coarse-to-fine",
            },
            "reference_runtime": {
                "mdl_sdk": "2025.0.0-387700.1252",
                "target_code_types": {"path": "runtime/types.hlsl", "sha256": "e" * 64},
                "renderer_runtime": {"path": "runtime/mdl.slangh", "sha256": "f" * 64},
            },
            "default_export_id": "1" * 64,
            "entries": [entry],
        }
    )


def _rehash(value: dict[str, object]) -> None:
    value.pop("catalog_id", None)
    value["catalog_id"] = sha256_json(value)


def test_viewer_material_catalog_accepts_linked_typed_entry_without_loading_payloads(
    tmp_path: Path,
) -> None:
    catalog = ViewerMaterialCatalog.from_dict(
        _document(), source_path=tmp_path / "catalog.json", verify_payloads=False
    )
    assert catalog.checkpoint_step == 20_000
    assert catalog.checkpoint_phase == "joint-coarse-to-fine"
    assert catalog.checkpoint_compatibility == "exact-diagnostic-evaluator-preview"
    assert catalog.default_export_id == catalog.entries[0].export_id
    assert catalog.entries[0].metal == "brass"


def test_viewer_material_catalog_rejects_tamper_duplicate_and_unsafe_path(
    tmp_path: Path,
) -> None:
    tampered = _document()
    tampered["checkpoint"]["step"] = 19_999  # type: ignore[index]
    with pytest.raises(ValueError, match="catalog_id"):
        ViewerMaterialCatalog.from_dict(
            tampered, source_path=tmp_path / "catalog.json", verify_payloads=False
        )

    duplicate = _document()
    duplicate["entries"].append(deepcopy(duplicate["entries"][0]))  # type: ignore[union-attr,index]
    duplicate["registry"]["opaque_entry_count"] = 2  # type: ignore[index]
    _rehash(duplicate)
    with pytest.raises(ValueError, match="duplicate export_id"):
        ViewerMaterialCatalog.from_dict(
            duplicate, source_path=tmp_path / "catalog.json", verify_payloads=False
        )

    unsafe = _document()
    unsafe["entries"][0]["artifact_root"] = "../escape"  # type: ignore[index]
    _rehash(unsafe)
    with pytest.raises(ValueError, match="stay below catalog root"):
        ViewerMaterialCatalog.from_dict(
            unsafe, source_path=tmp_path / "catalog.json", verify_payloads=False
        )


def test_viewer_material_catalog_rejects_missing_reference_write(
    tmp_path: Path,
) -> None:
    value = _document()
    metadata = value["entries"][0]["parameter_view"]["root"]["children"][0][
        "children"
    ][0]["metadata"]  # type: ignore[index]
    metadata.pop("reference_write")
    _rehash(value)
    with pytest.raises(ValueError, match="reference write"):
        ViewerMaterialCatalog.from_dict(
            value, source_path=tmp_path / "catalog.json", verify_payloads=False
        )

    wrong_size = _document()
    write = wrong_size["entries"][0]["parameter_view"]["root"]["children"][0][
        "children"
    ][0]["metadata"]["reference_write"]  # type: ignore[index]
    write["size"] = 4
    _rehash(wrong_size)
    with pytest.raises(ValueError, match="type/size"):
        ViewerMaterialCatalog.from_dict(
            wrong_size,
            source_path=tmp_path / "catalog.json",
            verify_payloads=False,
        )


def test_link_parameter_view_groups_coordinates_and_frame_without_name_switches() -> None:
    snapshot = "8" * 64
    view = _view(snapshot)
    frame = deepcopy(view["root"]["children"][0]["children"][0])  # type: ignore[index]
    frame["path"] = "/arguments/enable_round_corners"
    frame["label"] = "Enable Round Corners"
    frame["value_type"] = "bool"
    frame["value"] = False
    frame["metadata"]["runtime"]["normalization"]["default"] = False
    view["root"]["children"][0]["children"].append(frame)  # type: ignore[index]
    linked = link_parameter_view(
        view,
        [
            {
                "name": "texture_scale",
                "type": "float2",
                "editable": True,
                "offset": 8,
                "size": 8,
                "responsibility": "coordinates",
            },
            {
                "name": "enable_round_corners",
                "type": "bool",
                "editable": True,
                "offset": 44,
                "size": 1,
                "responsibility": "frame",
            },
        ],
    )
    groups = linked["root"]["children"]
    assert [item["path"] for item in groups] == [
        "/responsibilities/coordinates",
        "/responsibilities/frame",
    ]
    writes = {
        node["path"]: node["metadata"]["reference_write"]
        for group in groups
        for node in group["children"]
    }
    assert writes["/arguments/texture_scale"] == {
        "offset": 8,
        "size": 8,
        "mdl_type": "float2",
    }
    assert writes["/arguments/enable_round_corners"] == {
        "offset": 44,
        "size": 1,
        "mdl_type": "bool",
    }
