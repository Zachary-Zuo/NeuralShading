from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ncls.core.material import (
    ABI_MAGIC,
    BINARY_SIZE,
    DiffuseInterface,
    HomogeneousMedium,
    LayerStackIR,
    MaterialNode,
    MaterialProgram,
    OperationId,
    RoughConductorInterface,
    RoughDielectricInterface,
    SheenInterface,
    canonical_material_json,
    canonicalize_layer_stack,
    material_program_from_layer_stack,
    pack_layer_stack,
    physical_material_hash,
    unpack_layer_stack,
    validate_material_program,
)
from ncls.core.material.abi_layout import render_slang_header
from ncls.cli import main as cli_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_ir(base: str = "conductor") -> LayerStackIR:
    top = RoughDielectricInterface(0.08, 0.17, 1.5, 0.25)
    medium = HomogeneousMedium((0.1, 0.2, 0.3), (0.0, 0.0, 0.0), 0.0, 0.4)
    if base == "conductor":
        bottom = RoughConductorInterface(0.3, 0.5, (0.2, 0.9, 1.1), (3.9, 2.5, 2.1), -0.4)
    elif base == "diffuse":
        bottom = DiffuseInterface((0.7, 0.3, 0.1))
    else:
        bottom = SheenInterface((0.5, 0.2, 0.1), 0.35)
    return LayerStackIR((top, bottom), (medium,))


@pytest.mark.parametrize("base", ["conductor", "diffuse", "sheen"])
def test_layer_stack_ir_binary_round_trip(base: str) -> None:
    stack = make_ir(base)
    payload = pack_layer_stack(stack)
    assert len(payload) == BINARY_SIZE == 752
    assert int.from_bytes(payload[:4], "little") == ABI_MAGIC
    restored = unpack_layer_stack(payload)
    assert [item.kind for item in restored.interfaces] == [item.kind for item in stack.interfaces]
    assert restored.interfaces[0].relative_ior == pytest.approx(1.5)  # type: ignore[union-attr]
    assert pack_layer_stack(restored) == payload


def test_layer_stack_ir_rejects_non_transmissive_intermediate_interface() -> None:
    with pytest.raises(ValueError, match="transmissive rough dielectric"):
        LayerStackIR(
            (DiffuseInterface((0.5, 0.5, 0.5)), DiffuseInterface((0.2, 0.2, 0.2))),
            (HomogeneousMedium(),),
        )


def test_material_program_round_trip_and_canonicalization() -> None:
    stack = make_ir()
    program = material_program_from_layer_stack(stack, metadata={"display_name": "测试材质"})
    restored = MaterialProgram.from_json(program.to_json())
    validate_material_program(restored)
    assert canonicalize_layer_stack(restored) == stack
    assert physical_material_hash(restored) == physical_material_hash(program)
    assert "测试材质" in restored.to_json()


def test_physical_hash_ignores_metadata_and_node_order() -> None:
    program = material_program_from_layer_stack(make_ir(), metadata={"name": "A"})
    reordered = replace(program, nodes=tuple(reversed(program.nodes)), metadata={"name": "B"})
    assert canonical_material_json(program) == canonical_material_json(reordered)
    assert physical_material_hash(program) == physical_material_hash(reordered)


def test_physical_hash_ignores_graph_node_names() -> None:
    program = material_program_from_layer_stack(make_ir())
    document = program.to_dict(include_program_id=False)
    renamed = {node["id"]: f"renamed-{index}" for index, node in enumerate(document["nodes"])}
    for node in document["nodes"]:
        node["id"] = renamed[node["id"]]
        for value in node["inputs"].values():
            connections = value if isinstance(value, list) else [value]
            for connection in connections:
                connection["node"] = renamed[connection["node"]]
    for connection in document["outputs"].values():
        if connection is not None:
            connection["node"] = renamed[connection["node"]]

    renamed_program = MaterialProgram.from_dict(document)
    assert canonical_material_json(program) != canonical_material_json(renamed_program)
    assert physical_material_hash(program) == physical_material_hash(renamed_program)


def test_program_id_detects_modified_physical_content() -> None:
    program = material_program_from_layer_stack(make_ir())
    document = program.to_dict()
    document["nodes"][0]["parameters"]["alpha_x"]["value"] = 0.5
    with pytest.raises(ValueError, match="program_id"):
        MaterialProgram.from_dict(document)


def test_registry_rejects_unknown_operation() -> None:
    program = material_program_from_layer_stack(make_ir())
    bad = replace(program.nodes[0], operation=OperationId("example", "unknown", 1))
    invalid = replace(program, nodes=(bad, *program.nodes[1:]))
    with pytest.raises(ValueError, match="unsupported operation"):
        validate_material_program(invalid)


def test_generated_slang_abi_is_current() -> None:
    shader = PROJECT_ROOT / "shaders" / "ncls" / "contracts" / "layer_stack_ir.slang"
    assert shader.read_text(encoding="utf-8") == render_slang_header()


def test_material_cli_validates_and_packs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    program_path = tmp_path / "material.json"
    binary_path = tmp_path / "material.bin"
    program_path.write_text(material_program_from_layer_stack(make_ir()).to_json(), encoding="utf-8")

    assert cli_main(["material", "validate", str(program_path)]) == 0
    assert "MaterialProgram OK" in capsys.readouterr().out
    assert cli_main(["material", "pack", str(program_path), str(binary_path)]) == 0
    assert len(binary_path.read_bytes()) == BINARY_SIZE
