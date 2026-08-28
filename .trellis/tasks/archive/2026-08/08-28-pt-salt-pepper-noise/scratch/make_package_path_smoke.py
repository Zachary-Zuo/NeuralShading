from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ncls.bundle import write_scattering_package
from ncls.core.material import (
    DiffuseInterface,
    LayerStackIR,
    material_program_from_layer_stack,
)
from ncls.core.scattering import RuntimePayload
from ncls.references.programs.layer_stack import REFERENCE_PROGRAM_DEFINITION


@dataclass(frozen=True)
class ViewerStateSnapshot:
    family_id: str
    source_contract_version: int
    native_schema_id: str
    source_asset_sha256: str
    native_payload: bytes
    snapshot_id: str
    native_object: Any

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "source_contract_version": self.source_contract_version,
            "native_schema_id": self.native_schema_id,
            "source_asset_sha256": self.source_asset_sha256,
            "resource_hashes": {},
            "snapshot_id": self.snapshot_id,
        }


def main() -> None:
    root = Path.cwd()
    output = root / "artifacts/diagnostics/pt-salt-pepper-noise/package-path-smoke-v4"
    bundle_root = output / "bundles"
    package_root = bundle_root / "layer-stack-smoke"
    if package_root.exists() and any(package_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing package: {package_root}")

    stack = LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ())
    metadata = {"display_name": "package-path-smoke"}
    native_payload = material_program_from_layer_stack(
        stack, metadata=metadata
    ).to_json().encode("utf-8")
    source_scene_path = (
        root
        / "artifacts/diagnostics/pt-salt-pepper-noise/package-path-smoke-v3/capture/capture-scene.json"
    )
    source_scene = json.loads(source_scene_path.read_text(encoding="utf-8"))
    source_identity = source_scene["material_bindings"][0]["source"]
    snapshot = ViewerStateSnapshot(
        "ncls.layer-stack@1",
        1,
        "ncls.viewer-source-state@1",
        source_identity["source_asset_sha256"],
        native_payload,
        source_identity["state_sha256"],
        stack,
    )
    definition = REFERENCE_PROGRAM_DEFINITION
    compiled_runtime = definition.compile_runtime()
    module_closure = dict(compiled_runtime.module_closure)
    module_closure[compiled_runtime.program_module] += (
        b"\n// The viewer parity harness owns this probe-only buffer binding.\n"
        b"StructuredBuffer<uint> gNclsRuntimeWeights;\n"
    )
    runtime = RuntimePayload(
        compiled_runtime.program_module,
        module_closure,
        {"parity_probe": struct.pack("<I", 0)},
        {
            "parity_probe": {
                "dtype": "uint32",
                "shape": [1],
                "stride": 4,
                "alignment": 4,
                "usage": "gNclsRuntimeWeights",
            }
        },
        compiled_runtime.capabilities,
        compiled_runtime.defines,
    )
    manifest = write_scattering_package(
        package_root,
        program_kind="reference",
        program_key=definition.descriptor.program_key,
        program_version=definition.descriptor.version,
        program_descriptor_sha256=definition.descriptor.descriptor_sha256,
        runtime_abi=definition.descriptor.runtime_abi,
        source=snapshot,
        runtime=runtime,
        material=definition.compile_material(snapshot),
        validation={"status": "contract-compile"},
        provenance={"task": "08-28-pt-salt-pepper-noise", "purpose": "package-path-smoke"},
    )

    source_capture = (
        root / "artifacts/captures/unified-scattering-contract/layer-final/capture.json"
    )
    replay = json.loads(source_capture.read_text(encoding="utf-8"))
    replay["bundle_root"] = str(bundle_root.resolve())
    replay["viewer_scene"] = str(source_scene_path.resolve())
    replay["source_material"] = ""
    replay["slots"][0]["package_id"] = "source-reference"
    replay["slots"][0]["mode"] = "path-tracing"
    replay["slots"][1]["package_id"] = manifest.package_id
    replay["slots"][1]["mode"] = "path-tracing"
    replay_path = output / "replay.json"
    replay_path.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(replay_path)
    print(manifest.package_id)


if __name__ == "__main__":
    main()
