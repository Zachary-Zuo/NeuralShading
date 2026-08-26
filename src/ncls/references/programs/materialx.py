from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.core.scattering import BackendCapability, MaterialPayload, ReferenceProgramDescriptor
from ncls.core.source import SourceSnapshot

from .base import FileReferenceProgram, PROJECT_ROOT, implementation_identity


SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/materialx.slang"


class MaterialXReferenceProgram(FileReferenceProgram):
    shader = SHADER
    descriptor = ReferenceProgramDescriptor(
        "ncls.materialx-polyhaven", 1, "MaterialX 1.39.4 reference", "materialx.document@1.39.4", 1,
        implementation_identity((Path(__file__), SHADER)), "ncls.scattering-backend@1",
        int(BackendCapability.PREPARE | BackendCapability.EVALUATE | BackendCapability.SAMPLE | BackendCapability.PDF | BackendCapability.ANISOTROPIC_FRAME | BackendCapability.REVERSE_PDF),
        {"maximum_prepare_steps": 1, "maximum_evaluate_steps": 1, "maximum_state_bytes": 128, "maximum_reads": 16},
    )

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        self.validate_snapshot(snapshot)
        inputs = snapshot.editor_metadata.get("resolved_inputs")
        paths = snapshot.editor_metadata.get("resource_paths", {})
        if not isinstance(inputs, bytes) or len(inputs) != 24 * 4 or not isinstance(paths, dict):
            raise ValueError("MaterialX reference package compilation requires resolved inputs/resources")
        bindings = {
            "base-color": "gNclsMaterialXBaseColor", "roughness": "gNclsMaterialXRoughness",
            "metalness": "gNclsMaterialXMetalness", "normal": "gNclsMaterialXNormalMap",
            "displacement": "gNclsMaterialXDisplacement",
        }
        resources = {}
        resource_descriptors = {}
        for name, path_value in paths.items():
            path = Path(str(path_value))
            resource_name = f"{name}{path.suffix.lower()}"
            resources[resource_name] = path.read_bytes()
            resource_descriptors[resource_name] = {
                "dtype": "image", "shape": [path.stat().st_size], "stride": 1,
                "alignment": 1, "usage": bindings[str(name)],
            }
        return MaterialPayload(snapshot.snapshot_id, {"resolved-inputs": inputs}, {
            "resolved-inputs": {"dtype": "float32", "shape": [24], "stride": 4, "alignment": 16, "usage": "gNclsMaterialXInputs"}
        }, resources, resource_descriptors)


REFERENCE_PROGRAM_DEFINITION = MaterialXReferenceProgram()
