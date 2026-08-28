from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.core.scattering import BackendCapability, MaterialPayload, ReferenceProgramDescriptor
from ncls.core.source import SourceSnapshot
from ncls.source_materials import MerlBrdfReference, MerlMaterial

from .base import FileReferenceProgram, PROJECT_ROOT, implementation_identity


SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/merl.slang"


class MerlReferenceProgram(FileReferenceProgram):
    shader = SHADER
    descriptor = ReferenceProgramDescriptor(
        "ncls.merl-brdf", 1, "MERL measured BRDF reference", "merl.measured-brdf@1", 1,
        implementation_identity((Path(__file__), SHADER)), "ncls.scattering-backend@1",
        int(BackendCapability.PREPARE | BackendCapability.EVALUATE | BackendCapability.SAMPLE | BackendCapability.PDF | BackendCapability.ANISOTROPIC_FRAME | BackendCapability.REVERSE_PDF),
        {"maximum_prepare_steps": 1, "maximum_evaluate_steps": 1, "maximum_state_bytes": 32, "maximum_reads": 1},
    )

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        self.validate_snapshot(snapshot)
        material = snapshot.native_object if isinstance(snapshot.native_object, MerlMaterial) else MerlMaterial.from_json(snapshot.native_payload.decode("utf-8"))
        asset_root = snapshot.editor_metadata.get("asset_root")
        if not isinstance(asset_root, str):
            raise ValueError("MERL reference package compilation requires source asset_root")
        table = np.ascontiguousarray(MerlBrdfReference(material, Path(asset_root)).gpu_table(), dtype=np.float32)
        return MaterialPayload(snapshot.snapshot_id, {"brdf-table": table.tobytes()}, {
            "brdf-table": {"kind": "structured-buffer", "dtype": "float32", "shape": list(table.shape), "stride": 12, "alignment": 16, "usage": "gNclsMeasuredBrdfTable"}
        })


REFERENCE_PROGRAM_DEFINITION = MerlReferenceProgram()
