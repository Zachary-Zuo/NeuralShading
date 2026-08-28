from __future__ import annotations

from pathlib import Path

from ncls.core.material import LayerStackIR, MaterialProgram, canonicalize_layer_stack, pack_layer_stack
from ncls.core.scattering import BackendCapability, MaterialPayload, ReferenceProgramDescriptor
from ncls.core.source import SourceSnapshot

from .base import FileReferenceProgram, PROJECT_ROOT, implementation_identity


SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/layer_stack.slang"


class LayerStackReferenceProgram(FileReferenceProgram):
    shader = SHADER
    descriptor = ReferenceProgramDescriptor(
        "ncls.layer-stack-random-walk", 1, "LayerStack random-walk reference",
        "ncls.layer-stack@1", 1, implementation_identity((Path(__file__), SHADER)),
        "ncls.scattering-backend@1",
        int(BackendCapability.PREPARE | BackendCapability.EVALUATE | BackendCapability.SAMPLE | BackendCapability.PDF | BackendCapability.ANISOTROPIC_FRAME | BackendCapability.REVERSE_PDF),
        {"maximum_prepare_steps": 1, "maximum_evaluate_steps": 64, "maximum_state_bytes": 752, "maximum_reads": 64},
    )

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        self.validate_snapshot(snapshot)
        if isinstance(snapshot.native_object, LayerStackIR):
            stack = snapshot.native_object
        else:
            stack = canonicalize_layer_stack(MaterialProgram.from_json(snapshot.native_payload.decode("utf-8")))
        payload = pack_layer_stack(stack)
        return MaterialPayload(snapshot.snapshot_id, {"compiled-material": payload}, {
            "compiled-material": {"kind": "structured-buffer", "dtype": "uint8", "shape": [len(payload)], "stride": len(payload), "alignment": 16, "usage": "gNclsCompiledMaterials"}
        })


REFERENCE_PROGRAM_DEFINITION = LayerStackReferenceProgram()
