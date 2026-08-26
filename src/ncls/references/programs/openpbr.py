from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.core.scattering import BackendCapability, MaterialPayload, ReferenceProgramDescriptor
from ncls.core.source import SourceSnapshot
from ncls.source_materials import OpenPBRMaterial, load_openpbr_luts, resolve_openpbr_inputs

from .base import FileReferenceProgram, PROJECT_ROOT, implementation_identity


SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/openpbr.slang"


class OpenPbrReferenceProgram(FileReferenceProgram):
    shader = SHADER
    descriptor = ReferenceProgramDescriptor(
        "ncls.openpbr", 1, "OpenPBR 1.1.1 reference", "openpbr.material@1.1.1", 1,
        implementation_identity((Path(__file__), SHADER)), "ncls.scattering-backend@1",
        int(BackendCapability.PREPARE | BackendCapability.EVALUATE | BackendCapability.SAMPLE | BackendCapability.PDF | BackendCapability.ANISOTROPIC_FRAME | BackendCapability.REVERSE_PDF | BackendCapability.TRANSMISSION),
        {"maximum_prepare_steps": 1, "maximum_evaluate_steps": 1, "maximum_state_bytes": 2048, "maximum_reads": 96},
    )

    def runtime_blobs(self) -> tuple[dict[str, bytes], dict[str, dict]]:
        luts = load_openpbr_luts(PROJECT_ROOT / "external/openpbr-bsdf")
        names = {
            "ideal-dielectric-energy": luts.ideal_dielectric_energy,
            "ideal-dielectric-average": luts.ideal_dielectric_average,
            "ideal-dielectric-ratio": luts.ideal_dielectric_ratio,
            "opaque-dielectric-energy": luts.opaque_dielectric_energy,
            "opaque-dielectric-average": luts.opaque_dielectric_average,
            "ideal-metal-energy": luts.ideal_metal_energy,
            "ideal-metal-average": luts.ideal_metal_average,
            "ltc": luts.ltc,
        }
        blobs = {name: np.ascontiguousarray(value, dtype=np.float32).tobytes() for name, value in names.items()}
        descriptors = {
            name: {"dtype": "float32", "shape": list(value.shape), "stride": 16, "alignment": 16, "usage": binding}
            for (name, value), binding in zip(names.items(), (
                "gOpenPbrIdealDielectricEnergy", "gOpenPbrIdealDielectricAverage",
                "gOpenPbrIdealDielectricRatio", "gOpenPbrOpaqueDielectricEnergy",
                "gOpenPbrOpaqueDielectricAverage", "gOpenPbrIdealMetalEnergy",
                "gOpenPbrIdealMetalAverage", "gOpenPbrLtc",
            ), strict=True)
        }
        return blobs, descriptors

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        self.validate_snapshot(snapshot)
        material = snapshot.native_object if isinstance(snapshot.native_object, OpenPBRMaterial) else OpenPBRMaterial.from_json(snapshot.native_payload.decode("utf-8"))
        asset_root = snapshot.editor_metadata.get("asset_root")
        if not isinstance(asset_root, str):
            raise ValueError("OpenPBR reference package compilation requires source asset_root")
        values = np.ascontiguousarray(resolve_openpbr_inputs(material, asset_root=Path(asset_root)), dtype=np.float32)
        return MaterialPayload(snapshot.snapshot_id, {"resolved-inputs": values.tobytes()}, {
            "resolved-inputs": {"dtype": "float32", "shape": list(values.shape), "stride": 4, "alignment": 16, "usage": "gNclsCompiledMaterialValues"}
        })


REFERENCE_PROGRAM_DEFINITION = OpenPbrReferenceProgram()
