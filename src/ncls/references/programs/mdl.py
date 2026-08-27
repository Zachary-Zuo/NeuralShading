from __future__ import annotations

from pathlib import Path

from ncls.core.identity import sha256_json
from ncls.core.scattering import (
    BackendCapability,
    MaterialPayload,
    ReferenceProgramDescriptor,
    RuntimePayload,
)
from ncls.core.source import SourceSnapshot
from ncls.references.mdl import (
    CODEGEN_OPTIONS,
    MDL_SDK_BUILD,
    MDL_SDK_DIRECTORY,
    STB_COMMIT,
    STB_IMAGE_SHA256,
    MdlSdkCompilerBridge,
)
from ncls.source_materials.mdl import MdlMaterialSource

from .base import FileReferenceProgram, PROJECT_ROOT, implementation_identity, slang_module_closure


QUERY_SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_query.slang"
RUNTIME_SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
_CAPABILITIES = BackendCapability.PREPARE | BackendCapability.EVALUATE | BackendCapability.ANISOTROPIC_FRAME
_IMPLEMENTATION = sha256_json(
    {
        "files": implementation_identity(
            (Path(__file__), PROJECT_ROOT / "src/ncls/references/mdl.py", QUERY_SHADER, RUNTIME_SHADER)
        ),
        "mdl_sdk": MDL_SDK_BUILD,
        "falcor": "8.0-9dc819c162b2070335c65060436041690b7937f8",
        "slang": "2024.1.34",
        "stb": {"commit": STB_COMMIT, "stb_image_sha256": STB_IMAGE_SHA256},
        "codegen_options": CODEGEN_OPTIONS,
    }
)


class MdlReferenceProgram(FileReferenceProgram):
    shader = QUERY_SHADER
    descriptor = ReferenceProgramDescriptor(
        "ncls.mdl-vmaterials2",
        1,
        "Native MDL SDK reference",
        "mdl.program@1",
        1,
        _IMPLEMENTATION,
        "ncls.scattering-backend@1",
        int(_CAPABILITIES),
        {
            "maximum_prepare_steps": 1,
            "maximum_evaluate_steps": 1,
            "maximum_state_bytes": 512,
            "maximum_reads": 128,
        },
    )

    def compile_runtime(self) -> RuntimePayload:
        closure = slang_module_closure(QUERY_SHADER)
        runtime_name = RUNTIME_SHADER.relative_to(PROJECT_ROOT).as_posix()
        closure[runtime_name] = RUNTIME_SHADER.read_bytes()
        sdk_types = (
            PROJECT_ROOT
            / "external"
            / MDL_SDK_DIRECTORY
            / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
        )
        if not sdk_types.is_file():
            raise FileNotFoundError("锁定的 MDL SDK 未获取；无法构建 MDL reference runtime")
        return RuntimePayload(
            QUERY_SHADER.relative_to(PROJECT_ROOT).as_posix(),
            closure,
            {"mdl-target-code-types": sdk_types.read_bytes()},
            {
                "mdl-target-code-types": {
                    "dtype": "source",
                    "shape": [sdk_types.stat().st_size],
                    "stride": 1,
                    "alignment": 1,
                    "usage": "MDL SDK renderer ABI",
                }
            },
            int(_CAPABILITIES),
            {
                "MDL_NUM_TEXTURE_RESULTS": "16",
                "MDL_DF_HANDLE_SLOT_MODE": "-1",
            },
        )

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        self.validate_snapshot(snapshot)
        source = MdlMaterialSource.from_snapshot(snapshot)
        artifact = MdlSdkCompilerBridge(source.module_root).compile_snapshot(snapshot)
        blobs = {
            "generated-hlsl": artifact.hlsl.encode("utf-8"),
            "argument-block": artifact.argument_block,
        }
        descriptors = {
            "generated-hlsl": {
                "dtype": "source",
                "shape": [len(blobs["generated-hlsl"])],
                "stride": 1,
                "alignment": 1,
                "usage": "material-specific MDL HLSL",
            },
            "argument-block": {
                "dtype": "uint8",
                "shape": [len(blobs["argument-block"])],
                "stride": 1,
                "alignment": 16,
                "usage": "gMdlArgumentBlock",
            },
        }
        for index, segment in enumerate(artifact.manifest.get("ro_data", [])):
            name = f"ro-data-{index}"
            blobs[name] = (artifact.root / str(segment["path"])).read_bytes()
            descriptors[name] = {
                "dtype": "uint8",
                "shape": [len(blobs[name])],
                "stride": 1,
                "alignment": 16,
                "usage": "MDL read-only data segment",
            }
        resources: dict[str, bytes] = {}
        resource_descriptors: dict[str, dict[str, object]] = {}
        for texture in artifact.manifest.get("textures", []):
            index = int(texture["index"])
            if texture["shape"] == "2d":
                path = Path(str(texture["path"]))
            else:
                path = artifact.root / str(texture["data"])
            name = f"texture-{index}{path.suffix.lower()}"
            resources[name] = path.read_bytes()
            resource_descriptors[name] = {
                "dtype": "image" if texture["shape"] == "2d" else "float32",
                "shape": [int(texture["depth"]), int(texture["height"]), int(texture["width"])],
                "stride": 1 if texture["shape"] == "2d" else 4,
                "alignment": 1 if texture["shape"] == "2d" else 16,
                "usage": f"MDL texture index {index} ({texture['shape']}, {texture['gamma']})",
            }
        return MaterialPayload(snapshot.snapshot_id, blobs, descriptors, resources, resource_descriptors)


REFERENCE_PROGRAM_DEFINITION = MdlReferenceProgram()
