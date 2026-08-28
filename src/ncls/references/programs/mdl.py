from __future__ import annotations

from pathlib import Path

import numpy as np

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


BACKEND_SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl.slang"
RUNTIME_SHADER = PROJECT_ROOT / "shaders/ncls/reference_backends/mdl_runtime.slangh"
_CAPABILITIES = (
    BackendCapability.PREPARE
    | BackendCapability.EVALUATE
    | BackendCapability.SAMPLE
    | BackendCapability.PDF
    | BackendCapability.ANISOTROPIC_FRAME
    | BackendCapability.REVERSE_PDF
    | BackendCapability.DELTA_EVENTS
    | BackendCapability.TRANSMISSION
)
_IMPLEMENTATION = sha256_json(
    {
        "files": implementation_identity(
            (
                Path(__file__),
                PROJECT_ROOT / "src/ncls/references/mdl.py",
                BACKEND_SHADER,
                RUNTIME_SHADER,
            )
        ),
        "mdl_sdk": MDL_SDK_BUILD,
        "falcor": "8.0-9dc819c162b2070335c65060436041690b7937f8",
        "slang": "2024.1.34",
        "stb": {"commit": STB_COMMIT, "stb_image_sha256": STB_IMAGE_SHA256},
        "codegen_options": CODEGEN_OPTIONS,
    }
)


class MdlReferenceProgram(FileReferenceProgram):
    shader = BACKEND_SHADER
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
        closure = slang_module_closure(BACKEND_SHADER)
        return RuntimePayload(
            BACKEND_SHADER.relative_to(PROJECT_ROOT).as_posix(),
            closure,
            {},
            {},
            int(_CAPABILITIES),
        )

    def compile_material(self, snapshot: SourceSnapshot) -> MaterialPayload:
        self.validate_snapshot(snapshot)
        source = MdlMaterialSource.from_snapshot(snapshot)
        artifact = MdlSdkCompilerBridge(source.module_root).compile_snapshot(snapshot)
        texture_descriptors = artifact.manifest.get("textures", [])
        if any(
            texture.get("shape") not in {"2d", "bsdf_data"}
            for texture in texture_descriptors
        ):
            raise ValueError("MDL runtime accepts 2D and SDK BSDF-data textures only")
        ro_segments = artifact.manifest.get("ro_data", [])
        if len(ro_segments) > 1:
            raise ValueError("MDL reference supports at most one read-only data segment")
        sdk_types = (
            PROJECT_ROOT
            / "external"
            / MDL_SDK_DIRECTORY
            / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"
        )
        if not sdk_types.is_file():
            raise FileNotFoundError("锁定的 MDL SDK 未获取；无法构建 MDL reference runtime")
        generated_source = "\n".join(
            (
                "#define MDL_NUM_TEXTURE_RESULTS 16",
                "#define MDL_DF_HANDLE_SLOT_MODE -1",
                "struct NclsMdlRendererState { float3 view_direction; };",
                "#define RENDERER_STATE_TYPE NclsMdlRendererState",
                f"#define NCLS_MDL_TEXTURE_COUNT {max(1, len(texture_descriptors))}",
                sdk_types.read_text(encoding="utf-8"),
                RUNTIME_SHADER.read_text(encoding="utf-8"),
                artifact.hlsl,
            )
        ).encode("utf-8")
        argument_size = max(16, ((len(artifact.argument_block) + 15) // 16) * 16)
        argument_block = artifact.argument_block + bytes(
            argument_size - len(artifact.argument_block)
        )
        blobs = {
            "generated-module": generated_source,
            "argument-block": argument_block,
        }
        descriptors = {
            "generated-module": {
                "kind": "slang-module-source",
                "module_name": "NclsMdlGenerated",
                "dtype": "utf8-source",
                "shape": [len(generated_source)],
                "stride": 1,
                "alignment": 1,
                "usage": "NclsMdlGenerated",
            },
            "argument-block": {
                "kind": "structured-buffer",
                "dtype": "float32",
                "shape": [argument_size // 4],
                "stride": 16,
                "alignment": 16,
                "usage": "gMdlArgumentBlock",
            },
        }
        ro_payload = (
            (artifact.root / str(ro_segments[0]["path"])).read_bytes()
            if ro_segments
            else b""
        )
        ro_size = max(16, ((len(ro_payload) + 15) // 16) * 16)
        blobs["ro-data"] = ro_payload + bytes(ro_size - len(ro_payload))
        descriptors["ro-data"] = {
            "kind": "structured-buffer",
            "dtype": "float32",
            "shape": [ro_size // 4],
            "stride": 16,
            "alignment": 16,
            "usage": "gMdlRoData",
        }
        resources: dict[str, bytes] = {}
        resource_descriptors: dict[str, dict[str, object]] = {}
        for texture in texture_descriptors:
            index = int(texture["index"])
            if texture["shape"] == "2d":
                data = texture.get("data")
                if data is None:
                    path = Path(str(texture["path"]))
                    name = f"texture-{index}{path.suffix.lower()}"
                    resources[name] = path.read_bytes()
                    resource_descriptors[name] = {
                        "kind": "texture2d",
                        "dtype": "encoded-image",
                        "shape": [int(texture["height"]), int(texture["width"]), 4],
                        "stride": 1,
                        "alignment": 1,
                        "format": "rgba8-unorm",
                        "color_space": "srgb" if texture.get("gamma") == "srgb" else "linear",
                        "usage": f"gMdlTexture2D{index - 1}",
                    }
                    continue
                path = artifact.root / str(data)
                pixel_type = str(texture.get("pixel_type", ""))
                channels = {"Sint8": 1, "Rgb": 3, "Rgba": 4}.get(pixel_type)
                if channels is None:
                    raise ValueError(f"unsupported MDL decoded texture pixel type: {pixel_type}")
                values = bytearray(path.read_bytes())
                width = int(texture["width"])
                height = int(texture["height"])
                if len(values) != width * height * channels:
                    raise ValueError("MDL decoded texture payload has the wrong size")
                source_values = np.frombuffer(values, dtype=np.uint8).reshape(
                    height, width, channels
                )
                if channels == 1:
                    encoded = np.repeat(source_values, 4, axis=2)
                elif channels == 3:
                    encoded = np.empty((height, width, 4), dtype=np.uint8)
                    encoded[..., :3] = source_values
                    encoded[..., 3] = 255
                else:
                    encoded = source_values.copy()
                if texture.get("data_origin") == "lower_left":
                    encoded = encoded[::-1].copy()
                elif texture.get("data_origin") != "top_left":
                    raise ValueError("MDL decoded texture has an unsupported row origin")
                name = f"texture-{index}.rgba8"
                resources[name] = np.ascontiguousarray(encoded).tobytes()
                resource_descriptors[name] = {
                    "kind": "texture2d",
                    "dtype": "uint8",
                    "shape": [height, width, 4],
                    "stride": 4,
                    "alignment": 1,
                    "format": "rgba8-unorm",
                    "color_space": "srgb" if texture.get("gamma") == "srgb" else "linear",
                    "usage": f"gMdlTexture2D{index - 1}",
                }
            else:
                path = artifact.root / str(texture["data"])
                name = f"texture-{index}.r32f"
                resources[name] = path.read_bytes()
                resource_descriptors[name] = {
                    "kind": "texture3d",
                    "dtype": "float32",
                    "shape": [int(texture["depth"]), int(texture["height"]), int(texture["width"])],
                    "stride": 4,
                    "alignment": 16,
                    "format": "r32-float",
                    "color_space": "linear",
                    "usage": f"gMdlTexture3D{index - 1}",
                }
        return MaterialPayload(
            snapshot.snapshot_id,
            blobs,
            descriptors,
            resources,
            resource_descriptors,
            {
                "mdl-textures": {
                    "kind": "sampler",
                    "usage": "gMdlTextureSampler",
                    "filter": "linear",
                    "address_mode": "wrap",
                }
            },
        )


REFERENCE_PROGRAM_DEFINITION = MdlReferenceProgram()
