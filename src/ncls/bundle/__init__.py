"""ScatteringPackage@2 的program/asset/instance构建、验证和原子binding。"""

from .loader import (
    AssetBinding,
    InstanceBinding,
    ProgramRuntime,
    ScatteringBinding,
    ScatteringPackage,
)
from .manifest import FORMAT_NAME, FORMAT_VERSION, ScatteringPackageManifest
from .writer import write_scattering_package
from .typed_texture import (
    RGBA16F_DDS_DTYPE,
    RGBA8_SNORM_DDS_DTYPE,
    encode_rgba16f_dds,
    encode_rgba8_snorm_dds,
    inspect_rgba16f_dds,
    inspect_rgba8_snorm_dds,
    validate_typed_resource,
)

__all__ = [
    "AssetBinding", "FORMAT_NAME", "FORMAT_VERSION", "InstanceBinding",
    "ProgramRuntime", "ScatteringBinding", "ScatteringPackage",
    "ScatteringPackageManifest", "write_scattering_package",
    "RGBA16F_DDS_DTYPE", "RGBA8_SNORM_DDS_DTYPE",
    "encode_rgba16f_dds", "encode_rgba8_snorm_dds",
    "inspect_rgba16f_dds", "inspect_rgba8_snorm_dds",
    "validate_typed_resource",
]
