"""ScatteringPackage@1 的构建、验证和统一 binding。"""

from .loader import ScatteringBinding, ScatteringPackage
from .manifest import FORMAT_NAME, FORMAT_VERSION, ScatteringPackageManifest
from .writer import write_scattering_package
from .typed_texture import (
    RGBA16F_DDS_DTYPE,
    encode_rgba16f_dds,
    inspect_rgba16f_dds,
    validate_typed_resource,
)

__all__ = [
    "FORMAT_NAME", "FORMAT_VERSION", "ScatteringBinding", "ScatteringPackage",
    "ScatteringPackageManifest", "write_scattering_package",
    "RGBA16F_DDS_DTYPE", "encode_rgba16f_dds", "inspect_rgba16f_dds",
    "validate_typed_resource",
]
