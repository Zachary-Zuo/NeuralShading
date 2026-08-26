import pytest

from ncls.bundle import ScatteringPackage, write_scattering_package
from ncls.core.source import SourceSnapshot
from tests.fixtures.method_definition import METHOD_DEFINITION


def test_scattering_package_roundtrip_has_three_independent_identities(tmp_path):
    source = SourceSnapshot("ncls.layer-stack@1", 1, "fixture", "a" * 64, b"{}")
    package_root = tmp_path / "package"
    manifest = write_scattering_package(
        package_root, program_kind="method", program_key=METHOD_DEFINITION.descriptor.method_key,
        program_version=1, program_descriptor_sha256=METHOD_DEFINITION.descriptor.descriptor_sha256,
        runtime_abi=METHOD_DEFINITION.descriptor.runtime_abi, source=source,
        runtime=METHOD_DEFINITION.compile_runtime({}), material=METHOD_DEFINITION.compile_material(source, {}),
        validation={"status": "passed"}, provenance={"test": True},
    )
    assert len({manifest.package_id, manifest.program_runtime_id, manifest.material_asset_id}) == 3
    binding = ScatteringPackage.open(package_root).create_binding()
    assert binding.program_module.is_file() and binding.source_snapshot_id == source.snapshot_id
    binding.program_module.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ScatteringPackage.open(package_root)
