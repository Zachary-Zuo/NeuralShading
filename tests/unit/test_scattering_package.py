import json

import pytest

from ncls.bundle import ScatteringPackage, write_scattering_package
from ncls.core.source import SourceSnapshot
from tests.fixtures.method_definition import METHOD_DEFINITION


def test_scattering_package_roundtrip_has_program_asset_instance_identities(tmp_path):
    source = SourceSnapshot("ncls.layer-stack@1", 1, "fixture", "a" * 64, b"{}")
    package_root = tmp_path / "package"
    manifest = write_scattering_package(
        package_root, program_kind="method", program_key=METHOD_DEFINITION.descriptor.method_key,
        program_version=1, program_descriptor_sha256=METHOD_DEFINITION.descriptor.descriptor_sha256,
        runtime_abi=METHOD_DEFINITION.descriptor.runtime_abi, source=source,
        program_payload=METHOD_DEFINITION.compile_program({}),
        asset_payload=METHOD_DEFINITION.compile_asset(source, {}),
        validation={"status": "passed"}, provenance={"test": True},
    )
    assert len({manifest.package_id, manifest.program_id, manifest.asset_id, manifest.instance_id}) == 4
    binding = ScatteringPackage.open(package_root).create_binding()
    assert binding.program.module.is_file()
    assert binding.asset.source_snapshot_id == source.snapshot_id
    assert binding.instance.program_id == binding.program.program_id
    assert binding.instance.asset_id == binding.asset.asset_id
    assert binding.program.samplers["program-sampler"]["usage"] == "gFixtureProgramSampler"
    assert binding.asset.samplers["asset-sampler"]["address_mode"] == "wrap"
    binding.program.module.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ScatteringPackage.open(package_root)


def test_scattering_package_rejects_removed_format(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"format_name": "ncls.scattering-package", "format_version": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields|unsupported"):
        ScatteringPackage.open(root)
