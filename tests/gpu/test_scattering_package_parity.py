from __future__ import annotations

from pathlib import Path

import pytest

falcor = pytest.importorskip("falcor")

from ncls.bundle import ScatteringPackage, write_scattering_package
from ncls.core.material import DiffuseInterface, LayerStackIR
from ncls.data.falcor import create_falcor_device
from ncls.references.programs.layer_stack import REFERENCE_PROGRAM_DEFINITION
from ncls.source_materials.families.layer_stack import snapshot_from_layer_stack


@pytest.mark.falcor
def test_reference_package_module_is_loaded_from_package_path(tmp_path: Path):
    stack = LayerStackIR((DiffuseInterface((0.6, 0.3, 0.1)),), ())
    snapshot = snapshot_from_layer_stack(stack)
    definition = REFERENCE_PROGRAM_DEFINITION
    root = tmp_path / "layer-stack-package"
    write_scattering_package(
        root, program_kind="reference", program_key=definition.descriptor.program_key,
        program_version=definition.descriptor.version,
        program_descriptor_sha256=definition.descriptor.descriptor_sha256,
        runtime_abi=definition.descriptor.runtime_abi, source=snapshot,
        runtime=definition.compile_runtime(), material=definition.compile_material(snapshot),
        validation={"status": "contract-compile"}, provenance={"test": True},
    )
    binding = ScatteringPackage.open(root).create_binding()
    assert binding.program_module.is_relative_to(root.resolve())
    probe = tmp_path / "probe.cs.slang"
    probe.write_text(
        '#include "' + binding.program_module.as_posix() + '"\n'
        'RWStructuredBuffer<uint> gOutput;\n[numthreads(1,1,1)]\n'
        'void main(uint3 p:SV_DispatchThreadID){NclsPackageBackend b=nclsCreatePackageBackend();gOutput[0]=1;}\n',
        encoding="utf-8",
    )
    device = create_falcor_device(falcor)
    compute = falcor.ComputePass(device, file=probe, cs_entry="main")
    output = device.create_structured_buffer(struct_size=4, element_count=1,
        bind_flags=falcor.ResourceBindFlags.UnorderedAccess | falcor.ResourceBindFlags.ShaderResource)
    compute.globals.gOutput = output
    compute.execute(threads_x=1)
