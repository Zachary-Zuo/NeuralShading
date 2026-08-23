from pathlib import Path

import pytest


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.falcor
def test_scattering_contract_compiles_in_locked_slang() -> None:
    device = falcor.Device(type=falcor.DeviceType.D3D12)
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "scattering_contract.cs.slang",
        cs_entry="compileScatteringContract",
    )
    compute.execute(threads_x=1)
