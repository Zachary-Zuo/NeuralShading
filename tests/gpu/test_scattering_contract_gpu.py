from pathlib import Path

import pytest

from ncls.references.backend import create_reference_backend


falcor = pytest.importorskip("falcor")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.falcor
def test_scattering_contract_compiles_in_locked_slang() -> None:
    device = create_reference_backend()._create_device(falcor)  # noqa: SLF001
    compute = falcor.ComputePass(
        device,
        file=PROJECT_ROOT / "tests" / "gpu" / "kernels" / "scattering_contract.cs.slang",
        cs_entry="compileScatteringContract",
    )
    compute.execute(threads_x=1)
