from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

falcor = pytest.importorskip("falcor")

from ncls.data.falcor import create_falcor_device

KERNEL_ROOT = Path(__file__).resolve().with_name("kernels")


def _texture(device, offset: float):
    texture = device.create_texture(
        width=4,
        height=4,
        format=falcor.ResourceFormat.RGBA16Float,
        mip_levels=2,
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    base = np.arange(16, dtype=np.float16).reshape(4, 4, 1) + np.float16(offset)
    base = np.ascontiguousarray(np.broadcast_to(base, (4, 4, 4)))
    coarse = np.full((2, 2, 4), 42.0 + offset, dtype=np.float16)
    texture.from_numpy(base, mip_level=0)
    texture.from_numpy(coarse, mip_level=1)
    return texture


@pytest.mark.falcor
def test_nvidia_runtime_fetch_uses_stochastic_adjacent_mip_and_wrap_bilinear() -> None:
    device = create_falcor_device(falcor)
    compute = falcor.ComputePass(
        device,
        file=KERNEL_ROOT / "nvidia_latent_fetch.cs.slang",
        cs_entry="main",
    )
    output = device.create_structured_buffer(
        struct_size=16,
        element_count=6,
        bind_flags=(
            falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
        ),
    )
    compute.globals.gLatent0 = _texture(device, 0.0)
    compute.globals.gLatent1 = _texture(device, 100.0)
    compute.globals.gLatentSampler = device.create_sampler(
        mag_filter=falcor.TextureFilteringMode.Linear,
        min_filter=falcor.TextureFilteringMode.Linear,
        mip_filter=falcor.TextureFilteringMode.Point,
        address_mode_u=falcor.TextureAddressingMode.Wrap,
        address_mode_v=falcor.TextureAddressingMode.Wrap,
        address_mode_w=falcor.TextureAddressingMode.Wrap,
    )
    compute.globals.gOutput = output
    compute.execute(threads_x=3)
    actual = output.to_numpy().view(np.float32).reshape(6, 4).copy()
    device.end_frame()

    # lod=0.25: random 0.20 selects mip1, random 0.30 selects mip0.
    # uv=(0,0) with wrap/linear averages base corners 0,3,12,15.
    expected = np.asarray(
        (
            (42.0,) * 4,
            (142.0,) * 4,
            (5.0,) * 4,
            (105.0,) * 4,
            (7.5,) * 4,
            (107.5,) * 4,
        ),
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
