from __future__ import annotations

import ast
import json
from pathlib import Path

import torch

from ncls.data.falcor import create_falcor_device, import_falcor


SCRATCH = Path(__file__).resolve().parent


def _shared_output(device, falcor, count: int):
    return device.create_structured_buffer(
        struct_size=16,
        element_count=count,
        bind_flags=(
            falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared
        ),
    )


def _assert_no_host_readback_call() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = "to_" + "numpy"
    if any(isinstance(node, ast.Attribute) and node.attr == forbidden for node in ast.walk(tree)):
        raise AssertionError("interop probe must not call the Falcor host readback API")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Falcor interop probe")
    _assert_no_host_readback_call()
    falcor = import_falcor()
    device = create_falcor_device(falcor)

    count = 32
    output = _shared_output(device, falcor, count)
    compute = falcor.ComputePass(device, file=SCRATCH / "interop_probe.cs.slang", cs_entry="main")
    compute.globals.gOutput = output
    compute.globals.gCount = count
    compute.execute(threads_x=count)
    device.render_context.wait_for_falcor()
    target = output.to_torch([count, 4], falcor.float32)
    if not target.is_cuda or target.device.type != "cuda":
        raise AssertionError("Falcor shared output must become a CUDA tensor")

    parameter = torch.nn.Parameter(torch.zeros_like(target))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(parameter, target)
    loss.backward()
    if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not torch.any(parameter.grad != 0):
        raise AssertionError("CUDA loss/backward must produce a finite nonzero gradient")
    optimizer.step()

    package_output = _shared_output(device, falcor, 1)
    package_program = (SCRATCH / "package-runtime" / "program.slang").resolve()
    package_compute = falcor.ComputePass(device, file=package_program, cs_entry="main")
    package_compute.globals.gOutput = package_output
    package_compute.execute(threads_x=1)
    device.render_context.wait_for_falcor()
    package_result = package_output.to_torch([1, 4], falcor.float32)
    expected = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=package_result.device)
    if not torch.equal(package_result, expected):
        raise AssertionError("package shader did not resolve the stable scattering contract")

    print(
        json.dumps(
            {
                "cuda_device": torch.cuda.get_device_name(target.device),
                "interop_tensor_device": str(target.device),
                "interop_tensor_pointer": target.data_ptr(),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "gradient_nonzero": True,
                "package_program": package_program.as_posix(),
                "package_contract_version": int(package_result[0, 0].item()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
