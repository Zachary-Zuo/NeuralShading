"""03 spike 的检查函数：lobe/MLP 梯度对照与吞吐基准。"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ncls.core.representations.legacy_ltc_k2.torch_eval import eval_ltc_residual
from ncls.learning.models import ConditionedSharedEvaluator

INPUT, WIDTH, RAW = 16, 64, 9
W0, B0, W1, B1 = 0, WIDTH * INPUT, WIDTH * INPUT + WIDTH, WIDTH * INPUT + WIDTH + RAW * WIDTH
PARAMETER_COUNT = B1 + RAW


def hemisphere(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    z = rng.uniform(0.05, 1.0, shape)
    phi = rng.uniform(0.0, 2.0 * np.pi, shape)
    r = np.sqrt(1.0 - z * z)
    return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1).astype(np.float32)


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = np.maximum(np.abs(expected), 1e-3 * float(np.max(np.abs(expected))) + 1e-12)
    return float(np.max(np.abs(actual - expected) / scale))


def check_lobe_gradients(spy: Any, module: Any, device: Any, rng: np.random.Generator, groups: int, directions: int) -> dict[str, Any]:
    params = {
        "amplitude": rng.uniform(0.1, 2.0, (groups, 1, 3)),
        "inverse_scale": rng.uniform(0.5, 2.0, (groups, 1, 2)),
        "shear": rng.uniform(-1.0, 1.0, (groups, 1, 3)),
        "angle": rng.uniform(-3.0, 3.0, (groups, 1)),
    }
    wi = hemisphere(rng, (directions,))
    loss_weight = rng.uniform(0.5, 1.5, (groups, directions, 3)).astype(np.float32)
    # 显式展开到 (groups, directions)，避免依赖 broadcast 维度上的梯度归约语义。
    expanded = {
        name: np.ascontiguousarray(np.broadcast_to(value, (groups, directions) + value.shape[2:]), dtype=np.float32)
        for name, value in params.items()
    }
    tensors = {name: spy.Tensor.from_numpy(device, value).with_grads() for name, value in expanded.items()}
    result = spy.Tensor.from_numpy(device, np.zeros((groups, directions, 3), np.float32)).with_grads()
    wi_batch = np.ascontiguousarray(np.broadcast_to(wi, (groups, directions, 3)))
    args = (tensors["amplitude"], tensors["inverse_scale"], tensors["shear"], tensors["angle"], wi_batch)
    module.spikeLobeProbe(*args, _result=result)
    result.grad_in = spy.Tensor.from_numpy(device, loss_weight)
    module.spikeLobeProbe.bwds(*args, _result=result)

    torch_params = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True) for name, value in params.items()}
    reference = eval_ltc_residual(torch.tensor(wi, dtype=torch.float64), **torch_params)
    (reference * torch.tensor(loss_weight, dtype=torch.float64)).sum().backward()
    errors = {"forward": relative_error(result.to_numpy(), reference.detach().numpy())}
    for name, tensor in tensors.items():
        grad = tensor.grad_out.to_numpy().reshape(groups, directions, -1).sum(axis=1)
        expected = torch_params[name].grad.detach().numpy().reshape(groups, -1)
        errors[f"grad_{name}"] = relative_error(grad, expected)
    return {"errors": errors, "pass": max(errors.values()) <= 1e-3}


def mirror_response(weights: torch.Tensor, x: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
    """spikeEvaluate 的 float64 Torch 镜像；lobe 部分直接复用 torch_eval.eval_ltc_residual。"""
    hidden = torch.nn.functional.silu(x @ weights[W0:B0].reshape(WIDTH, INPUT).T + weights[B0:W1])
    raw = hidden @ weights[W1:B1].reshape(RAW, WIDTH).T + weights[B1:]
    return eval_ltc_residual(
        wi,
        torch.nn.functional.softplus(raw[:, None, 0:3]),
        torch.exp(torch.clamp(raw[:, None, 3:5], -3.0, 3.0)),
        3.0 * torch.tanh(raw[:, None, 5:8]),
        np.pi * torch.tanh(raw[:, None, 8]),
    )


def check_mlp_gradients(spy: Any, module: Any, device: Any, rng: np.random.Generator, groups: int, directions: int, probes: int) -> dict[str, Any]:
    weights = (rng.standard_normal(PARAMETER_COUNT) * 0.2).astype(np.float32)
    x = rng.standard_normal((groups, INPUT)).astype(np.float32)
    wi = hemisphere(rng, (directions,))
    loss_weight = rng.uniform(0.5, 1.5, (groups, directions, 3)).astype(np.float32)
    x_batch = np.ascontiguousarray(np.broadcast_to(x[:, None, :], (groups, directions, INPUT)))
    wi_batch = np.ascontiguousarray(np.broadcast_to(wi, (groups, directions, 3)))
    weight_tensor = torch.tensor(weights, device="cuda", requires_grad=True)
    w0 = weight_tensor[W0:B0].reshape(WIDTH, INPUT)
    b0 = weight_tensor[B0:W1]
    w1 = weight_tensor[W1:B1].reshape(RAW, WIDTH)
    b1 = weight_tensor[B1:]
    x_tensor = torch.tensor(x_batch, device="cuda")
    hidden = module.spikeLinear0(w0, b0, x_tensor)
    hidden = module.spikeActivate(hidden)
    raw = module.spikeLinear1(w1, b1, hidden)
    result = module.spikeDecodeEvaluate(raw, torch.tensor(wi_batch, device="cuda"))
    (result * torch.tensor(loss_weight, device="cuda")).sum().backward()
    gradient = weight_tensor.grad.detach().cpu().numpy().reshape(-1)

    w64, x64, wi64 = (torch.tensor(v, dtype=torch.float64) for v in (weights, x, wi))
    lw64 = torch.tensor(loss_weight, dtype=torch.float64)
    forward_error = relative_error(result.detach().cpu().numpy(), mirror_response(w64, x64, wi64).numpy())

    def loss(w: torch.Tensor) -> float:
        return float((mirror_response(w, x64, wi64) * lw64).sum())

    indices = rng.choice(PARAMETER_COUNT, probes, replace=False)
    finite = np.empty(probes)
    for slot, index in enumerate(indices):
        step = torch.zeros_like(w64)
        step[index] = 1e-3
        finite[slot] = (loss(w64 + step) - loss(w64 - step)) / 2e-3
    error = relative_error(gradient[indices], finite)
    return {
        "probed_weights": int(probes), "forward_error": forward_error,
        "finite_difference_error": error, "pass": max(error, forward_error) <= 1e-3,
        "indices": indices.tolist(),
        "slang_gradient": gradient[indices].tolist(),
        "finite_difference": finite.tolist(),
    }


def probe_struct_params(spy: Any, module: Any, device: Any, rng: np.random.Generator) -> dict[str, Any]:
    """S2.1 把权重张量放进 NclsLobeResidualParams struct；这里探测 struct 字段是否仍能拿到 grad_out。"""
    weights = (rng.standard_normal(8)).astype(np.float32)
    x = rng.standard_normal(32).astype(np.float32)
    loss_weight = rng.uniform(0.5, 1.5, 32).astype(np.float32)
    weight_tensor = spy.Tensor.from_numpy(device, weights).with_grads()
    result = spy.Tensor.from_numpy(device, np.zeros(32, np.float32)).with_grads()
    params = {"weights": weight_tensor, "bias": 3}
    try:
        module.spikeStructProbe(params, x, _result=result)
        result.grad_in = spy.Tensor.from_numpy(device, loss_weight)
        module.spikeStructProbe.bwds(params, x, _result=result)
        gradient = weight_tensor.grad_out.to_numpy().reshape(-1)
        expected = np.zeros(8, np.float32)
        expected[3] = float(np.sum(x * loss_weight))
        error = relative_error(gradient, expected)
        return {"ok": True, "gradient_error": error, "pass": error <= 1e-3}
    except Exception as error:  # noqa: BLE001 - 失败即 spike 结论：Params 须拆成张量 + 偏移两个参数
        return {"ok": False, "error": str(error)[:4000], "pass": False}


def benchmark(spy: Any, module: Any, device: Any, rng: np.random.Generator, groups: int, directions: int, iterations: int) -> dict[str, Any]:
    weight_tensor = spy.Tensor.from_numpy(device, (rng.standard_normal(PARAMETER_COUNT) * 0.2).astype(np.float32)).with_grads()
    x_batch = rng.standard_normal((groups, directions, INPUT)).astype(np.float32)
    args = (
        weight_tensor,
        *(np.ascontiguousarray(x_batch[..., 4 * i:4 * i + 4]) for i in range(4)),
        hemisphere(rng, (groups, directions)),
    )
    result = spy.Tensor.from_numpy(device, np.zeros((groups, directions, 3), np.float32)).with_grads()
    result.grad_in = spy.Tensor.from_numpy(device, np.ones((groups, directions, 3), np.float32))

    def timed(backward: bool) -> float:
        for _ in range(3):
            module.spikeEvaluate(*args, _result=result)
        result.to_numpy()
        start = time.perf_counter()
        for _ in range(iterations):
            module.spikeEvaluate(*args, _result=result)
            if backward:
                module.spikeEvaluate.bwds(*args, _result=result)
        result.to_numpy()  # 读回强制同步
        return (time.perf_counter() - start) * 1e3 / iterations

    forward_ms, train_ms = timed(False), timed(True)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedSharedEvaluator(
        state_count=30, output_scale=np.full((30, 3), 0.1), width=128, latent_dim=32,
        prepare_blocks=2, evaluate_blocks=3, fourier_bands=4, initial_output_ratio=0.01,
    ).to(torch_device)
    state = torch.randint(0, 30, (groups,), device=torch_device)
    wo = torch.tensor(hemisphere(rng, (groups,)), device=torch_device)
    wi = torch.tensor(hemisphere(rng, (groups, directions)), device=torch_device)
    for _ in range(3):
        model(state, wo, wi).sum().backward()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        model(state, wo, wi).sum().backward()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) * 1e3 / iterations
    return {
        "batch": [groups, directions], "iterations": iterations,
        "slang_forward_ms": forward_ms, "slang_forward_backward_ms": train_ms,
        "torch_m1s_forward_backward_ms": torch_ms, "torch_device": str(torch_device),
        "throughput_ratio_vs_m1s": torch_ms / train_ms, "pass": torch_ms / train_ms >= 0.5,
    }
