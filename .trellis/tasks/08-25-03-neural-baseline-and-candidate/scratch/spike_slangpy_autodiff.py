"""P1.0 spike：SlangPy 对最小 [Differentiable] 核求梯度并与 Torch 对照。只在远程 GPU 机运行。

输出 JSON：slangpy/slang 版本、编译成功的权重张量写法、lobe/MLP 梯度误差、前向+反向吞吐与
M1-S Torch 的比值、Torch 互操作探测。通过判据见 TESTING.md「P1.0 spike」。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from spike_slangpy_checks import (  # noqa: E402
    benchmark,
    check_lobe_gradients,
    check_mlp_gradients,
    probe_struct_params,
)

# 候选写法按可能性排序；哪一个能编译就是「slangpy 携带的 slang」接受的写法，写入输出 JSON。
WEIGHT_CANDIDATES = (
    ("DiffTensor<float, 1>", "weights.get({ int(index) })"),
    ("DiffTensor<float, 1>", "weights.load(int(index))"),
    ("GradOutTensor<float, 1>", "weights.get({ int(index) })"),
    ("GradInOutTensor<float, 1>", "weights.get({ int(index) })"),
    ("GradOutTensor<float, 1>", "weights[{ int(index) }]"),
)


def version_info(spy: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"slangpy": getattr(spy, "__version__", None), "slangc": None}
    info["version_attributes"] = {
        name: str(getattr(spy, name)) for name in dir(spy) if "VERSION" in name.upper()
    }
    package = Path(spy.__file__).resolve().parent
    for executable in sorted(package.rglob("slangc*")):
        if executable.is_file() and executable.suffix in {"", ".exe"}:
            result = subprocess.run([str(executable), "-v"], capture_output=True, text=True)
            info["slangc"] = (result.stdout + result.stderr).strip()
            break
    return info


def load_module(spy: Any, device: Any, scratch: Path) -> tuple[Any, dict[str, Any]]:
    template = Path(__file__).with_suffix(".slang").read_text(encoding="utf-8")
    attempts: list[dict[str, Any]] = []
    for index, (tensor_type, read) in enumerate(WEIGHT_CANDIDATES):
        source = template.replace("@WEIGHT_TENSOR@", tensor_type).replace("@WEIGHT_READ@", read)
        # SlangPy caches modules by path. A unique name is required so a failed
        # first syntax candidate cannot poison all later attempts.
        path = scratch / f"spike_slangpy_autodiff_candidate_{index}.slang"
        path.write_text(source, encoding="utf-8")
        try:
            module = spy.Module.load_from_file(device, str(path))
            attempts.append({"weight_tensor": tensor_type, "weight_read": read, "ok": True})
            return module, {"weight_tensor": tensor_type, "weight_read": read, "attempts": attempts}
        except Exception as error:  # noqa: BLE001 - 编译失败本身就是 spike 要记录的结果
            attempts.append({"weight_tensor": tensor_type, "weight_read": read, "error": str(error)[:4000]})
    raise RuntimeError("no weight tensor candidate compiled: " + json.dumps(attempts, ensure_ascii=False))


def probe_torch_interop(spy: Any) -> dict[str, Any]:
    names = ("from_torch", "to_torch", "from_dlpack", "__dlpack__")
    return {
        "tensor_methods": {name: hasattr(spy.Tensor, name) for name in names},
        "torch_module": hasattr(spy, "TorchModule"),
        "cuda_device_type": hasattr(getattr(spy, "DeviceType", None), "cuda"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device-type", default="cuda", help="spy.DeviceType 名；失败则回退默认设备")
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--directions", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "spikes" / "slangpy-autodiff.json")
    args = parser.parse_args()
    import slangpy as spy

    report: dict[str, Any] = {"versions": version_info(spy), "torch": torch.__version__, "torch_interop": probe_torch_interop(spy)}
    try:
        device = spy.create_device(type=getattr(spy.DeviceType, args.device_type))
    except Exception as error:  # noqa: BLE001
        report["device_fallback"] = str(error)[:500]
        device = spy.create_device()
    report["device"] = str(getattr(device, "info", device))
    scratch = args.output.parent / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    module, report["module"] = load_module(spy, device, scratch)
    rng = np.random.default_rng(20260824)
    report["lobe_gradients"] = check_lobe_gradients(spy, module, device, rng, args.groups, args.directions)
    report["mlp_gradients"] = check_mlp_gradients(spy, module, device, rng, args.groups, args.directions, args.probes)
    report["struct_params"] = probe_struct_params(spy, module, device, rng)
    report["throughput"] = benchmark(spy, module, device, rng, args.groups, args.directions, args.iterations)
    report["pass"] = all(report[key]["pass"] for key in ("lobe_gradients", "mlp_gradients", "throughput"))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
