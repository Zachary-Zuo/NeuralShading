"""在导入 Torch/Falcor 前准备当前平台环境，也用于 GPU 测试与工具。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping

from .paths import PROJECT_ROOT


def parse_devices(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(not part.isdecimal() for part in parts):
        raise argparse.ArgumentTypeError("GPU 应为物理编号列表，例如 0 或 0,1")
    devices = tuple(map(int, parts))
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("GPU 编号不能重复")
    return devices


def configure_distributed_debug_environment(
    environment: dict[str, str],
) -> None:
    """按显式 opt-in 启用 PyTorch 2.11 已存在的 NCCL 诊断开关。"""

    if environment.get("NCLS_DDP_DEBUG") != "1":
        return
    environment.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    environment.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "20000")
    environment.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    environment.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
    environment.setdefault("TORCH_NCCL_ENABLE_TIMING", "1")



def process_environment(
    devices: tuple[int, ...], *, project_root: Path = PROJECT_ROOT,
    system: str | None = None, environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    system = platform.system() if system is None else system
    if system not in {"Windows", "Linux"}:
        raise RuntimeError(f"尚不支持 {system}")
    if len(devices) > 1 and system != "Linux":
        raise RuntimeError("多 GPU 训练目前使用 Linux/NCCL")
    manifest = json.loads((project_root / "references/reference-backend-toolchains.json").read_text(encoding="utf-8"))
    platform_id = f"{system.lower()}-x86_64@1"
    entry = next(item for item in manifest["platforms"] if item["platform_id"] == platform_id)
    falcor = entry["falcor"]
    binary = project_root / falcor["runtime_library_root"]
    module = project_root / falcor["python_module_root"]
    if not list(module.glob(falcor["python_extension"])):
        raise FileNotFoundError(f"缺少 Falcor Python 构建：{module / falcor['python_extension']}")
    result = dict(os.environ if environment is None else environment)
    separator = ";" if system == "Windows" else ":"

    def prepend(name: str, paths: list[Path]) -> None:
        result[name] = separator.join([*(str(path) for path in paths), *filter(None, [result.get(name, "")])])

    prepend("PATH", [binary])
    prepend("PYTHONPATH", [project_root / "src", module])
    if system == "Linux":
        libraries = [binary]
        compat = Path(sys.prefix) / "cuda-compat"
        if compat.is_dir():
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True,
            ).splitlines()[0]
            if int(driver.split(".")[0]) < 570:
                libraries.append(compat)
        libraries.append(Path(sys.prefix) / "lib")
        prepend("LD_LIBRARY_PATH", libraries)
    result["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    if len(devices) == 1:
        result.pop("NCLS_DDP_GPU_LIST", None)
        result["NCLS_FALCOR_GPU_INDEX"] = str(devices[0])
    else:
        result.pop("NCLS_FALCOR_GPU_INDEX", None)
        result["NCLS_DDP_GPU_LIST"] = result["CUDA_VISIBLE_DEVICES"]
    configure_distributed_debug_environment(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为 Python GPU 工具准备统一的 Falcor/CUDA 环境")
    parser.add_argument("--device", type=int, default=0)
    args, arguments = parser.parse_known_args(argv)
    if args.device < 0:
        parser.error("GPU 编号必须非负")
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        parser.error("需要 Python 参数，例如 -- -m pytest tests/gpu")
    return subprocess.call([sys.executable, *arguments], env=process_environment((args.device,)))


if __name__ == "__main__":
    raise SystemExit(main())
