"""Per-rank environment shim for Linux torchrun/Falcor interop."""

from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        raise SystemExit("ncls.ddp_worker requires: -m <module> [args ...]")
    module = sys.argv[2]
    gpu_list = os.environ.get("NCLS_DDP_GPU_LIST", "")
    local_raw = os.environ.get("LOCAL_RANK")
    if not gpu_list or local_raw is None:
        raise SystemExit("DDP worker requires NCLS_DDP_GPU_LIST and LOCAL_RANK")
    try:
        gpu_indices = [int(value) for value in gpu_list.split(",")]
        local_rank = int(local_raw)
    except ValueError as error:
        raise SystemExit("DDP worker GPU and rank values must be integers") from error
    if (
        not gpu_indices
        or len(set(gpu_indices)) != len(gpu_indices)
        or local_rank < 0
        or local_rank >= len(gpu_indices)
        or any(value < 0 for value in gpu_indices)
    ):
        raise SystemExit("DDP worker GPU list or LOCAL_RANK is invalid")

    # Keep the full list in NCLS_DDP_GPU_LIST for identity checks, while the
    # CUDA runtime sees only this worker's physical adapter as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_indices[local_rank])
    os.environ["NCLS_FALCOR_GPU_INDEX"] = str(gpu_indices[local_rank])
    os.environ["NCLS_DDP_DEVICE_INDEX"] = "0"
    os.environ["NCLS_DDP_WORKER"] = "1"
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
