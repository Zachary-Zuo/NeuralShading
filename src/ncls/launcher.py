"""唯一用户入口：先配置环境，再启动单卡或 Linux DDP。"""

from __future__ import annotations

import subprocess
import sys

from .commands import build_parser
from .runs import RunPaths
from .runtime import process_environment
from .visual_eval import visual_evaluation_available


def worker_command(arguments: list[str], devices: tuple[int, ...]) -> list[str]:
    if len(devices) == 1:
        return [sys.executable, "-m", "ncls.cli", *arguments]
    return [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={len(devices)}", "-m", "ncls.ddp_worker",
        "-m", "ncls.cli", *arguments,
    ]


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(arguments)
    if args.command == "train":
        config = args.config.resolve(strict=True)
        environment = process_environment(args.devices)
        paths = RunPaths.create(config) if args.resume is None else RunPaths.from_checkpoint(args.resume)
        paths.begin(config, args.devices, args.resume)
        environment["NCLS_RUN_DIR"] = str(paths.root)
        print(f"训练输出：{paths.root}", flush=True)
        try:
            code = subprocess.call(worker_command(arguments, args.devices), env=environment)
        except BaseException:
            paths.record(status="interrupted")
            raise
        paths.record(status="finished" if code == 0 else "failed", exit_code=code)
        return code
    needs_runtime = args.command in {"validate", "export"} or (
        args.command == "eval" and visual_evaluation_available()
    ) or (
        args.command == "reference" and args.reference_command == "probe"
    )
    if needs_runtime:
        environment = process_environment((getattr(args, "device", 0),))
        return subprocess.call([sys.executable, "-m", "ncls.cli", *arguments], env=environment)
    from .cli import main as dispatch

    return dispatch(arguments)
