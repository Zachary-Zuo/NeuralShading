from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from ..events import TrainingEvent


class TensorBoardHook:
    """使用 SummaryWriter 自带的有界异步写入，不再套第二层队列。"""

    def __init__(
        self, output_dir: Path | str, *, rank: int = 0,
        flush_seconds: int = 10, queue_capacity: int = 4096,
        resume_step: int | None = None,
        writer_factory: Callable[..., Any] = SummaryWriter,
    ) -> None:
        self.rank = rank
        self.writer = writer_factory(
            log_dir=str(output_dir), flush_secs=flush_seconds, max_queue=queue_capacity,
            purge_step=None if resume_step is None else resume_step + 1,
        )

    def handle(self, event: TrainingEvent) -> None:
        if event.rank != 0:
            return
        if event.kind == "visual-eval-completed":
            for tag, path in event.artifacts.items():
                with Image.open(path) as source:
                    pixels = np.asarray(source.convert("RGB"))
                self.writer.add_image(f"visual-eval/{tag}", pixels, event.global_step, dataformats="HWC")
            for name, value in event.scalars.items():
                self.writer.add_scalar(f"visual-eval/{name}", value, event.global_step)
            return
        namespace = {"step-completed": "train", "validation-completed": "validation"}.get(event.kind)
        if namespace is not None:
            for name, value in event.scalars.items():
                self.writer.add_scalar(f"{namespace}/{name.removeprefix(namespace + '/')}", value, event.global_step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
