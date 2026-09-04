from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
import threading
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from ..events import TrainingEvent


@dataclass(frozen=True)
class _ScalarCommand:
    namespace: str
    step: int
    scalars: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _ImageCommand:
    tag: str
    step: int
    path: Path


@dataclass(frozen=True)
class _FlushCommand:
    done: threading.Event


@dataclass(frozen=True)
class _CloseCommand:
    done: threading.Event


class TensorBoardHook:
    """Rank-0 bounded asynchronous TensorBoard scalar sink."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        rank: int,
        flush_seconds: int = 10,
        queue_capacity: int = 4096,
        writer_factory: Callable[..., Any] = SummaryWriter,
    ) -> None:
        if rank != 0:
            raise ValueError("TensorBoardHook may only be constructed on rank 0")
        if flush_seconds < 1 or queue_capacity < 1:
            raise ValueError("TensorBoard flush interval and queue capacity must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._flush_seconds = int(flush_seconds)
        self._writer_factory = writer_factory
        self._queue: Queue[
            _ScalarCommand | _ImageCommand | _FlushCommand | _CloseCommand
        ] = Queue(
            maxsize=int(queue_capacity)
        )
        self._last_step: dict[str, int] = {}
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker_main,
            name="ncls-tensorboard-writer",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _namespace(kind: str) -> str | None:
        return {
            "step-completed": "train",
            "validation-completed": "validation",
        }.get(kind)

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("TensorBoard writer failed") from error

    def _worker_main(self) -> None:
        writer: Any | None = None
        failed = False
        try:
            writer = self._writer_factory(
                log_dir=str(self.output_dir), flush_secs=self._flush_seconds
            )
            while True:
                command = self._queue.get()
                try:
                    if isinstance(command, _CloseCommand):
                        if not failed:
                            writer.flush()
                            writer.close()
                        command.done.set()
                        return
                    if isinstance(command, _FlushCommand):
                        if not failed:
                            writer.flush()
                        command.done.set()
                        continue
                    if isinstance(command, _ImageCommand) and not failed:
                        with Image.open(command.path) as source:
                            image = np.asarray(source.convert("RGB"))
                        writer.add_image(command.tag, image, command.step, dataformats="HWC")
                        continue
                    if not failed:
                        for name, value in command.scalars:
                            tag_name = (
                                name[len(command.namespace) + 1 :]
                                if name.startswith(command.namespace + "/")
                                else name
                            )
                            writer.add_scalar(
                                f"{command.namespace}/{tag_name}", value, command.step
                            )
                except BaseException as error:
                    failed = True
                    self._set_error(error)
                    if isinstance(command, (_FlushCommand, _CloseCommand)):
                        command.done.set()
                    if isinstance(command, _CloseCommand):
                        return
                finally:
                    self._queue.task_done()
        except BaseException as error:
            self._set_error(error)
            while True:
                command = self._queue.get()
                self._queue.task_done()
                if isinstance(command, (_FlushCommand, _CloseCommand)):
                    command.done.set()
                if isinstance(command, _CloseCommand):
                    return

    def _put(
        self, command: _ScalarCommand | _ImageCommand | _FlushCommand | _CloseCommand
    ) -> None:
        self._raise_if_failed()
        try:
            self._queue.put_nowait(command)
        except Full as error:
            raise RuntimeError("TensorBoard queue capacity is exhausted") from error

    def handle(self, event: TrainingEvent) -> None:
        if self._closed:
            raise RuntimeError("TensorBoardHook is closed")
        if event.rank != 0:
            return
        if event.kind == "visual-eval-completed":
            try:
                display = event.artifacts["display"]
            except KeyError as error:
                raise ValueError(
                    "visual-eval-completed event requires a display artifact"
                ) from error
            self._put(
                _ImageCommand("visual-eval/comparison", event.global_step, Path(display))
            )
            return
        namespace = self._namespace(event.kind)
        if namespace is None or not event.scalars:
            self._raise_if_failed()
            return
        previous = self._last_step.get(namespace, -1)
        if event.global_step < previous:
            raise ValueError(
                f"TensorBoard {namespace} step regressed from {previous} to {event.global_step}"
            )
        self._last_step[namespace] = event.global_step
        self._put(
            _ScalarCommand(
                namespace,
                event.global_step,
                tuple(sorted(event.scalars.items())),
            )
        )

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("TensorBoardHook is closed")
        done = threading.Event()
        self._put(_FlushCommand(done))
        done.wait()
        self._raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        done = threading.Event()
        error: BaseException | None = None
        try:
            try:
                self._queue.put(_CloseCommand(done), timeout=5.0)
            except Full as caught:
                raise RuntimeError("TensorBoard queue did not accept close") from caught
            if not done.wait(timeout=5.0):
                raise RuntimeError("TensorBoard writer did not close within five seconds")
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("TensorBoard writer thread remained alive after close")
            self._raise_if_failed()
        except BaseException as caught:
            error = caught
        finally:
            self._closed = True
        if error is not None:
            raise error


__all__ = ["TensorBoardHook"]
