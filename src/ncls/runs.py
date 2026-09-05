"""一次训练的目录及运行记录；不参与模型身份判断。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from .paths import OUTPUT_ROOT, PROJECT_ROOT


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def create(cls, config: Path, *, output_root: Path = OUTPUT_ROOT) -> "RunPaths":
        run_id = datetime.now().strftime("%y%m%d-%H%M%S-") + uuid4().hex[:6]
        result = cls((output_root / config.stem / run_id).resolve())
        result.root.mkdir(parents=True)
        shutil.copyfile(config, result.root / "config.yaml")
        return result

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> "RunPaths":
        checkpoint = checkpoint.resolve(strict=True)
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("续训 checkpoint 应位于 run/checkpoints/ 中")
        return cls(checkpoint.parent.parent)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def checkpoint(self) -> Path:
        return self.checkpoints / "latest.pt"

    def step_checkpoint(self, step: int) -> Path:
        return self.checkpoints / f"step-{step:08d}.pt"

    @property
    def tensorboard(self) -> Path:
        return self.root / "tensorboard"

    @property
    def evaluation(self) -> Path:
        return self.root / "eval"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def metrics(self) -> Path:
        return self.logs / "metrics.jsonl"

    def record(self, **values: object) -> None:
        target = self.root / "run.json"
        record = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        record.update(values)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    def begin(self, config: Path, devices: tuple[int, ...], resume: Path | None) -> None:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=False,
        )
        self.record(
            config=str(config.resolve()), devices=list(devices),
            revision=revision.stdout.strip() if revision.returncode == 0 else None,
            resume=None if resume is None else str(resume.resolve()), status="running",
        )
