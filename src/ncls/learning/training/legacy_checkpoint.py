from __future__ import annotations

from pathlib import Path

import torch

from ncls.core.identity import sha256_file

from .checkpoint import load_checkpoint
from .evaluation_snapshot import (
    EvaluationSnapshot,
    _public_key_for_checkpoint,
    _snapshot_from_runner,
)


class LegacyCheckpointV4Importer:
    """Strict read-only v4 importer for validation, export and visual diagnostics."""

    def load(
        self,
        path: Path | str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> EvaluationSnapshot:
        target = Path(path)
        checkpoint = load_checkpoint(target, map_location=map_location)
        public_key = _public_key_for_checkpoint(checkpoint)
        return _snapshot_from_runner(
            checkpoint,
            public_method_key=public_key,
            checkpoint_sha256=sha256_file(target),
            legacy_v4=True,
        )


__all__ = ["LegacyCheckpointV4Importer"]
