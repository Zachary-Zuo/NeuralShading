from __future__ import annotations

from pathlib import Path

from ncls.cli import _batch_source
from ncls.learning.methods import get_method
from ncls.learning.training import TrainingConfig, TrainingRunner, save_checkpoint


config = TrainingConfig.load(
    Path("configs/learning/nvidia-rta2024-materialx-smoke.json")
)
definition = get_method(config.method_key)
source = _batch_source(config)
try:
    result = TrainingRunner(definition, source, config).run(stop_at_step=1)
    output = Path(
        "artifacts/nvidia-faithful/materialx-resume-smoke/partial.step00000001.pt"
    )
    digest = save_checkpoint(output, result.checkpoint)
    print(f"partial_checkpoint_sha256={digest}")
finally:
    source.close()
