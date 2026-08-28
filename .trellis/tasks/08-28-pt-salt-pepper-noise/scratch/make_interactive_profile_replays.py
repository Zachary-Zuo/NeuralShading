"""生成同一 MDL 场景的 headless capture 调度诊断 replay。"""

from __future__ import annotations

import json
from pathlib import Path


TASK_SCRATCH = Path(__file__).resolve().parent
SOURCE = TASK_SCRATCH / "mdl-carpaint-960x540.json"


def main() -> None:
    replay = json.loads(SOURCE.read_text(encoding="utf-8"))
    for samples_per_frame in (1, 4):
        variant = dict(replay)
        variant["reference_samples_per_frame"] = samples_per_frame
        output = TASK_SCRATCH / f"mdl-carpaint-960x540-spf{samples_per_frame}.json"
        output.write_text(
            json.dumps(variant, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(output)

    remainder_variant = dict(replay)
    remainder_variant["reference_spp"] = 18
    remainder_variant["reference_samples_per_frame"] = 16
    remainder_output = TASK_SCRATCH / "mdl-carpaint-960x540-target18-spf16.json"
    remainder_output.write_text(
        json.dumps(remainder_variant, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(remainder_output)


if __name__ == "__main__":
    main()
