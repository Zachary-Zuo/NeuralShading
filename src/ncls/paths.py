from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 原始、不可由本项目重新计算得到的输入资产。
ASSET_ROOT = PROJECT_ROOT / "assets"
SOURCE_MATERIAL_ROOT = ASSET_ROOT / "source-materials"
VIEWER_ASSET_ROOT = ASSET_ROOT / "viewer"

# 任务临时产物；已有视觉证据仍按原路径保留。
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

# 新实验的权重、日志、图像与部署导出。
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
