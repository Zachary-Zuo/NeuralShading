from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 原始、不可由本项目重新计算得到的输入资产。
ASSET_ROOT = PROJECT_ROOT / "assets"
SOURCE_MATERIAL_ROOT = ASSET_ROOT / "source-materials"
VIEWER_ASSET_ROOT = ASSET_ROOT / "viewer"

# 可再生成的报告、捕获、缓存和旧实验输出。
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
