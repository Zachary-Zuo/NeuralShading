# 验证记录

环境：完整 Windows；GPU=NVIDIA GeForce RTX 4090；`neural-shading`=有；Falcor Windows Python/Release viewer=有。

## 自动化质量门

| 验证 | 结果 |
|---|---|
| viewer/unit 相关矩阵 | 9 passed，68 deselected |
| surface/latent/package Falcor GPU 矩阵 | 3 passed |
| 全量 pytest（经 `run_falcor_python.ps1`） | 96 passed |
| 最终 surface-field + viewer slot 复测 | 5 passed |
| Release viewer build | 通过；reference/package/shared surface shader真实编译 |
| `git diff --check` | 通过 |
| `external/Falcor` worktree | 干净 |

项目环境没有声明或安装 `ruff`，因此没有额外 Python lint 命令可运行；未临时安装非项目依赖。新增 Python 测试已进入全量 pytest，新增 Slang 同时经过 Falcor GPU test 与 Release viewer build。

## 视觉与 report-only 指标

- walnut source-reference / 200k neural PT：`artifacts/nvidia-faithful/materialx-recorded-200k/viewer-reference-neural/walnut-direct-fixed-64spp.json`。
- denim source-reference PT：`artifacts/nvidia-faithful/materialx-recorded-200k/viewer-reference-neural/denim-fixed-32spp.json`。
- neural PT/deferred：`artifacts/nvidia-faithful/materialx-recorded-200k/viewer-reference-neural/walnut-neural-pt-deferred-fixed-64spp.json`。
- spatial metric 分析脚本：`scratch/analyze_spatial_captures.py`；观察值只用于说明修复效果，不决定任务成败。

最终交互 viewer 使用 walnut source-reference PT / 200k neural PT、1280×720、local-light replay 启动；未传 `--headless` 或 `--frames`，交互按时间持续累计。
