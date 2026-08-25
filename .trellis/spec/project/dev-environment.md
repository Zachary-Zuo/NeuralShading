---
name: project-dev-environment
description: 开发机三种状态（完整 / 仅 GPU / 静态）的判定探针，每种状态允许做的验证、禁止的宣称，以及静态状态下向 TESTING.md 的交接
paths:
  - tests/**
  - scripts/**
  - TESTING.md
  - environment.yml
  - requirements-torch-cu128.txt
---

# 开发机状态判定

> 验证深度由**当前机器**决定，不由旧笔记、习惯或另一台机器决定。每个会话在做任何"验证 / 测试 / 构建 / 训练"之前先判定一次，并在第一次涉及验证的回复里写明状态与证据。

## 四个探针

| # | 探针 | 满足条件 |
|---|---|---|
| G | `nvidia-smi --query-gpu=name --format=csv,noheader` | 输出含高性能卡关键词：`RTX 4090`、`A100`、`H100`、`L40S`。笔记本卡（如 `RTX 3060 Laptop`）不算 |
| E | `conda env list` | 存在名为 `neural-shading` 的环境 |
| F | 文件 `external/Falcor/build/windows-vs2022/bin/Release/python/falcor/falcor_ext.cp310-win_amd64.pyd` 存在 | Falcor Python 已按锁定提交构建；与 `scripts/run_falcor_python.ps1` 的检查相同 |
| W | 原生 Windows（非 WSL） | viewer / D3D12 / pbrt 与 OpenPBR probe 只能在这里构建 |

```powershell
nvidia-smi --query-gpu=name --format=csv,noheader
conda env list
Test-Path external\Falcor\build\windows-vs2022\bin\Release\python\falcor\falcor_ext.cp310-win_amd64.pyd
```

```bash
nvidia-smi --query-gpu=name --format=csv,noheader; conda env list
test -f external/Falcor/build/windows-vs2022/bin/Release/python/falcor/falcor_ext.cp310-win_amd64.pyd && echo falcor-ok
```

探针被 hook 拦截、输出像 mock、或无法确认解释器就是 `neural-shading` 时，一律按"静态"处理。

## 三种状态

| 状态 | 条件 | 允许 | 禁止 |
|---|---|---|---|
| **完整** | G ∧ E ∧ F ∧ W | `TESTING.md` 中的全部命令：`tests/unit`、`tests/gpu`、`tests/integration`、`ncls data / learn / bundle` 全链路、`scripts/build_viewer.ps1`、`scripts/benchmark_viewer.ps1`、正式训练与采集 | — |
| **仅 GPU** | G ∧ E，缺 F 或 W | `tests/unit`、`slangpy` marker 的测试、`ncls learn train / evaluate / compare / benchmark`、SlangPy spike | 任何 `falcor` marker 的测试与 `run_falcor_python.ps1` 路径、viewer、正式采集（LayerStack reference 依赖 Falcor） |
| **静态** | 其余（当前本机：WSL2 + `RTX 3060 Laptop`，无 conda、无 `external/`） | 读代码、字节码编译检查、`git diff --check`、写代码与文档 | 运行任何项目代码；宣称"已验证 / 已通过 / 已复现" |

`environment.yml` 声明的 `neural-shading` 是唯一运行时真相；`base`、系统 Python、`.venv` 都不是。

## 静态状态下的交接

- 本应运行的命令与期望结果写进根目录 `TESTING.md` 对应小节（Setup / 命令 / 期望 / 静态分析覆盖不到的边界 / 已知 `type: ignore`）。只写本次改动相关的部分，不重复全项目测试计划。
- 报告里区分「已静态检查」与「待远程验证」；不用"应该能跑"代替后者。
- 远程结果回来后，由本地会话把结论回写 `docs/research/experiment_log.md` 或对应文档，并删掉 `TESTING.md` 中已过时的段落。

## 报告格式

会话内第一次涉及验证时写一行：

`环境：<完整 | 仅 GPU | 静态>；GPU=<卡名或无>；neural-shading=<有 | 无>；Falcor 构建=<有 | 无>；本会话可做：<...>`
