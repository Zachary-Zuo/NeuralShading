---
name: project-dev-environment
description: 开发机四种状态（完整 Windows / Linux reference / 仅 GPU / 静态）的判定探针、验证边界与静态交接
paths:
  - tests/**
  - scripts/**
  - TESTING.md
  - environment.yml
  - requirements-torch-cu128.txt
---

# 开发机状态判定

> 验证深度由**当前机器**决定，不由旧笔记、习惯或另一台机器决定。每个会话在做任何"验证 / 测试 / 构建 / 训练"之前先判定一次，并在第一次涉及验证的回复里写明状态与证据。

## 六个探针

| # | 探针 | 满足条件 |
|---|---|---|
| G | `nvidia-smi --query-gpu=name --format=csv,noheader` | 输出含高性能卡关键词：`RTX 4090`、`RTX A6000`、`A100`、`H100`、`L40S`。笔记本卡（如 `RTX 3060 Laptop`）不算 |
| E | `conda env list` | 存在名为 `neural-shading` 的环境 |
| FW | 文件 `external/Falcor/build/windows-vs2022/bin/Release/python/falcor/falcor_ext.cp310-win_amd64.pyd` 存在 | Windows Falcor Python 已按锁定提交构建；与 `scripts/run_falcor_python.ps1` 的检查相同 |
| FL | `external/Falcor/build/linux-gcc/bin/Release/python/falcor/falcor_ext*.so` 恰有匹配 | Linux Falcor Python 已按锁定提交构建；与 `scripts/run_falcor_python.sh` 的检查相同 |
| W | 原生 Windows（非 WSL） | viewer / D3D12 / pbrt probe 只能在这里构建 |
| L | 原生 Linux | headless Falcor/Vulkan reference 采集可在这里运行；WSL 不算原生 Linux 部署证据 |

```powershell
nvidia-smi --query-gpu=name --format=csv,noheader
conda env list
Test-Path external\Falcor\build\windows-vs2022\bin\Release\python\falcor\falcor_ext.cp310-win_amd64.pyd
```

```bash
nvidia-smi --query-gpu=name --format=csv,noheader; conda env list
compgen -G 'external/Falcor/build/linux-gcc/bin/Release/python/falcor/falcor_ext*.so' >/dev/null && echo falcor-linux-ok
```

探针被 hook 拦截、输出像 mock、或无法确认解释器就是 `neural-shading` 时，一律按"静态"处理。

## 四种状态

| 状态 | 条件 | 允许 | 禁止 |
|---|---|---|---|
| **完整 Windows** | G ∧ E ∧ FW ∧ W | `TESTING.md` 中的全部命令：`tests/unit`、`tests/gpu`、`tests/integration`、`ncls data / learn / bundle` 全链路、`scripts/build_viewer.ps1`、`scripts/benchmark_viewer.ps1`、正式训练与采集 | — |
| **Linux reference** | G ∧ E ∧ FL ∧ L | `tests/unit`、`slangpy` 测试、CUDA 训练/评测，以及经 `scripts/run_falcor_python.sh` 的 headless LayerStack/MERL/OpenPBR/MaterialX reference 采集 | Windows viewer、D3D12-only GPU 测试、尚未迁移的 pbrt Windows probe；不能把 HDF5 collector 称为 GPU-resident online training |
| **仅 GPU** | G ∧ E，缺少当前平台对应的 FW/FL | `tests/unit`、`slangpy` marker 的测试、`ncls learn train / evaluate / compare / benchmark`、SlangPy spike | 任何 `falcor` marker 的测试与 Falcor launcher、viewer、正式 LayerStack reference 采集 |
| **静态** | 其余（当前本机：WSL2 + `RTX 3060 Laptop`，无 conda、无 `external/`） | 读代码、字节码编译检查、`git diff --check`、写代码与文档 | 运行任何项目代码；宣称"已验证 / 已通过 / 已复现" |

`environment.yml` 声明的 `neural-shading` 是唯一运行时真相；`base`、系统 Python、`.venv` 都不是。

## 静态状态下的交接

- 本应运行的命令与期望结果写进根目录 `TESTING.md` 对应小节（Setup / 命令 / 期望 / 静态分析覆盖不到的边界 / 已知 `type: ignore`）。只写本次改动相关的部分，不重复全项目测试计划。
- 报告里区分「已静态检查」与「待远程验证」；不用"应该能跑"代替后者。
- 远程结果回来后，由本地会话把结论回写 `docs/research/experiment_log.md` 或对应文档，并删掉 `TESTING.md` 中已过时的段落。

## 报告格式

会话内第一次涉及验证时写一行：

`环境：<完整 Windows | Linux reference | 仅 GPU | 静态>；GPU=<卡名或无>；neural-shading=<有 | 无>；Falcor 构建=<Windows | Linux | 无>；本会话可做：<...>`
