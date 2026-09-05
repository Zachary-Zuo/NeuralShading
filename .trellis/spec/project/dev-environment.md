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
| FW | 文件 `external/Falcor/build/windows-vs2022/bin/Release/python/falcor/falcor_ext.cp310-win_amd64.pyd` 存在 | Windows Falcor Python 已按锁定提交构建；由 `ncls.runtime` 读取锁定 manifest 定位 |
| FL | `external/Falcor/build/linux-gcc/bin/Release/python/falcor/falcor_ext*.so` 恰有匹配 | Linux Falcor Python 已按锁定提交构建；由 `ncls.runtime` 读取锁定 manifest 定位 |
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
| **完整 Windows** | G ∧ E ∧ FW ∧ W | `TESTING.md` 中的全部命令：`tests/unit`、`tests/gpu`、`tests/integration`、`ncls train / validate / export` 全链路、`scripts/build_viewer.ps1`、`scripts/benchmark_viewer.ps1` 与正式online训练 | — |
| **Linux reference** | G ∧ E ∧ FL ∧ L | `tests/unit`、`slangpy`、CUDA online训练/评测，以及经统一backend的headless LayerStack/MERL/OpenPBR/MaterialX/MDL reference query | Windows viewer、尚未迁移的pbrt Windows probe |
| **仅 GPU** | G ∧ E，缺少当前平台对应的 FW/FL | `tests/unit`、`slangpy` marker 的测试、无 Falcor 依赖的模型/GPU 测试与 SlangPy spike | 任何 `falcor` marker 的测试与 Falcor launcher、viewer、正式 online reference 训练 |
| **静态** | 其余 | 读代码、字节码编译检查、`git diff --check`、写代码与文档 | 运行任何项目代码；宣称"已验证 / 已通过 / 已复现" |

`environment.yml` 声明的 `neural-shading` 是唯一运行时真相；`base`、系统 Python、`.venv` 都不是。

## Reference backend 部署边界

- Ubuntu/Linux具体版本不预先冻结；`scripts/deploy_reference_linux.sh`记录并提示实际distro/glibc/compiler，是否支持由真实configure/build/device/probe决定。
- 部署脚本可获取根manifest锁定的`external/`源码与MDL SDK binary package，可创建/更新既有Conda中的`neural-shading`；不得安装Conda、driver或使用`sudo`。
- 部署永不下载、移动或写入`assets/`。用户复制source assets后，才运行五族真实snapshot与MDL training gate；资产缺失不影响compile deployment和仓库fixture probe成功。
- Windows公共入口为`scripts/build_reference_backend.ps1`；Linux公共入口为`bash scripts/deploy_reference_linux.sh`。upper tools只运行`ncls reference doctor/probe`或`backend.open(ReferenceExecutionPlan@1)`。

## Linux 多 GPU 与平台运行时

### 1. 适用范围

在 Linux 运行 online reference/训练或 GPU 工具时，保证 Torch、SlangPy 和 Falcor 选择同一物理卡。平台差异只由 Python runtime 与 backend 装配。

### 2. 签名

```bash
python -m ncls train 2,5 --config configs/training/runs/nvidia-layer-stack-smoke.yaml
python -m ncls reference probe --device 2
python -m ncls.runtime --device 2 -- -m pytest tests/gpu/test_reference_query_dispatcher.py
```

部署入口仍为 `CUDA_VISIBLE_DEVICES=2 bash scripts/deploy_reference_linux.sh`；该变量仅选部署 probe。日常 train 参数只指定一次 GPU，不需要叠加 torchrun、--gpus 或环境变量。

### 3. 合同

- runtime 从 toolchain manifest 选择各平台 Falcor 构建；在新 Python 进程导入原生库前设置 PATH/PYTHONPATH，Linux 另设 LD_LIBRARY_PATH。
- Linux 多卡由同一 launcher 启动一个 torchrun。worker 以 LOCAL_RANK 索引物理列表，设置 CUDA_VISIBLE_DEVICES 与 NCLS_FALCOR_GPU_INDEX；Torch/SlangPy 只看到自身 cuda:0，Falcor 使用物理序号。
- NCLS_RUN_DIR 在启动 DDP 前分配一次。rank 0 写 checkpoint/日志/TensorBoard；数值 validation 由各 rank 共同汇总。图像的共同接口当前在 Linux 返回 None，无渲染/GPU/文件副作用。
- NCCL 梯度 reducer 与 Gloo 控制组的次序保持一致。NCLS_DDP_TIMEOUT_SECONDS 默认 300，NCLS_DDP_CONTROL_TIMEOUT_SECONDS 默认 1800；NCLS_DDP_DEBUG=1 显式开启诊断。
- 同一进程按 Falcor module/API/physical GPU 复用 device；session 关闭只释放自身资源，CLI 最后显式 close_reference_backend_devices，再关闭分布式组。
- backend 拒绝软件 adapter，不能将其结果登记为 GPU 证据。
- Linux 当前 Conda 环境安装 cuda-compat=12.8.1；driver 主版本低于 570 时优先使用该环境的 cuda-compat，否则使用系统 driver。当前 Conda 的 lib 始终进入动态库搜索路径。
- manifest 同时保留 Windows/Linux 相对构建布局。external/build/assets/outputs/artifacts 是每台主机本地 ignored 状态，本机移动或构建不能作为另一台主机已完成的证据。

### 4. 错误矩阵

| 条件 | 行为 |
|---|---|
| GPU 参数重复、负数或非法 | 入口报告参数错误 |
| Windows 多卡 | 在创建原生 runtime 前报告当前只支持 Linux/NCCL |
| rank/world 与物理列表不同 | 装配失败，不进入错误的 interop/collective |
| Falcor 构建缺失 | 指明缺失模块位置 |
| Falcor 选择软件 adapter | backend 报错 |
| 同卡数下更换物理编号或预取设置 | 可以恢复实际训练状态，不使用完整执行 plan hash 门禁 |
| Linux 图像 hook | 返回 None，数值 validation 保留 |

### 5. 案例

正常：train 2,5 启动一个两卡作业，rank 1 的 Falcor 选物理卡 5，Torch 用 cuda:0。
基础：train 0 在 Windows/Linux 都进入公共 engine。
错误：只限制 Torch 可见域，却让 Falcor 默认选另一张卡；或让各 rank 独立生成 run 目录。

### 6. 验证

unit 覆盖 bootstrap、物理卡映射、driver 分支、设备复用/关闭、run 分配与 Linux 空图像实现。Linux 实机验证 NCCL、Vulkan interop、rank state/数值汇总和有序退出；具体命令与待验证边界见 TESTING.md。

### 7. 错误与正确

```text
错误：用户重复填写 CUDA_VISIBLE_DEVICES、shell --gpus、train --devices。
正确：python -m ncls train 2,5 --config run.yaml。
错误：Windows 测试通过便声称 Linux/NCCL 已验证。
正确：分别记录本机证据与目标 Linux 机待执行项。
```

## 静态状态下的交接

- 本应运行的命令与期望结果写进根目录 `TESTING.md` 对应小节（Setup / 命令 / 期望 / 静态分析覆盖不到的边界 / 已知 `type: ignore`）。只写本次改动相关的部分，不重复全项目测试计划。
- 报告里区分「已静态检查」与「待远程验证」；不用"应该能跑"代替后者。
- 远程结果回来后，由本地会话把结论回写 `docs/research/experiment_log.md` 或对应文档，并删掉 `TESTING.md` 中已过时的段落。

## 报告格式

会话内第一次涉及验证时写一行：

`环境：<完整 Windows | Linux reference | 仅 GPU | 静态>；GPU=<卡名或无>；neural-shading=<有 | 无>；Falcor 构建=<Windows | Linux | 无>；本会话可做：<...>`
