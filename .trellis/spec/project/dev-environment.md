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
| **完整 Windows** | G ∧ E ∧ FW ∧ W | `TESTING.md` 中的全部命令：`tests/unit`、`tests/gpu`、`tests/integration`、`ncls learn / bundle` 全链路、`scripts/build_viewer.ps1`、`scripts/benchmark_viewer.ps1` 与正式online训练 | — |
| **Linux reference** | G ∧ E ∧ FL ∧ L | `tests/unit`、`slangpy`、CUDA online训练/评测，以及经统一backend的headless LayerStack/MERL/OpenPBR/MaterialX/MDL reference query | Windows viewer、尚未迁移的pbrt Windows probe |
| **仅 GPU** | G ∧ E，缺少当前平台对应的 FW/FL | `tests/unit`、`slangpy` marker 的测试、`ncls learn train / evaluate / compare / benchmark`、SlangPy spike | 任何 `falcor` marker 的测试与 Falcor launcher、viewer、正式 LayerStack reference 采集 |
| **静态** | 其余（当前本机：WSL2 + `RTX 3060 Laptop`，无 conda、无 `external/`） | 读代码、字节码编译检查、`git diff --check`、写代码与文档 | 运行任何项目代码；宣称"已验证 / 已通过 / 已复现" |

`environment.yml` 声明的 `neural-shading` 是唯一运行时真相；`base`、系统 Python、`.venv` 都不是。

## Reference backend 部署边界

- Ubuntu/Linux具体版本不预先冻结；`scripts/deploy_reference_linux.sh`记录并提示实际distro/glibc/compiler，是否支持由真实configure/build/device/probe决定。
- 部署脚本可获取根manifest锁定的`external/`源码与MDL SDK binary package，可创建/更新既有Conda中的`neural-shading`；不得安装Conda、driver或使用`sudo`。
- 部署永不下载、移动或写入`assets/`。用户复制source assets后，才运行五族真实snapshot与MDL training gate；资产缺失不影响compile deployment和仓库fixture probe成功。
- Windows公共入口为`scripts/build_reference_backend.ps1`；Linux公共入口为`bash scripts/deploy_reference_linux.sh`。upper tools只运行`ncls reference doctor/probe`或`backend.open(ReferenceExecutionPlan@1)`。

## Linux多GPU、Falcor device与CUDA 12.8 compatibility合同

### 1. Scope / Trigger

在原生Linux多GPU机器运行deployment、Falcor reference query或SlangPy online训练时适用。它防止Falcor Vulkan与Torch/SlangPy落到不同物理GPU，也防止Falcor反复创建设备后静默退回软件Vulkan，或旧宿主driver拒绝CUDA 12.8运行时PTX。

### 2. Signatures

```bash
CUDA_VISIBLE_DEVICES=<单个物理GPU序号> bash scripts/deploy_reference_linux.sh
CUDA_VISIBLE_DEVICES=<单个物理GPU序号> bash scripts/run_falcor_python.sh <python-args>
bash scripts/run_falcor_python.sh --gpus <gpu0,gpu1,...> -- <python-args>
```

公共backend内部读取`NCLS_FALCOR_GPU_INDEX`并调用`falcor.Device(type=<API>, gpu=<物理序号>)`；该内部变量只能由Linux launcher从`CUDA_VISIBLE_DEVICES`派生，用户不单独设置。

### 3. Contracts

- `CUDA_VISIBLE_DEVICES`若存在，必须是一个十进制非负整数，表示物理GPU序号；不接受逗号列表、UUID或空列表。
- launcher把该值写入`NCLS_FALCOR_GPU_INDEX`。Falcor使用物理序号，Torch/SlangPy只看到一张卡并使用重映射后的`cuda:0`。
- `--gpus`只接受不重复的十进制物理序号列表；入口启动一个torchrun/NCCL DDP作业，各rank使用重映射`cuda:<LOCAL_RANK>`并将梯度all-reduce，统一checkpoint/metrics仅由rank0写出。
- 同一Python进程按`(Falcor module, device API, physical GPU index)`复用一个Falcor device；session关闭只释放session资源，不销毁进程级device。
- backend拒绝`llvmpipe`、`lavapipe`、WARP和Microsoft Basic Render等软件adapter，不把其结果登记为GPU证据。
- Linux deployment固定安装`defaults::cuda-compat=12.8.1`。宿主driver主版本低于570时，launcher在`LD_LIBRARY_PATH`中优先放`${CONDA_PREFIX}/cuda-compat`；570及以上使用系统driver库。
- deployment report必须记录`cuda_visible_devices`和`falcor_gpu_index`；两者在指定GPU运行时相等。
- Git同步只传播受管的manifest、脚本、源码和文档。被根仓库忽略的`external/`、`build/`、`assets/`与`artifacts/`是每台主机各自维护的本地状态；Ubuntu上的移动、权限修复或构建不会改变Windows副本。
- 受管manifest必须同时保留Windows的`external/Falcor/build/windows-vs2022`和Linux的`external/Falcor/build/linux-gcc`相对布局。不得因为某台Ubuntu临时持有从Windows复制来的ignored目录，就改写、移动或删除Windows布局声明。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0,1`、UUID或负数 | launcher在启动Python前失败 |
| `--gpus`列表含重复项、非法值或缺少Python参数 | DDP launcher在启动torchrun前失败 |
| DDP rank环境、GPU列表与`WORLD_SIZE`不一致 | launcher/CLI fail closed |
| 只设置`NCLS_FALCOR_GPU_INDEX` | launcher拒绝，要求改用`CUDA_VISIBLE_DEVICES` |
| 两个变量值不一致 | launcher拒绝 |
| Falcor选择软件adapter | backend抛出`RuntimeError`，不继续query |
| driver 550加载CUDA 12.8 PTX | 使用Conda compatibility库；SlangPy CUDA device必须实际创建成功 |
| driver 570及以上 | 不覆盖系统`libcuda` |
| Ubuntu移动或修复ignored目录 | 只影响当前Ubuntu；不得宣称Windows路径已迁移或已修复 |
| 受管配置把Windows build root改成Linux路径或写入主机绝对路径 | 静态检查失败；恢复两个平台各自的仓库相对路径 |

### 5. Good / Base / Bad Cases

- Good：`CUDA_VISIBLE_DEVICES=3`时Falcor使用物理GPU 3，Torch/SlangPy使用进程内`cuda:0`，report两字段均记录`3`。
- Good：`bash scripts/run_falcor_python.sh --gpus 2,3,4 -- -m ncls.cli learn train <config> artifacts/run/checkpoint.pt`启动一个三卡同步DDP训练。
- Good：manifest同时登记`windows-vs2022`与`linux-gcc`；每台机器只生成和维护自己的ignored build output。
- Base：未设置GPU变量时单GPU机器沿用Falcor与Torch默认GPU 0。
- Bad：只限制Torch可见域却让Falcor始终使用物理GPU 0；在选择GPU 3时会形成跨卡interop。
- Bad：绕过`--gpus`手动启动多个进程，或让非rank0写checkpoint造成竞争。
- Bad：每个测试/session新建一个Falcor device；Falcor 8 Linux可能在多次创建后退回`llvmpipe`而继续运行。
- Bad：在Ubuntu移动从Windows复制来的ignored目录，然后把该本机操作当作Windows路径迁移证据。

### 6. Tests Required

- unit：平台API选择、显式物理序号、非法序号、进程级复用和软件adapter拒绝。
- shell/static：launcher拒绝多值和内部变量单独设置；Conda compatibility版本固定；多GPU驱动探针不得用会触发`pipefail`的`head`提前关闭管道。
- shell/static：DDP列表校验、torchrun参数、rank0写出与NCCL backend。
- Linux GPU：在`CUDA_VISIBLE_DEVICES=0`下同时断言Torch只见一张卡、Falcor adapter为目标NVIDIA卡、七文件GPU集合无skip，并完成SlangPy训练/evaluate。
- deployment：成功报告断言`cuda_visible_devices == falcor_gpu_index == "0"`且所有verified output可复用。
- 跨机边界：静态断言toolchain manifest保留两个平台的仓库相对build root，受管文件不含开发机绝对路径；`git check-ignore`确认本机构建与external内容不进入同步范围。

### 7. Wrong vs Correct

```bash
# 错：Torch只看到物理GPU 3，但Falcor仍按默认物理GPU 0创建Vulkan device
CUDA_VISIBLE_DEVICES=3 python -m ncls.cli learn train ...

# 对：统一launcher把物理序号同时交给Falcor，并为Torch映射为cuda:0
CUDA_VISIBLE_DEVICES=3 bash scripts/run_falcor_python.sh -m ncls.cli learn train ...
```

```jsonc
// 错：用Ubuntu本机构建目录覆盖Windows布局
{"windows": {"build_root": "external/Falcor/build/linux-gcc"}}

// 对：受管manifest保留各平台相对布局，生成物由各自主机维护
{"windows": {"build_root": "external/Falcor/build/windows-vs2022"},
 "linux": {"build_root": "external/Falcor/build/linux-gcc"}}
```

## 静态状态下的交接

- 本应运行的命令与期望结果写进根目录 `TESTING.md` 对应小节（Setup / 命令 / 期望 / 静态分析覆盖不到的边界 / 已知 `type: ignore`）。只写本次改动相关的部分，不重复全项目测试计划。
- 报告里区分「已静态检查」与「待远程验证」；不用"应该能跑"代替后者。
- 远程结果回来后，由本地会话把结论回写 `docs/research/experiment_log.md` 或对应文档，并删掉 `TESTING.md` 中已过时的段落。

## 报告格式

会话内第一次涉及验证时写一行：

`环境：<完整 Windows | Linux reference | 仅 GPU | 静态>；GPU=<卡名或无>；neural-shading=<有 | 无>；Falcor 构建=<Windows | Linux | 无>；本会话可做：<...>`
