# Reference backend 部署

`ReferenceBackendCapability` 统一执行 LayerStack、MERL、OpenPBR、MaterialX 和 MDL reference。构建清单位于 `references/reference-backend-toolchains.json`，其中保留 Windows/Linux 各自的相对构建布局。

## 资产与环境

部署只获取锁定的第三方源码/SDK，构建 reference runtime，并运行仓库 fixture probe。脚本不管理源材质资产，也不安装 Conda、显卡驱动或使用 sudo。原始 vMaterials、MERL、Poly Haven 等由用户放在 `assets/`。

唯一环境为 `neural-shading`。被忽略的 external/build/assets/outputs/artifacts 属于各主机本地状态，Git 同步不会搬运它们。Windows 布局为 `external/Falcor/build/windows-vs2022`，Linux 为 `external/Falcor/build/linux-gcc`。

## Windows

```powershell
.\scripts\build_reference_backend.ps1 -Configuration Release
conda run -n neural-shading python -m ncls reference doctor
conda run -n neural-shading python -m ncls reference probe --device 0
```

viewer 使用 `scripts/build_viewer.ps1` 单独构建；overlay 构建结束后 Falcor 工作树保持干净。

## Linux

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
conda activate neural-shading
python -m ncls reference doctor
python -m ncls reference probe --device 0
python -m ncls train 0,1 --config configs/training/runs/nvidia-layer-stack-smoke.yaml
```

部署脚本的设备选择用于部署 probe。日常训练只在 train 参数指定 GPU 一次，自动完成单卡或 DDP 装配。每个 rank 的 Torch/SlangPy 使用 cuda:0，Falcor 使用该 rank 的物理卡。

Linux 部署安装 `defaults::cuda-compat=12.8.1`。当实际 driver 主版本低于 570 时，Python runtime 将当前 Conda 前缀的 cuda-compat 放到动态库搜索路径；较新 driver 使用系统库。发行版与 driver 是否可运行由真实构建和设备 probe 判断。

## 工具与验收

```bash
python -m ncls.runtime --device 0 -- -m pytest tests/gpu/test_reference_query_dispatcher.py tests/gpu/test_reference_backend_contracts.py -q
python -m ncls train 0 --config configs/training/runs/nvidia-mdl-effect-pigment-smoke.yaml
python -m ncls validate outputs/<config>/<run>/checkpoints/latest.pt --batches 1 --device 0
```

新产出在对应 outputs run 内。Linux 数值 validation 保留；图像 eval 当前在共同接口返回 None，不启动 viewer 或跨机任务。测试的实际执行范围与待 Linux 实机验证项目见 [TESTING.md](../TESTING.md)。

2026-08-29 的旧 Linux 部署记录覆盖 Ubuntu 22.04.5、RTX A6000、driver 550.78 与锁定 Falcor/MDL SDK，当时七文件 GPU 集合为 20 passed。该历史记录不能作为 2026-09-05 新入口和 DDP 的实机验证证据。
