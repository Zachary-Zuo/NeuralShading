# 统一 Reference Backend 部署

## 它是什么

`ReferenceBackendCapability`是LayerStack、MERL、OpenPBR、MaterialX和MDL五类canonical reference的唯一公共底层能力。上层只调用`create_reference_backend()`、`doctor()`和`open()`；平台device、Falcor import/build layout以及各program toolchain都封装在backend内部。

唯一构建真相是`references/reference-backend-toolchains.json`。它锁定Falcor/Slang、项目需要的第三方源码、Windows/Linux MDL SDK package和五个program的provider requirement。`ReferenceBackendDescriptor`区分跨平台semantic identity与本机build identity；session会把完整identity写入reference query和training checkpoint。

## 资产边界

部署只处理项目源码、第三方源码/toolchain和编译输出。它不下载、移动、复制或写入`assets/`，也不调用任何source-material fetcher。用户迁移工程时自行复制source assets；缺少资产不妨碍compile deployment和仓库fixture probe完成。

MDL SDK binary package属于编译toolchain，不属于source asset，因此可以由manifest驱动获取。vMaterials、MERL表、Poly Haven纹理和其他源材质都属于`assets/`，不在部署范围内。

Git同步只覆盖受管的manifest、脚本、源码和文档。`external/`、`build/`、`assets/`与`artifacts/`均为每台主机的本地ignored状态，因此Ubuntu上的目录移动、权限修复和构建不会传播到Windows，也不能替Windows完成路径迁移。受管manifest会同时保留Windows的`external/Falcor/build/windows-vs2022`与Linux的`external/Falcor/build/linux-gcc`相对布局；两侧分别生成自己的构建输出，不互相覆盖。

## Windows

Windows完整环境运行：

```powershell
.\scripts\build_reference_backend.ps1 -Configuration Release
.\scripts\run_falcor_python.ps1 -m ncls.cli reference doctor
.\scripts\run_falcor_python.ps1 -m ncls.cli reference probe
```

公共build入口验证manifest与既有external/toolchain，编译portable MDL program provider，并用LayerStack和仓库内MDL fixture完成真实D3D12 query。Windows viewer仍由`scripts/build_viewer.ps1`单独构建，因为Linux部署只承载headless reference/Vulkan。

## Linux/A6000

Linux发行版和具体系统版本由实际服务器验证，不提前冻结。前提是机器已经安装可用的NVIDIA driver、Vulkan runtime、Conda、Git、CMake和C++编译器；脚本不会使用`sudo`，不会安装driver或Conda。Linux部署会在`neural-shading`中固定安装`cuda-compat=12.8.1`：宿主driver主版本低于570时，launcher优先加载这组用户态compatibility库，以支持SlangPy运行时生成的CUDA 12.8 PTX；driver 570及以上继续使用系统库。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy_reference_linux.sh
```

脚本依次执行环境preflight、Conda环境创建/更新、manifest锁定依赖获取与校验、Falcor setup/build、MDL provider build、backend doctor/device/compile probe，并把逐步状态与`assets: not-managed`写入ignored deployment report。再次运行必须复用已经通过hash/commit校验的输出；dirty external、错误commit、partial SDK或hash漂移会fail closed，不会清理用户目录。

部署完成后，上层命令始终通过 launcher 进入同一环境。单卡入口接受 `CUDA_VISIBLE_DEVICES=<单个物理序号>`，训练命令同时使用 `--devices <同一序号>`；Falcor 使用该物理卡，Torch/SlangPy 使用进程内 `cuda:0`。训练配置指定多个 device 时，CLI 自动启动 torchrun/NCCL DDP job，各 rank 共享梯度且仅 rank 0 写 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls reference doctor
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls reference probe

# 多GPU训练仍走同一入口；具体run必须是已冻结的DDP配置
bash scripts/run_falcor_python.sh -m ncls train \
  <ddp-run.yaml> --devices 2,3,4 --output <checkpoint.pt>
```

## 用户复制资产后的验收

资产到位后先构建validation-only的OpenPBR C++ probe，再执行五个program的代表性真实snapshot、same-device CUDA/lease测试，以及固定`effect-pigment-metallic`的两步online training：

```bash
conda run -n neural-shading cmake -S tools/reference/openpbr_probe \
  -B build/openpbr-probe -G Ninja -DCMAKE_BUILD_TYPE=Release
conda run -n neural-shading cmake --build build/openpbr-probe \
  --config Release --target ncls_openpbr_probe --parallel 16
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m pytest \
  tests/gpu/test_reference_query_dispatcher.py \
  tests/gpu/test_reference_backend_contracts.py \
  tests/gpu/test_mdl_native_crosscheck.py \
  tests/gpu/test_mdl_hlsl_feasibility.py \
  tests/gpu/test_merl_reference_gpu.py \
  tests/gpu/test_openpbr_reference_gpu.py \
  tests/gpu/test_layer_stack_ir_gpu.py -q

CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls train \
  configs/training/runs/nvidia-mdl-effect-pigment-smoke.yaml --devices 0 \
  --output artifacts/training/mdl-linux-smoke/checkpoint.pt
CUDA_VISIBLE_DEVICES=0 bash scripts/run_falcor_python.sh -m ncls validate \
  artifacts/training/mdl-linux-smoke/checkpoint.pt --batches 1 --device 0
```

2026-08-29的已验证环境为Ubuntu 22.04.5、10张RTX A6000、driver 550.78、glibc 2.35、GCC 11.4.0、Conda 24.1.2、Falcor `9dc819c162b2070335c65060436041690b7937f8`与MDL SDK `2025.0.0-387700.1252`。GPU 0上的七文件集合为`20 passed`，固定MDL两步训练与checkpoint evaluate通过；最终重复部署报告为`artifacts/deployment/reference-linux/20260829T125648Z/report.json`，其中`cuda_visible_devices`与`falcor_gpu_index`均为`0`。这个结论只覆盖该实机环境，Windows结果仍不能替代Linux证据。

vMaterials 2 Metal 当前在reference backend与source assets就绪后，继续执行[预算内Metal Linux pilot](metal_linux_training.md)中的single-material direct/hybrid matched pair。该pilot固定单进程单GPU，不是DDP scaling或692-export long；不把上面的NVIDIA两步smoke当作Metal训练证据。
