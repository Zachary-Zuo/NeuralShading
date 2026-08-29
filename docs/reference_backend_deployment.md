# 统一 Reference Backend 部署

## 它是什么

`ReferenceBackendCapability`是LayerStack、MERL、OpenPBR、MaterialX和MDL五类canonical reference的唯一公共底层能力。上层只调用`create_reference_backend()`、`doctor()`和`open()`；平台device、Falcor import/build layout以及各program toolchain都封装在backend内部。

唯一构建真相是`references/reference-backend-toolchains.json`。它锁定Falcor/Slang、项目需要的第三方源码、Windows/Linux MDL SDK package和五个program的provider requirement。`ReferenceBackendDescriptor`区分跨平台semantic identity与本机build identity；session会把完整identity写入reference query和training checkpoint。

## 资产边界

部署只处理项目源码、第三方源码/toolchain和编译输出。它不下载、移动、复制或写入`assets/`，也不调用任何source-material fetcher。用户迁移工程时自行复制source assets；缺少资产不妨碍compile deployment和仓库fixture probe完成。

MDL SDK binary package属于编译toolchain，不属于source asset，因此可以由manifest驱动获取。vMaterials、MERL表、Poly Haven纹理和其他源材质都属于`assets/`，不在部署范围内。

## Windows

Windows完整环境运行：

```powershell
.\scripts\build_reference_backend.ps1 -Configuration Release
.\scripts\run_falcor_python.ps1 -m ncls.cli reference doctor
.\scripts\run_falcor_python.ps1 -m ncls.cli reference probe
```

公共build入口验证manifest与既有external/toolchain，编译portable MDL program provider，并用LayerStack和仓库内MDL fixture完成真实D3D12 query。Windows viewer仍由`scripts/build_viewer.ps1`单独构建，因为Linux部署只承载headless reference/Vulkan。

## Linux/A6000

Linux发行版和具体系统版本在实际服务器上确定，不提前冻结。前提是机器已经安装可用的NVIDIA driver、Vulkan runtime、Conda、Git、CMake和C++编译器；脚本不会使用`sudo`，不会安装driver或Conda。

```bash
bash scripts/deploy_reference_linux.sh
```

脚本依次执行环境preflight、Conda环境创建/更新、manifest锁定依赖获取与校验、Falcor setup/build、MDL provider build、backend doctor/device/compile probe，并把逐步状态与`assets: not-managed`写入ignored deployment report。再次运行必须复用已经通过hash/commit校验的输出；dirty external、错误commit、partial SDK或hash漂移会fail closed，不会清理用户目录。

部署完成后，上层命令始终通过launcher进入同一环境：

```bash
bash scripts/run_falcor_python.sh -m ncls.cli reference doctor
bash scripts/run_falcor_python.sh -m ncls.cli reference probe
```

## 用户复制资产后的验收

资产到位后才执行五个program的代表性真实snapshot、same-device CUDA/lease测试，以及固定`effect-pigment-metallic`的两步online training：

```bash
bash scripts/run_falcor_python.sh -m pytest \
  tests/gpu/test_reference_backend_contracts.py \
  tests/gpu/test_reference_query_dispatcher.py \
  tests/gpu/test_mdl_native_crosscheck.py -q

bash scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json \
  artifacts/training/mdl-linux-smoke/checkpoint.pt
bash scripts/run_falcor_python.sh -m ncls.cli learn evaluate \
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json \
  artifacts/training/mdl-linux-smoke/checkpoint.pt --batches 1
```

只有这些Linux/Vulkan实机gate通过后，才能把该服务器的distro、driver、glibc、compiler、Conda、Falcor、MDL与backend identity登记为已验证。Windows结果不能替代Linux完成声明。
