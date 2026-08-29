# 统一跨平台 Reference Backend：实施计划

## 0. 执行原则

- inline模式由主会话直接实现和检查，不分派sub-agent。
- Phase 2重新加载`trellis-before-dev`及project/core/data/learning/viewer specs；每次切换Windows/Linux环境都按`dev-environment.md`重新报告状态。
- 不修改用户已有dirty files，不修改`external/`上游；若需要upstream patch，停止并回planning。
- 本任务保留Windows、Linux两个gate，不拆任务。公共identity、API、部署report与training checkpoint需要跨阶段共同演进；Linux完成前不archive。
- 当前工作树中的MDL capability试验尚未提交。正式实现第一步按本设计重构或删除，不把其compatibility常量/legacy构造器当既成合同。

## 1. 冻结公共registry与build manifest

- [x] 审计五个canonical program/source descriptor/reference registry一致性，修复stable registry中的旧family id或过时capability文字。
- [x] 新增`ncls.reference-backend-toolchains@1` manifest/schema，登记Falcor/Slang、external Git providers、Windows/Linux MDL SDK与program requirement映射；明确`asset_policy=external-only-no-source-assets`。
- [x] 将当前MDL-only toolchain试验收进root build manifest的内部provider record；删除公共层MDL专属命名和Windows compatibility constant。
- [x] unit tests覆盖schema/version、duplicate/unknown id、platform/arch、unsafe path、Git revision、archive size/hash/type与source-asset URL拒绝。
- [x] implementation identity包含manifest/schema与相关source/CMake；不读取未跟踪absolute path作为semantic identity。

**Rollback A**：只建立manifest/registry tests。若依赖模型不成立，回滚新manifest，不触碰runtime调用链。

## 2. 实现公共backend capability/session

- [x] 新增`ReferenceBackendDescriptor`、generic status/report、resolver与`ReferenceBackendCapability`；Windows映射D3D12，Linux映射Vulkan，unknown OS/arch fail closed。
- [x] 将Falcor import、environment augmentation、device创建、typed binder与现有dispatcher实现收进backend；公开`backend.open()`返回`ReferenceBackendSession`。
- [x] session identity加入backend build identity；保持query/result/lease、same-device CUDA tensor、双slot和无host response readback合同。
- [x] `OnlineTrainingProducer`、tools、viewer helper和GPU tests全部改用backend factory/session。
- [x] 删除`import_falcor()`、`create_falcor_device()`、旧dispatcher直接构造入口与无外部合同的compatibility alias。
- [x] 静态test确认正式upper modules没有`sys.platform`、`DeviceType.D3D12/Vulkan`、platform build目录或动态库后缀。

**Review B**：五个program使用同一backend/session类型；没有family-specific query shader/producer/dispatcher，且全部旧调用方已迁移。

## 3. Program preflight与MDL内部provider

- [x] 为`ReferenceProgramDefinition`增加默认provider descriptor/preflight hook；backend doctor统一枚举五个program并汇总generic status。
- [x] LayerStack/MERL/OpenPBR/MaterialX声明真实code/toolchain requirement，不把source asset缺失计为deployment失败。
- [x] 实现`MdlProgramToolchainProvider` factory与descriptor；source family/program/tools不再直接构造subprocess bridge或传legacy路径参数。
- [x] portable C++ bridge使用RAII `SharedLibrary`，Windows/Linux loader只在该类分支；plugin路径来自descriptor/CLI，业务代码不写后缀。
- [x] CMake统一C++20/warnings/Release layout，Linux链接`${CMAKE_DL_LIBS}`；Windows Release先证明无回归。
- [x] artifact/compiler identity扩展semantic/build/platform/toolchain字段；cache key包含两级identity，旧cache只miss不删除。
- [x] tests覆盖缺SDK/library/plugin/target-code/bridge、错误arch、artifact tamper、cache identity和discovery/inspect/native操作。

**Rollback C**：公共backend已能运行四个非MDL program；MDL provider可独立回滚registration，不建立旧Windows bridge兼容层。

## 4. Build helpers与Linux compile deployment

- [x] 建立manifest驱动的fetch/verify/build orchestrator，Git clone只获取固定revision并拒绝dirty/wrong target；binary archive安全处理partial、containment、traversal与symlink。
- [x] 新增Windows公共backend build/doctor入口并迁移viewer/测试脚本；删除仅转发旧命令的wrapper。
- [x] 新增`build_mdl_program_provider.sh`或等价内部provider helper，由公共orchestrator调用，不作为上层入口。
- [x] 新增`bash scripts/deploy_reference_linux.sh`：preflight、Conda env、external/toolchain fetch、Falcor setup/build、MDL provider build、doctor/device/compile probe和report。
- [x] deployment代码静态拒绝`sudo`、Conda/driver installer、source-material fetcher与`assets/`写路径；assets缺失unit/integration fixture必须仍成功走到compile plan。
- [x] `build_falcor_python_linux.sh`改为读取公共manifest；发行版字符串只记录/warn，真实build/probe决定结果。
- [x] report schema记录实际环境、identity、每步fresh/reused/status、`assets: not-managed`和下一条launcher命令。
- [x] tests覆盖argument parsing、preflight-before-download、no-assets policy、idempotent decision、dirty external、partial target与report schema。

**Rollback D**：deployment只新增root-owned scripts/tools；失败保留report，不修改或清理existing external/build/assets。

## 5. Fixed MDL online training

- [x] 定义`nvidia.mdl-fixed-uniform@1` native feature layout，稳定编码支持的float/int/bool/enum/color/vector typed参数与归一化规则。
- [x] adapter只接受一个snapshot和1×1 materialization；拒绝2D spatial texture依赖、unsupported/nonfinite参数和multi-snapshot。
- [x] NVIDIA method注册`mdl.program@1` adaptation contract；producer/runner/CLI保持family-agnostic。
- [x] 新增`effect-pigment-metallic`两步smoke config和unit tests，identity准确表达fixed-uniform范围。
- [x] Windows runner smoke验证target来自backend session、finite loss/gradient、materialization、checkpoint load与完整backend/reference identity。
- [ ] Linux/A6000 runner执行同一smoke并验证checkpoint identity。

**Rollback E**：adapter独立注册；若模型合同不成立，移除registration/config并回planning，不用dummy feature绕过。

## 6. 文档与spec迁移

- [x] 更新`.trellis/spec/project`的unified pipeline/dev environment/repository policy相关规则，写清backend与assets-not-managed边界。
- [x] 更新data reference-query/MDL、learning online-training、viewer相关spec与稳定文档。
- [x] 更新五个reference package README/reference registry，统一canonical program/source id与Linux支持表述。
- [x] 更新根`TESTING.md`和部署说明，分别给出compile deployment、资产复制后的query验证、MDL training与Windows viewer命令。
- [x] 删除旧Windows-only、MDL-only“公共后端”与deployment下载资产表述。

## 7. Windows quality gate

执行前重新报告完整Windows状态。所有Python命令使用`neural-shading`：

```powershell
conda run -n neural-shading python -m pytest tests/unit -q
conda run -n neural-shading python -m compileall -q src tests tools
git diff --check

.\scripts\build_reference_backend.ps1 -Configuration Release
.\scripts\run_falcor_python.ps1 -m ncls.cli reference doctor
.\scripts\run_falcor_python.ps1 -m pytest `
  tests/gpu/test_reference_query_dispatcher.py `
  tests/gpu/test_reference_backend_contracts.py `
  tests/gpu/test_mdl_native_crosscheck.py `
  tests/gpu/test_mdl_hlsl_feasibility.py `
  tests/gpu/test_merl_reference_gpu.py `
  tests/gpu/test_openpbr_reference_gpu.py `
  tests/gpu/test_layer_stack_ir_gpu.py -q

.\scripts\run_mdl_native_parity.ps1 -OutputDir <new-artifacts-dir>
.\scripts\run_mdl_reference_parity.ps1 -Mode formal -OutputDir <new-artifacts-dir>

.\scripts\run_falcor_python.ps1 -m ncls.cli learn train `
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json `
  artifacts/training/mdl-windows-smoke/checkpoint.pt
.\scripts\run_falcor_python.ps1 -m ncls.cli learn evaluate `
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json `
  artifacts/training/mdl-windows-smoke/checkpoint.pt --batches 1

.\scripts\build_viewer.ps1 -Configuration Release
git -C external/Falcor status --short
```

- [x] unit/static、五program公共backend GPU、MDL parity/training与viewer gate通过。
- [x] tolerance只用任务前既有gate或formal前独立calibration，不根据结果放宽。
- [x] observed build/training时间和显存只写`artifacts/`，不成为hard gate。
- [x] Windows完成后记录task progress，但保持`in_progress`等待Linux。

## 8. Linux/A6000 quality gate（目标服务器后半段）

### 8.1 环境与无资产deployment

```bash
uname -a
cat /etc/os-release
ldd --version | head -n 1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
gcc --version
conda env list

bash scripts/deploy_reference_linux.sh
bash scripts/deploy_reference_linux.sh
```

- [ ] 首次和重复deployment都成功，第二次真实复用verified outputs。
- [ ] 在`assets/`缺失或临时不可见的检验环境中仍成功；report明确`assets: not-managed`。
- [ ] report记录实际distro/driver/glibc/Vulkan/compiler/Conda/Falcor/MDL/backend identity。

### 8.2 用户复制资产后的公共backend验收

```bash
scripts/run_falcor_python.sh -m ncls.cli reference doctor
scripts/run_falcor_python.sh -m pytest \
  tests/gpu/test_reference_query_dispatcher.py \
  tests/gpu/test_reference_backend_contracts.py \
  tests/gpu/test_mdl_native_crosscheck.py \
  tests/gpu/test_mdl_hlsl_feasibility.py \
  tests/gpu/test_merl_reference_gpu.py \
  tests/gpu/test_openpbr_reference_gpu.py \
  tests/gpu/test_layer_stack_ir_gpu.py -q

scripts/run_falcor_python.sh -m ncls.cli learn train \
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json \
  artifacts/training/mdl-linux-smoke/checkpoint.pt
scripts/run_falcor_python.sh -m ncls.cli learn evaluate \
  configs/learning/nvidia-rta2024-mdl-effect-pigment-smoke.json \
  artifacts/training/mdl-linux-smoke/checkpoint.pt --batches 1
```

- [ ] 五个program代表性真实snapshot经同一backend/session完成evaluate/sample/pdf、same-device CUDA与lease tests。
- [ ] MDL fixture/native query与Windows冻结query按预先冻结容差一致。
- [ ] MDL fixed snapshot training/checkpoint通过。
- [ ] external clones保持固定commit与clean。

## 9. 最终检查与收尾

- [ ] 使用`trellis-check`执行spec compliance、lint/type/static、unit/GPU/integration与diff检查。
- [ ] 使用`trellis-update-spec`固化公共backend、provider和deployment资产边界。
- [ ] 确认Git diff只包含本任务文件，排除所有既有/未知dirty文件。
- [ ] Windows与Linux gate都通过后再按Trellis流程commit、finish/archive；不push。
