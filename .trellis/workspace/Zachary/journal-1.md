# Journal - Zachary (Part 1)

> AI development session journal
> Started: 2026-08-25

---



## Session 1: 01 可复用散射数学原语

**Date**: 2026-08-26
**Task**: 01 可复用散射数学原语
**Branch**: `main`

### Summary

建立公共 Falcor-free scattering math，迁移 LayerStack reference/legacy 调用者，加入 NVIDIA proposal、LTC analytic control 与独立 GPU 数学 oracle；unit/GPU/viewer/reference 全 gate 通过。

### Git Commits

| Hash | Message |
|------|---------|
| `c9e5ae5` | (see git log) |

### Status

[OK] **Completed**


## Session 2: 完成 02 Directional Mollification 数据充分性

**Date**: 2026-08-26
**Task**: 完成 02 Directional Mollification 数据充分性
**Branch**: `main`

### Summary

冻结并验证方向 mollification adequacy protocol；旧 v5 audit 触发 supplement；发布 30-state composite corpus 与 training entry，补齐 fail-stop curriculum reader、统计重算 validator、schema、测试和长期 spec。

### Git Commits

| Hash | Message |
|------|---------|
| `123a94b` | (see git log) |

### Status

[OK] **Completed**


## Session 3: 统一 Pipeline 架构重整

**Date**: 2026-08-27
**Task**: 统一 Pipeline 架构重整
**Branch**: `main`

### Summary

完成 source、reference query、offline/live batch、方法注册、TrainingCheckpoint@2、ScatteringPackage@1 与双 slot viewer 的原子迁移；删除旧并行路径，通过全量 unit、Falcor GPU 与 Release viewer 构建，并归档任务。

### Git Commits

| Hash | Message |
|------|---------|
| `7a10a5f` | (see git log) |
| `d5b8016` | (see git log) |

### Status

[OK] **Completed**


## Session 4: 忠实复现 NVIDIA Neural Materials

**Date**: 2026-08-27
**Task**: 忠实复现 NVIDIA Neural Materials
**Branch**: `main`

### Summary

补齐 NVIDIA RTA 2024 functional reproduction 的 encoder、hierarchical latent、双 65k online training、matched sampler、packed FP16 package 与真实 neural PT；按用户决定冻结并登记 step 200k 结果，完成全量验证、报告与归档。

### Git Commits

| Hash | Message |
|------|---------|
| `5e013d3d616dea14f3192fe8293f06ef72e880ca` | (see git log) |

### Status

[OK] **Completed**


## Session 5: 修复 Viewer PT 空间材质 footprint

**Date**: 2026-08-27
**Task**: 修复 Viewer PT 空间材质 footprint
**Branch**: `main`

### Summary

用 raw UV/LOD probe 定位 Falcor camera basis 共同尺度导致 ray cone 放大约一万倍；抽取共享 PathSurface，修复 source/neural PT 纹理与 latent 过滤，补 GPU oracle、walnut/denim 与 PT/deferred 证据，并完成全量测试和 Release 构建。

### Git Commits

| Hash | Message |
|------|---------|
| `b23c3f6` | (see git log) |

### Status

[OK] **Completed**


## Session 6: 固定 Viewer Capture Harness

**Date**: 2026-08-27
**Task**: 固定 Viewer Capture Harness
**Branch**: `main`

### Summary

修复 difference EXR 复用双 panel 纹理导致的横向拉伸与 shape 不一致；固定 ready PT slot 导出为 1024 spp，并完成 unit、Release build 与真实 headless EXR header 验证。

### Git Commits

| Hash | Message |
|------|---------|
| `411ef6f` | (see git log) |

### Status

[OK] **Completed**


## Session 7: Fancy reference material 候选研究

**Date**: 2026-08-27
**Task**: Fancy reference material 候选研究
**Branch**: `main`

### Summary

完成 NVIDIA 2024/2026、公开 MaterialX、Omniverse MDL packs、Ling-Qi Yan 与历史材质模型的候选报告；推荐 vMaterials 2 作为首个新增 MDL source。

### Git Commits

| Hash | Message |
|------|---------|
| `cdc5fd9` | (see git log) |

### Status

[OK] **Completed**


## Session 8: 原生 MDL Reference 与 falcor2 官方对照

**Date**: 2026-08-27
**Task**: 原生 MDL Reference 与 falcor2 官方对照
**Branch**: `main`

### Summary

实现项目 MDL SDK bridge 到当前 Falcor 8 的唯一正式 reference 路径，接入 canonical source/editor、offline HDF5、CUDA live batch 与六种 vMaterials；falcor2 保持隔离 oracle。artifact identity、capability audit、native cross-check、formal parity、unit 与 MDL GPU gate 均已通过，并已归档任务。

### Git Commits

| Hash | Message |
|------|---------|
| `5e69ba9` | (see git log) |
| `4f80300` | (see git log) |

### Status

[OK] **Completed**


## Session 9: MDL reference viewer 与 firefly 根因修复

**Date**: 2026-08-28
**Task**: MDL reference viewer 与 firefly 根因修复
**Branch**: `main`

### Summary

将六种 vMaterials 的 MDL SDK compiled artifact 接入当前 Falcor 8 viewer；修复 car paint/ceramic 因 fixed-GGX 与 MDL PDF 错配产生的 firefly，改用同 target code matched sample/pdf 与正确 MIS，并通过 unit、GPU、Release build 和 1024 spp capture。

### Git Commits

| Hash | Message |
|------|---------|
| `234bb0a` | (see git log) |

### Status

[OK] **Completed**


## Session 10: 统一材质散射合同根本迁移

**Date**: 2026-08-28
**Task**: 统一材质散射合同根本迁移
**Branch**: `main`

### Summary

将五种 source reference 与 neural package 迁到唯一 prepare/evaluate/sample/pdf 合同，删除 viewer estimator 分支与兼容路径；修复 OpenPBR 掠射 sample tuple 重建导致的 firefly 和 half EXR 溢出，并完成 unit、GPU、integration、Release viewer 与 1024 spp capture 验证。

### Git Commits

| Hash | Message |
|------|---------|
| `7a7d78d` | (see git log) |

### Status

[OK] **Completed**


## Session 11: 完成 PT 椒盐噪点与交互累积根迁移

**Date**: 2026-08-28
**Task**: 完成 PT 椒盐噪点与交互累积根迁移
**Branch**: `main`

### Summary

修复公共 PT primary continuation 长尾，并将交互式 PT 迁为每 dispatch 1 spp、持续累积；headless 独占 capture target/batch，完成 source/package、MDL 与 Release 验证。

### Git Commits

| Hash | Message |
|------|---------|
| `2804c11` | (see git log) |
| `7e55750` | (see git log) |
| `5c81551` | (see git log) |
| `d24a3f4` | (see git log) |

### Status

[OK] **Completed**


## Session 12: 统一材质 Reference 在线训练数据回路

**Date**: 2026-08-28
**Task**: 统一材质 Reference 在线训练数据回路
**Branch**: `main`

### Summary

统一五类源材质的 prepare/evaluate/sample/pdf 查询与在线训练生产回路，删除 LayerStack 专用离线 HDF5/旧 data 路径和兼容层，并同步训练、MethodBundle、viewer、测试及中文规范文档。

### Main Changes

- 引入统一 ReferenceQueryDispatcher、typed evaluator/sampler batch 与 OnlineTrainingProducer。
- 统一 NVIDIA evaluator 的线性 f 语义及与 evaluate 匹配的 sample/pdf 训练数据。
- 删除旧 ncls.data、corpus/HDF5 schema 和 LayerStack 专用采集入口，不保留向后兼容。
- 修复 MaterialX 局部法线导致的无效方向亮点问题，并为五类材质接入同一查询合同。

### Git Commits

| Hash | Message |
|------|---------|
| `cf01d33` | (see git log) |
| `8ea0e60` | (see git log) |

### Testing

- [OK] conda run -n neural-shading python -m pytest tests\\unit -q：83 passed。
- [OK] scripts/run_falcor_python.ps1 -m pytest tests\\gpu -q：29 passed。
- [OK] scripts/run_falcor_python.ps1 -m pytest tests\\integration -q：3 passed。
- [OK] LayerStack/MaterialX 两步训练 smoke、learn evaluate、learn export 与 package validate 均通过。
- [OK] Release viewer 构建、compileall、git diff --check 与锁定上游 clean 检查均通过。

### Status

[OK] **Completed**


## Session 13: 补齐 vMaterials preset catalog 与 capability audit

**Date**: 2026-08-28
**Task**: 补齐 vMaterials preset catalog 与 capability audit
**Branch**: `main`

### Summary

完成 11 个 vMaterials family、172 个 authored preset 的 SDK discovery、资源闭包与 capability catalog；将 Rgba_16 等输入格式无损接入通用 MDL/Falcor/viewer typed texture 路径，punched suede 仅因 geometry.cutout_opacity 输出合同缺失而显式失败关闭。

### Git Commits

| Hash | Message |
|------|---------|
| `d8d9795` | (see git log) |
| `5b9672c` | (see git log) |

### Status

[OK] **Completed**


## Session 14: 完成 Neural Shading 与 Appearance 文献深度研究

**Date**: 2026-08-29
**Task**: 完成 Neural Shading 与 Appearance 文献深度研究
**Branch**: `main`

### Summary

完成28篇可追溯论文报告、五份跨论文综合、NVIDIA correspondence与17项可证伪假设；补齐NeLiF、NeLT、Superposed DFF和AE(uniform)证据边界，并通过独立复核与任务质量门。

### Git Commits

| Hash | Message |
|------|---------|
| `4f54a53` | (see git log) |

### Status

[OK] **Completed**


## Session 15: 完成 vMaterial Metal 神经材质系统

**Date**: 2026-08-30
**Task**: 完成 vMaterial Metal 神经材质系统
**Branch**: `main`

### Summary

完成 canonical 架构递归迁移、692-export Metal reference、共享 texture codec 与 full evaluator、matched sampler、Package@2/Slang/viewer、Windows 四阶段 online correctness 验证和 Linux 单 GPU 长训 handoff；六个子任务及父任务均已提交归档，Linux 原生 smoke 保持待目标机执行。

### Git Commits

| Hash | Message |
|------|---------|
| `0cfeb88` | (see git log) |
| `6c20af9` | (see git log) |
| `867c67b` | (see git log) |
| `e8deaaa` | (see git log) |
| `b1008b6` | (see git log) |
| `bccf2d8` | (see git log) |
| `3757450` | (see git log) |

### Status

[OK] **Completed**


## Session 16: Viewer 通用材质接口与 Metal 双侧 PT 部署

**Date**: 2026-09-05
**Task**: Viewer 通用材质接口与 Metal 双侧 PT 部署
**Branch**: `main`

### Summary

完成最新 hybrid 四入口部署、统一 scene scattering 和 PT/Deferred renderer、双侧模式标题、整帧调度及相关旧接口清理；修复 deferred 材质输入与 shader 链接错误。96 项相关测试通过，双侧 1024 spp capture 与逐位一致 replay 完成。800×900 单侧 Neural PT GPU median 420.8 ms，deferred 单次 3.66 ms。用户已批准本地提交并归档，任务已归档，未 push；结果位于 artifacts/viewer/metal-viewer-refresh-lighting/current/validation.md。

### Git Commits

| Hash | Message |
|------|---------|
| `19073e07c5398630000fc9536fbc46cf95e08e48` | (see git log) |

### Status

[OK] **Completed**
