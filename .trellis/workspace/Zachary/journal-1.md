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
