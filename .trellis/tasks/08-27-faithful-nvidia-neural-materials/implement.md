# 忠实复现 NVIDIA Neural Materials：执行计划

## 执行原则

- 按用户授权持续执行；只在超出冻结范围、需要破坏性外部操作或不可恢复外部阻塞时停下。
- 开始任何测试、构建或训练前，先按 `dev-environment.md` 判定本机是完整/仅 GPU/静态并在进度报告中给证据。
- Python/pytest只用 `conda run -n neural-shading ...`；viewer只用 `scripts/build_viewer.ps1`；不修改 `external/Falcor`。
- 每个阶段先完成最小 vertical slice和对应测试，再扩大；不以“先跑300k”掩盖合同错误。

## Phase 0：规划冻结与环境门

- [x] 收敛 `prd.md`、`design.md`、本文件和 `research/current-fidelity-audit.md`，删除开放问题与互相冲突的措辞。
- [x] 记录 task-scoped continuous metadata，运行 task artifact validation并启动任务。
- [x] 加载 `trellis-before-dev`，按修改层读取 project/core/data/learning/viewer spec。
- [x] 判定开发机状态；检查 conda、CUDA/PyTorch、Falcor build/import、D3D12 viewer prerequisites和磁盘空间。若非完整状态，按规则写 `TESTING.md`，不得宣称 runtime已验证。

回滚点：只含任务文档/元数据；未动产品代码。

## Phase 1：correspondence gate 与数学 core

- [x] 将审计表扩成逐项 correspondence：论文页/Listing → PyTorch/Slang symbol → test → 差异标签/recipe field。
- [x] 恢复并改写历史 exact tests：learned frame、20D/11D input、3×64 evaluator、3×32 sampler、`exp(raw-3)`、tanh/sinh、two-lobe 2D mixture remap、linear-f adapter。
- [x] 删除 formal sampler 的 `1/32` safety lobe和 3D random mapping；若通用 proposal仍需要旧 robust 形态，用不同 fixture identity隔离，不能留在 NVIDIA formal runtime。
- [x] 让 Slang functional core的部署数据类型/中间值与 FP16 identity一致，训练 AD core保持 FP32；建立固定 calibration tolerance。

验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "nvidia or proposal or response_measure" -q
conda run -n neural-shading python -m pytest tests/gpu -k "nvidia or proposal" -q
```

回滚点：数学 core + exact tests可独立通过；尚未改变训练/package/viewer。

## Phase 2：role-aware batch 与 source-native spatial data

- [x] 定义 `TrainingRouteRequest`/named batch mapping；公共 runner与 BatchSource不按 method ID分支，fixture覆盖单 route与双 route。
- [x] 扩展 `TrainingBatch@1` 的 documented optional fields/role validation：flat query、UV/gradient/mip、native features/layout identity、stream/request identity。
- [x] OfflineBatchSource读取 HDF5中已有 position/UV/gradient/native payload字段，不丢弃 spatial context。
- [x] 为 MaterialX实现 NativeFeatureLayout、GPU texture/native parameter采样、exponential mip/Gaussian footprint/LEAN-style coarse features与 half/difference direction proposal。
- [x] 为 LayerStack实现 canonical native layout与显式 1×1 adapter，random-walk reference不改语义。
- [x] 把 live executor改为 flat tiled GPU query和至少两个 in-flight lease slots；去掉 logical batch `≤64` 限制，formal每 route真实产生65k独立 samples且无 host readback。
- [x] 实现 global-step-aware 20k cosine mollification与256 cone estimator；smoke recipe可缩小但身份隔离。

验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "training_batch or batch_source or native_feature or materialx" -q
conda run -n neural-shading python -m pytest tests/gpu -k "live_batch or materialx or layer_stack" -q
```

回滚点：data vertical slice能独立产出两条 batch并通过 contract；未接训练 lifecycle。

## Phase 3：encoder、hierarchy与统一训练 lifecycle

- [x] 将 per-state free latent模型替换为 `K→64×4→8` encoder + trainable latent mip hierarchy；网络 decoder继续使用同一 PyTorch/Slang数学 core。
- [x] 实现 bootstrap直连 encoder、全 mip materialization、drop/freeze encoder和 bilinear latent finetune；LayerStack运行1×1同一 lifecycle。
- [x] TrainingConfig增加 correspondence/run/source adaptation/recipe identities、formal静态约束、stage boundary、route recipes、Adam与scheduler字段；删除 evaluator/joint/sampler旧 phase含义。
- [x] runner用一个 Adam + global cosine贯穿300k，两条独立 route一次 joint backward；实现吞吐、work units、显存和 ETA。
- [x] sampler loss只更新 sampler head且不改变 latent/evaluator；实现并测试已冻结 KL estimator。
- [x] checkpoint保存/恢复 lifecycle、optimizer、scheduler、双 stream/RNG/materialization/validation状态；加入 interrupted-vs-uninterrupted exact trajectory测试。
- [x] 建立互斥 smoke/profile/formal configs；formal静态 validator必须拒绝25k/16、共享 stream、错误 optimizer/mollification/网络尺寸。

验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "training or checkpoint or nvidia" -q
conda run -n neural-shading python -m ncls learn train configs/learning/nvidia-rta2024-materialx-smoke.json artifacts/nvidia-faithful/materialx-smoke/checkpoint.pt
```

回滚点：smoke可完整 checkpoint/resume并导出训练状态；formal尚未启动。

## Phase 4：hierarchical ScatteringPackage 与 generic binding

- [x] material compiler输出小 compiled record + 两个 RGBA16F DDS mip chains；runtime compiler只输出共享 FP16 network weights/module closure。
- [x] 扩展 typed descriptor vocabulary和 Python manifest/tamper tests，保证 resource参与 material/package identity。
- [x] C++ loader重算/校验三身份与 descriptors，创建通用 ScatteringBinding并按 usage/reflection绑定 buffer/texture/sampler；错误拒绝、不 fallback。
- [x] viewer CMake移除 NVIDIA shader清单，package absolute module closure成为真实加载源。
- [x] NVIDIA `prepare`实现 supplemental LOD、stochastic adjacent mip与 bilinear fetch，并打包复用 learned frames/view/sampler state；parity probe提供 UV/gradient/random seed。
- [x] LayerStack 1×1和 MaterialX spatial package均能 roundtrip；人工 mip texture验证 stochastic adjacent LOD 与 wrap bilinear。

验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "scattering_package or bundle or nvidia" -q
conda run -n neural-shading python -m pytest tests/gpu -k "package or latent or nvidia" -q
```

回滚点：package parity/deferred可独立工作；PT尚可沿 reference-only构建。

## Phase 5：双 slot 与真实 neural PT

- [x] 把 NclsViewer单 reference/method状态改为两个对称 ComparisonSlot binding；每侧独立 package、mode、status、GPU资源、accumulation和timing。
- [x] 实现/重构 generic package path tracer；命中点构造完整 context/ray-cone footprint并调用 binding `prepare/evaluate/sample/pdf`。
- [x] deferred从 G-buffer传 `uv/uvDx/uvDy/materialInstanceId`，与 PT使用同一 package shader core。
- [x] `source-reference` 作为显式内建权威 transport请求进入任一 slot，capture不虚构其 package/runtime/material身份；package renderer不按 source family分支。
- [x] capture/replay实现 `ncls.viewer-capture@4 slots[2]`；失败只影响单 slot，panel/相机比例保持规格。
- [x] 增加 fixture package/受控场景，断言 neural PT确实调用 package sampler而非 reference transport。
- [x] 更新 viewer/architecture/contracts/learning/data中文稳定文档。

验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "viewer or comparison_slot or scattering_binding" -q
conda run -n neural-shading python -m pytest tests/gpu -k "path or scattering or package" -q
powershell -ExecutionPolicy Bypass -File scripts/build_viewer.ps1 -Configuration Release
```

回滚点：两个 slot分别完成 reference/neural PT/deferred capture；Falcor worktree恢复干净。

## Phase 6：formal preflight、200k冻结运行与报告

- [x] 用实际 formal tensor尺寸做显存/吞吐 preflight，确认 flat dispatch tile只影响执行、不改变 logical recipe；冻结 stage boundary与所有 author-underspecified选择。
- [x] 运行短程 MaterialX spatial和 LayerStack 1×1 live smoke，检查有限 loss/grad、route独立、checkpoint resume和ETA。
- [x] 按冻结的 300k recipe 启动 MaterialX spatial formal run；用户在慢收敛区间决定停止并以已验证的 step 200k checkpoint 登记。保留周期 checkpoint/validation/metrics，报告明确不声称完成 300k。
- [x] 从 step 200k checkpoint 导出 final ScatteringPackage；运行 SlangPy/Falcor/viewer PT/deferred parity和定量/视觉评测。
- [x] 在 `artifacts/` 写训练摘要、质量—时间—显存—package—shader cost报告；在稳定文档只登记可追溯结论与 artifact/package identities。

验证命令由 frozen formal config确定，至少包含：

```powershell
conda run -n neural-shading python -m ncls learn train configs/learning/nvidia-rta2024-materialx-formal.json artifacts/nvidia-faithful/materialx-formal-300k/checkpoint.pt
conda run -n neural-shading python -m ncls learn evaluate configs/learning/nvidia-rta2024-materialx-formal.json <checkpoint.pt> --batches 1
conda run -n neural-shading python -m ncls learn export <checkpoint.pt> <source.mtlx> <package-dir>
```

回滚点：formal产物以 identity/checkpoint可恢复；observed quality不反向修改 recipe。

## Phase 7：全量质量门、归档与提交

- [x] 运行 scoped unit/GPU后再跑全量 pytest；Release viewer/headless dual-slot capture；package tamper matrix；`git diff --check`。
- [x] 检查 `external/Falcor` 与其他锁定上游全部干净；检查根仓库无被误加入的 `artifacts/data/build/reports`。
- [x] 使用 `trellis-check`做 spec/cross-layer/data-flow/reuse/一致性审查并修复问题。
- [ ] 更新 correspondence/experiment log/任务 checklist，运行 `trellis-finish-work`，归档任务。
- [ ] 只 stage 本任务识别文件，按逻辑批次创建本地 commit；不 amend、不 push、不夹带用户原有 dirty files。

最终质量命令：

```powershell
conda run -n neural-shading python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/build_viewer.ps1 -Configuration Release
git diff --check
git -C external/Falcor status --short
```
