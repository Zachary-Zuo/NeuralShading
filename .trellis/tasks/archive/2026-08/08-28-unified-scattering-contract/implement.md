# 实施计划：统一材质散射接口与 estimator

## 执行原则

- 只有用户批准本任务最新 planning summary 后才运行 `task.py start`。
- Codex inline 模式由主会话直接实现和检查，不创建 implement/check sub-agent，也不需要 JSONL context gate。
- 所有源码编辑使用 `apply_patch`；不触碰当前工作树中用户已有的 `.trellis/config.yaml`、旧任务 scratch、字体、截图与 `tmp/`。
- 不使用 clamp、generic fallback 或 capability 降级换取测试通过。
- 不建立 compatibility shim：迁移一个 family 时同步切换 formal/viewer consumer 并删除旧实现；中间态只存在于未提交工作树，最终仓库不保留双轨。

## Phase A：激活与基线

- [x] A1. 用户批准后运行 `task.py start`，加载 Phase 2.1 细则和 `trellis-before-dev`。
- [x] A2. 按 `.trellis/spec/project/dev-environment.md` 串行执行 G/E/FW/W 探针；第一次涉及验证时报告环境分类与证据。
- [x] A3. 记录任务起点 `git status --short`、相关现有 unit/GPU smoke 与六个 upstream status；不把无关 dirty files纳入 task diff。
- [x] A4. 在正式 GPU tolerance/tail run 前，把点对点 float32 tolerance、统计 seed/sample count/置信区间写入测试或 `research/validation-protocol.md`。

## Phase B：冻结公共合同

- [x] B1. 在 Python core 增加 `REQUIRED_PATH_TRACING_CAPABILITIES`，让所有 `ReferenceProgramDescriptor` fail closed 校验 `Prepare|Evaluate|Sample|Pdf`。
- [x] B2. 把 Python/Slang response measure 从 positive cosine 改为 absolute cosine，更新所有调用、导出、测试与中文合同；删除旧符号，不留 alias。
- [x] B3. 增加 reflection/transmission response regression，确认 evaluator 只输出 `f`，余弦恰好乘一次。
- [x] B4. 运行最小 unit gate；若发现既有 schema/ABI 依赖 positive cosine，回到 planning 记录语义影响，不能双轨兼容。

## Phase C：source-private proposal 与 backend

- [x] C1. 新增 bounded cosine + rotated anisotropic multi-GGX mixture primitive；实现 exact component selection、mixture PDF、null event 与 reverse-PDF helper。
- [x] C2. LayerStack：单界面复用 native interface sampler；多界面用层参数构造 mixture，sample 方向后调用 random-walk evaluate，删除 0-PDF 结果。
- [x] C3. MERL：用固定次数的 measured-table peak/off-peak probes 构造 view-conditioned multi-scale proposal；保持 table evaluator 不变。
- [x] C4. MaterialX：用 resolved inputs 构造 diffuse + rotated anisotropic GGX proposal；normal/UV-footprint 解析归入 prepare。
- [x] C5. OpenPBR：保持 `openpbr_eval/sample/pdf` 原生语义；sample 原样保留 official direction/event/PDF/weight tuple，independent query 使用 official eval/pdf，并统一 event/reverse PDF/working-space 包装。
- [x] C6. 每完成一个 family，先运行该 backend 的 GPU sample/pdf/evaluate probe；任何不一致在进入 composite viewer 前修复。

## Phase D：MDL 完整 runtime contract

- [x] D1. 建立 viewer/formal 共用、直接实现 `INclsScatteringBackend` 的 canonical MDL backend；generated target code仍是唯一 DF 实现，迁移后删除 `MdlViewerAdapter.slang` viewer-only API。
- [x] D2. 包装 forward/reverse PDF、delta/transmission event 与 `bsdf_over_pdf`，并让 formal reference runtime capability完整声明。
- [x] D3. 更新 MDL provider/viewer 动态 source composition 与 implementation identity；保持 Falcor2 只作隔离 oracle。
- [x] D4. 扩展现有 MDL GPU probe，验证 common state、formal query 与 SDK target code 的点对点一致性。

## Phase E：viewer 统一接线

- [x] E1. 新增 `apps/viewer/shaders/SceneReferenceProgram.slang`，把五个 canonical concrete backend 组合成 heterogeneous sum type；family switch 只表达 sum dispatch，不转调任何 legacy estimator。
- [x] E2. 重写 `ReferencePathTracer.cs.slang`：命中后只构造 `NclsScatteringContext`，调用 `backend.prepare`、`state.evaluate/pdf/sample`；移除 `NclsReferenceSample`、`nclsEvalReferencePath`、`nclsSampleReferencePath`、`nclsReferencePdfPath`、`nclsReflectionProposalPdf` 与 family-specific light gates。
- [x] E3. 恢复 LayerStack 多界面环境 NEE/MIS；实际 sampled direction 相对 geometric normal 的符号决定 ray origin side，renderer 不读取 family。
- [x] E4. 更新 CMake/overlay dependency 列表和必要 C++ resource binding；`NCLS_REFERENCE_FAMILY_MASK` 只保留静态裁剪用途。
- [x] E5. 新增静态架构测试，明确禁止上述旧符号、`MdlViewerAdapter`、compatibility/fallback shim 与 renderer-side `surface.family` estimator 分支，并要求 source/package PT 出现同一四个合同调用。

## Phase F：验证与缺陷审计

- [x] F1. Unit：core contract、reference program descriptors、viewer static route、source identity/capture/replay/slot failure。
- [x] F2. GPU contract：generic mixture、LayerStack 单/多界面、MERL、MaterialX、OpenPBR、MDL、neural package。
- [x] F3. 代表性 headless 1024 spp：MDL car paint/ceramic、MERL chrome、MaterialX low-roughness/texture、LayerStack multilayer、OpenPBR brushed metal/car paint/glass transmission。产物写 `artifacts/captures/unified-scattering-contract/`，最终线性 EXR 为 float32 channel。
- [x] F4. 用任务 `scratch/` 中的分析器输出 max/high quantile/local-isolation 与随 spp 变化；数学门按冻结协议判断，tail 数值作为诊断和视觉证据。
- [x] F5. 运行相关 integration/full unit，随后唯一允许的 `scripts/build_viewer.ps1 -Configuration Release`。
- [x] F6. 扫描 generic fallback、旧符号、`MdlViewerAdapter`、compatibility shim、重复 cosine adapter、family estimator branch 和死代码；检查六个 upstream clean。

## Phase G：文档、质量门与收尾

- [x] G1. 用 `trellis-update-spec` 更新 core/viewer/guides 稳定规范：唯一 scattering contract、absolute cosine、source-private proposal、capability fail-closed、tail 检查。
- [x] G2. 更新正式中文文档与 viewer README，删除“MDL sample/pdf 只是 viewer-internal、公共 capability 仍 evaluate-only”的旧边界。
- [x] G3. 用 `trellis-break-loop` 复核这次从 MDL 单点缺陷提升为架构防线后是否仍有同类入口。
- [x] G4. 用 `trellis-check` 做最终 spec compliance、tests、build、diff、dead-path 与 upstream-clean 检查。
- [x] G5. 仅提交本任务 scoped files；不 amend、不 push。记录 commit hash、验证矩阵和未纳入的用户 dirty paths。
- [x] G6. 质量门通过后归档 `08-28-unified-scattering-contract` 并记录 journal。

## 计划验证命令

环境探针（串行）：

```powershell
nvidia-smi --query-gpu=name --format=csv,noheader
conda env list
Test-Path external\Falcor\build\windows-vs2022\bin\Release\python\falcor\falcor_ext.cp310-win_amd64.pyd
```

Unit 与静态合同：

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_scattering_contract.py tests/unit/test_viewer_slots.py tests/unit/test_mdl_reference_boundary.py tests/unit/test_reference_evaluator_lifecycle.py -q
```

Falcor/D3D12 GPU（新增测试名在实现时固定）：

```powershell
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu/test_scattering_contract_gpu.py tests/gpu/test_reference_scattering_backends.py tests/gpu/test_openpbr_reference_gpu.py tests/gpu/test_merl_reference_gpu.py tests/gpu/test_mdl_hlsl_feasibility.py -q
```

全量 unit、Release 与 clean：

```powershell
conda run -n neural-shading python -m pytest tests/unit -q
.\scripts\build_viewer.ps1 -Configuration Release
git -C external/Falcor status --short
git -C external/MaterialX status --short
git -C external/OpenPBR status --short
git -C external/openpbr-bsdf status --short
git -C external/pbrt-v4 status --short
git -C external/glm status --short
git diff --check
```

headless capture 使用版本化 replay，经 Release `NclsViewer.exe --replay ... --headless --capture ...` 执行；replay 与输出写入 `artifacts/`，不进入根仓库。

## 高风险文件与 rollback point

| 风险 | 文件/区域 | rollback point |
|---|---|---|
| response measure 改变 transmission 语义 | `src/ncls/core/scattering/*`、`shaders/ncls/contracts/response_measure.slang` | B4；若数据/ABI 证据冲突，停止并回 planning，不留双入口 |
| source sampler 与 PDF 漂移 | `shaders/ncls/reference_backends/*`、新 proposal primitive | 每个 family 的 C6 GPU probe；未通过不接 viewer |
| MDL dynamic module 编译边界 | `MdlViewerAdapter.slang`、`mdl_query.slang`、MDL Python/C++ composition | D4；保留上一已提交 MDL viewer 实现作为可恢复基线 |
| heterogeneous source state register/resource 压力 | 新 `SceneReferenceProgram.slang`、`ReferencePathTracer` | E5 后做真实 mixed-family compile/capture；超预算时优化 composer 的 resource-free storage，不把分支放回 integrator |
| Falcor overlay 污染 | `apps/viewer/CMakeLists.txt`、overlay patch | 只经 `build_viewer.ps1`；失败后立即检查并恢复 upstream clean |

## 启动前复核

- [x] `prd.md`、`design.md`、`implement.md` 已从头到尾复读，需求与验收无重复冲突。
- [x] 无 blocking open question；五个现有 source family 均在范围内。
- [x] 用户已在最新 planning summary 之后明确批准实施，并补充“根本性迁移/接口适配，不要兼容层”。
