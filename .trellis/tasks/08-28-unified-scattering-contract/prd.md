# 统一材质散射接口与 estimator 审计

## Goal

让所有 source reference 与 neural material 都通过同一套干净的 scattering 包装进入 renderer；每个可用于路径追踪的材质实现必须拥有语义一致、彼此 matched 的 `evaluate()`、`sample()` 与 `pdf()`，主路径追踪器不得按材质族拼装 estimator，也不得用静默 generic proposal 冒充材质自身的采样合同。用户得到的直接价值是：reference 既保留各 source 的权威语义，又能在路径追踪中稳定收敛，后续 neural 对比不会混入 renderer 特判或错误 PDF 带来的噪声。

## Background

- MDL car paint 与 ceramic 的不收敛白点已经定位为 estimator 不匹配：MDL evaluator 含尖锐 flakes/coat，而旧 viewer 使用固定 roughness 的 generic cosine/GGX proposal 与 PDF。MDL target code 实际已经提供 matched `surface_scattering_sample/evaluate/pdf`；修复与证据见归档任务 `08-27-mdl-reference-viewer/research/firefly-bug-analysis.md`。
- 公共 `INclsScatteringBackend` 已经规定 `prepare → evaluate/sample/pdf`（`shaders/ncls/contracts/scattering_backend.slang:8-28`），neural `PackagePathTracer` 也只沿该合同调用（`apps/viewer/shaders/PackagePathTracer.cs.slang:215,286,370,377`）。
- source `ReferencePathTracer` 仍在 `apps/viewer/shaders/ReferencePathTracer.cs.slang:243-639` 按 family 拼装 evaluator/sampler/PDF。MERL、MaterialX 仍会落到 generic proposal，LayerStack 多界面返回 0 PDF 并关闭环境 NEE；完整审计见 `research/interface-audit.md`。
- 正式 MERL、MaterialX、LayerStack reference backend 虽然声明了三件套，当前实现仍使用纯 cosine proposal（`shaders/ncls/reference_backends/{merl,materialx,layer_stack}.slang`），因此“接口形状存在”还没有形成严格架构保证。

## Requirements

- R1：公共 shader 合同只保留现有 `INclsScatteringBackend` / backend-specific `ScatteringState`；source 与 neural renderer 都只消费 `prepare/evaluate/sample/pdf`，路径积分器不识别 source family。
- R1a：这是根本性迁移而不是兼容包装：五个 source-private module 必须直接实现正式 scattering contract；删除旧 viewer-only evaluate/sample/pdf 函数族、双轨调用与 fallback，不允许让新接口转调一套继续存活的 legacy estimator。
- R2：方向采样结果必须携带与 `evaluate` 和 `pdf` 属于同一 estimator 的方向、throughput weight、forward/reverse PDF、event 与有效性；直接光 MIS 与后续 bounce 使用同一个 PDF 定义。
- R3：删除 reference 主路径中的 generic proposal/PDF fallback、family-specific estimator assembly、LayerStack 多界面 0-PDF 特例及对应环境 NEE 禁用。缺少完整三件套时必须 fail closed，不能静默进入 PT。
- R4：逐一实现 LayerStack、OpenPBR、MERL、MaterialX、MDL 与现有 neural package 的合同一致性；五个现有 source family 都必须达到完整 PT capability，不把任何一个降级为 optional/unsupported。
- R5：保留各 source 的原生语义、资源与 typed state，不把它们归约为 LayerStack 或共同 closure 参数布局。没有原生 sampler 的 family 可以拥有 source-private 派生 proposal，但它不得改变 evaluator 或 source identity。
- R6：公共 response measure 必须对 reflection/transmission 使用 `f * abs(dot(Ns, wi))`，与 `reference_dataset_v5` 的 absolute-cosine 语义一致。
- R7：增加 estimator 一致性、PDF 支持域/归一性、finite/event、sample-weight 与尖锐高光 tail 回归；测试不能只检查 `finite`，也不能通过 clamp 掩盖高权重。
- R8：viewer UI、capture/replay、source identity/hash、两个 slot 的失败隔离与现有资产编辑行为保持兼容；不得修改锁定上游仓库。
- R9：公共接口、query-plane 与 scattering capability 的边界、各材质族 sampler 状态写入稳定中文规范；删除被新合同替代的死路径与兼容分支。

## Acceptance Criteria

- [x] AC1【需求交付｜来源：用户“所有方法在完全相同接口包装下”】：source reference 与 neural material 的 PT 调用都经 `INclsScatteringBackend` 包装；路径积分器中不存在按 LayerStack/OpenPBR/MERL/MaterialX/MDL 选择 evaluate/sample/PDF 的分支。
- [x] AC2【理论/语义正确性｜来源：公共 scattering 合同与 Monte Carlo estimator 不变量】：五个现有 source backend 与 neural package 都提供 matched `evaluate/sample/pdf`；连续 sample 的实际分布、报告 PDF、weight 与环境 MIS 一致，delta/null 显式处理。
- [x] AC3【需求交付｜来源：用户要求 strict architecture】：旧 generic fallback、LayerStack 0-PDF/MIS 禁用和 renderer-side family estimator assembly 全部删除；任一 backend 缺 capability、编译或资源失败均 fail closed。
- [x] AC4【数值实现正确性｜来源：float32 独立 oracle、PDF 数值积分与统计置信度设计】：LayerStack、MERL、MaterialX、MDL 与 neural package 通过 `evaluate.pdf↔pdf`、`sample.pdf↔pdf(sample.wi)`、适用的 `weight↔f·absCos/pdf`、support/null mass、event 与有限性 GPU 检查；OpenPBR 通过 official native sample tuple identity、independent `evaluate.pdf↔pdf`、稳定方向域逐点检查与 logarithmic grazing/capture tail 回归。协议修正证据登记在 `research/validation-protocol.md`，不能用 independent query 重建极窄掠射 native sample。
- [x] AC5【需求交付｜来源：用户报告的现场缺陷】：MDL car paint/ceramic 1024 spp 继续无随 spp 增长的随机孤立白点；MERL 尖锐 measured BRDF、MaterialX 低 roughness/anisotropy、LayerStack 多界面与 OpenPBR coat/transmission 产生对应 tail/capture 证据，若出现同类不收敛不得以 clamp 通过。
- [x] AC6【理论/语义正确性｜来源：`reference_dataset_v5` response measure 与 transmission transport】：Python/Slang response adapter 使用 absolute light cosine，reflection/transmission regression 均通过且不重复乘余弦。
- [x] AC7【需求交付｜来源：现有 viewer 合同】：reference/neural 对比、capture/replay、source identity/hash、slot 失败隔离与 UI 行为保持兼容，Release viewer 构建通过。
- [x] AC8【工程约束｜来源：根 `AGENTS.md`】：相关 unit/GPU/integration 验证完成；六个锁定 `external/` worktree 保持干净；运行产物只写 `artifacts/`，临时诊断只写本任务 `scratch/`。
- [x] AC9【需求交付｜来源：用户“保持统一接口高度干净”】：`.trellis/spec/` 与正式中文文档同步，query-plane/scattering capability 不再混淆，旧符号、死代码与兼容入口扫描无残留。
- [x] AC10【需求交付｜来源：用户“根本性迁移/接口适配，而不是兼容层”】：LayerStack、MERL、OpenPBR、MaterialX、MDL 的 canonical shader implementation 直接实现 `INclsScatteringBackend`；viewer 与 formal runtime 复用这些实现，不存在转调旧 reference path 的 compatibility shim。

## Out of Scope

- 不在本任务中改变 neural evaluator 的研究形态、训练目标或质量结论。
- 不把任一 source 的原生参数、图结构或资源翻译成统一 closure/LayerStack 后再作为 GT。
- 不修改锁定第三方源码；需要对照时只读取 Falcor/MDL SDK/MaterialX/OpenPBR 等上游实现。
- 不以 radiance/throughput clamp 掩盖 estimator 不匹配。
- 不扩展当前 MaterialX evaluator 子集到新的 closure，也不改变 OpenPBR/MDL/LayerStack/MERL 的物理定义；本任务只统一执行合同与 source-private sampling。
- 不用本次 observed max/p99 值创建新的任意 hard gate；除数学/数值正确性外，tail 数值用于诊断与相对证据。
