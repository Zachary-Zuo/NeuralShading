# PT 细碎椒盐噪点视觉诊断与修复

## Goal

在不裁剪 radiance/throughput、不加 denoiser、也不破坏各材质 `prepare/evaluate/sample/pdf` 原生语义的前提下，找出并修复 viewer 公共 path tracer 在 glossy 材质与复杂 shaderball 几何上残留的细碎亮 firefly。修复必须同时覆盖 source reference 与 neural package PT，并保留合法的连续高光、car-paint flakes 与 HDR 能量。

## 已冻结的现场事实

- 同一相机、环境、曝光与 1024 spp 下，960×540 composite（单 slot 480×540）的 MDL car paint、MDL ceramic 与 OpenPBR car paint 都能看到单像素或极小簇的正向亮离群点。
- 亮点主要集中在 shaderball 底座的掠射、遮挡、凹槽和接触区；球体主体明显更稳定。
- ceramic 不含 car-paint flake closure，却出现相同空间签名，因此该现象不是 flakes texture。
- MDL ceramic 的 `max_bounces=0` 图像基本干净；开放第一条 BSDF continuation ray（`max_bounces=1`）后亮点立即出现。首要调查边界因此是公共的 `continuation → environment miss / secondary-hit NEE` 段，而不是 primary-hit environment NEE。
- 上一任务已经修复 OpenPBR native sample tuple 被 independent PDF 重建放大的缺陷，并把 linear EXR 固定为 float32；本任务不得回退这些合同，也不能用旧的 `>100` 单阈值替代视觉与路径归因。

## Requirements

- R1：先保存可复现的视觉基线。固定现场相机、环境、曝光、分辨率和 1024 spp，至少覆盖 MDL car paint、MDL ceramic、OpenPBR car paint，以及 ceramic 的 bounce 0/1 对照。
- R2：把第一条 continuation 之后的贡献拆成 `BSDF-hit environment`、`secondary-hit environment NEE` 与更深路径，并把离群点与 `Ns/Ng` 偏差、几何半球有效性、BSDF/light PDF、MIS weight 和 throughput 尾部关联；没有路径证据前不选择修法。
- R3：若确认 shading normal / geometric normal 域不一致，修复必须位于公共 path-surface / transport 边界，并对所有 backend 一致生效；不得在 MDL、OpenPBR 或材质 ID 上分支。
- R4：若确认剩余离群值来自几何有效的 environment direct estimator 方差，采用保持无偏、固定运行上界的多样本 MIS；direct-environment estimator 与 indirect continuation 必须显式分工，不能双计 environment radiance。
- R5：source-native `sample()` 返回的 direction/event/PDF/weight tuple 保持不可拆分；公共 transport 可以拒绝几何上不可能的 event，但不得重算或夹紧已接受 sample 的 weight/PDF。
- R6：`ReferencePathTracer` 与 `PackagePathTracer` 使用同一份环境采样、MIS、shading-normal 与几何半球规则；材质仍只通过 canonical state 接口进入 renderer。
- R7：修复后审计 MDL car paint/ceramic、OpenPBR car paint/aluminum/glass、MERL chrome、MaterialX 与 LayerStack，区分合法连续高光、材质纹理结构和随机孤立亮点。

## Out of Scope

- 不引入 radiance/throughput clamp、firefly filter、temporal/spatial denoiser 或后处理遮掩。
- 不通过提高 capture 的 1024 spp 门槛宣布修复。
- 不修改 MDL SDK、Falcor、OpenPBR 等 `external/` 上游源码。
- 不改变源材质参数、纹理、图结构或 reference identity。
- 不把本任务扩大为通用 adaptive sampling、path guiding、ReSTIR 或生产级降噪系统。

## Acceptance Criteria

- [ ] 【需求交付；来源：用户现场要求】在同一冻结 replay、960×540 composite、1024 spp 下给出 before/after 视觉板；MDL car paint 与 ceramic 不再出现当前这种随第一条 continuation 引入的细碎随机亮点，同时合法高光、tile 图案与 car-paint flakes 保留。
- [ ] 【理论正确性；来源：公共 transport/散射合同】每个接受的 reflection/transmission sample 在 shading 与 geometric hemisphere 上与 event 一致；不一致 sample 作为显式 null path 终止，ray origin 仍按实际方向选择 geometric-normal 偏移侧。
- [ ] 【理论正确性；来源：multiple-sample MIS】若启用 direct-BSDF 多样本分支，light/BSDF 两侧使用同一 `n_light·p_light`、`n_bsdf·p_bsdf` power heuristic；direct environment 与 indirect continuation 不双计、不漏计。
- [ ] 【架构正确性；来源：用户要求与 unified pipeline spec】source/neural renderer 均只调用 `state.prepare/evaluate/sample/pdf`，无 source-family estimator 分支、generic fallback、native tuple 重建或 viewer-only backend shim。
- [ ] 【数值正确性；来源：float32 与 sample contract】相关 GPU probe 全 finite；既有 native tuple identity、independent evaluate↔pdf、project-owned sample/PDF/weight 与 response-measure 测试继续通过。
- [ ] 【诊断正确性；来源：本任务冻结设计】临时 contribution AOV 的分量和与 beauty 在冻结 float32 容差内一致；正式实现不保留会污染公共 scattering ABI 的诊断字段。
- [ ] 【回归交付；来源：用户追问“其它材质”】完成 R7 材质矩阵 capture 并以视觉分类和 report-only tail/RSE 报告结果；不从 observed 数值反推新的 hard gate。
- [ ] 【工程质量；来源：项目规范】相关 unit、Falcor/D3D12 GPU、integration 与 Release viewer build 通过；六个锁定 upstream 工作树保持 clean。

## Notes

- 视觉与 bounce 隔离证据记录在 `research/visual-diagnosis.md`；单次统计与截图保存在 `artifacts/diagnostics/pt-salt-pepper-noise/`。
- 当前 leading hypothesis 是公共 shading-normal / geometric-normal transport 域缺口；几何有效的 one-bounce environment 方差是竞争假设。实施先用 contribution 与 normal AOV 判别，不按观感直接选其中一个。
