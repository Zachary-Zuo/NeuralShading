# Source / neural scattering interface 审计

## 结论

项目已经有正确的公共形状：`INclsScatteringBackend.prepare()` 返回实现 `evaluate/sample/pdf` 的 `INclsScatteringState`（`shaders/ncls/contracts/scattering_backend.slang:8-28`）。neural `PackagePathTracer` 也确实只通过该合同调用 backend（`apps/viewer/shaders/PackagePathTracer.cs.slang:215,286,370,377`）。

架构缺口在 source viewer：`ReferencePathTracer` 自己按 family 选择 evaluator、sampler 与 MIS PDF，并保留 `nclsReflectionProposalPdf()` generic fallback（`apps/viewer/shaders/ReferencePathTracer.cs.slang:243-639`）。因此“存在公共接口”没有自动保证 source renderer 使用它。

## 当前能力矩阵

| backend | evaluate | sample / pdf | 当前风险 | 规划结论 |
|---|---|---|---|---|
| neural package | backend state 自有 | NVIDIA proposal 的 sample 与 mixture PDF 配对 | 主调用路径干净 | 保留，加入跨 source 共用合同回归 |
| OpenPBR | `openpbr_eval` | `openpbr_sample/openpbr_pdf`（`shaders/ncls/reference_backends/openpbr.slang:22-54`） | viewer 仍有 family branch；公共 response cosine 对 transmission 使用正半球余弦 | 保留原生 sampler，迁入统一 source backend；修正 absolute cosine |
| MDL | generated target code evaluate | viewer 已使用同一 target code sample/pdf | viewer 数值已修复，但 `ReferenceProgramDescriptor` 仍只声明 prepare/evaluate（`src/ncls/references/programs/mdl.py:28`） | 复用 target-code transport，补全正式 runtime capability 与统一 state 包装 |
| MaterialX | 锁定 1.39.4 `standard_surface` 子集 | 正式 backend 仍是纯 cosine（`shaders/ncls/reference_backends/materialx.slang:30-53`）；viewer 是 cosine + 主 GGX 的内联 proposal | anisotropy rotation、lobe 权重和 estimator 所有权分散；低 roughness 有长尾风险 | 实现 resolved-input 驱动的 diffuse + rotated anisotropic GGX mixture，sample/pdf 完全同式 |
| MERL | 原始测量表 | 正式 backend 仍是纯 cosine（`shaders/ncls/reference_backends/merl.slang:21-44`）；viewer 是固定 roughness 0.2 generic mixture | 与 MDL 同类的尖锐高光长尾风险，尤其 chrome/specular phenolic | 实现由当前 `wo` 与测量表响应校准的有界 multi-scale GGX + cosine proposal |
| LayerStack | 随机游走 reference | 正式 backend 仍是纯 cosine（`shaders/ncls/reference_backends/layer_stack.slang:24-51`）；viewer 单界面有 matched interface sampler，多界面 analog walk 返回 `pdf=0` | 多界面关闭环境 NEE/MIS，三件套语义不完整；窄层间峰可能有长尾 | 单界面保留原生 sampler；多界面改用由层参数构造的有界 lobe mixture，并用随机游走 evaluate 形成 `f·|cos|/pdf` |

## 额外合同缺口

- `NclsScatteringEval.f` 定义为纯 BSDF，但公共 response adapter 使用 `max(dot(n, wi), 0)`（`shaders/ncls/contracts/response_measure.slang:8-19`）；数据 schema 与 transmissive BSDF 需要 `abs(dot(n, wi))`。这会让 OpenPBR/MDL 的 transmitted direct-light response 被错误清零。
- heterogeneous scene 仍需要 host 绑定不同 family 的资源，并以 `NCLS_REFERENCE_FAMILY_MASK` 做 shader specialization（`apps/viewer/NclsViewer.cpp:739-777`）。这是资源发现/编译裁剪，不应进入 estimator。允许 family dispatch 只存在于一个 source backend adapter；路径积分器本身不得识别 family。
- `evaluate().pdf`、独立 `pdf()` 与 `sample().pdf` 目前没有覆盖五个 source family 的共同 GPU contract test；existing finite smoke 无法发现极端但有限的 `f·|cos|/pdf`。

## 数学边界

- “matched”在本任务中首先指 estimator 一致：`sample` 的实际方向分布必须与 `pdf` 完全相同，连续事件的 `weight` 必须等于同一次 source `evaluate` 所代表的 `f·|cos|/pdf`；直接光 MIS 使用同一个 `pdf`。
- proposal 不要求等于 BSDF 的归一化分布。没有原生 sampler 的 measured/stochastic reference 可以拥有派生的重要性 proposal，但 proposal 的构造、sample 与 PDF 必须由该 source backend 私有实现，并为整个支持域保留非零概率。
- delta 事件允许 solid-angle PDF 为 0，但必须设置 `Delta` event 并直接使用 source sampler 给出的有限 throughput weight；不能把 delta 当连续事件除以 epsilon PDF。
- LayerStack 的 directional evaluate 是 stochastic。其 sample-weight 单次相等性需要冻结同一 evaluator RNG stream；一般回归同时检查同流等式与跨流统计一致性，不能把 Monte Carlo 差异误判成接口漂移。

## 历史决策接续

`trellis mem search "统一 scattering evaluate sample pdf"` 找到 2026-08-23 会话 `01a02ea9-7c5`：已确认目标调用形状是 `prepare → evaluate/sample/pdf`，且 sampler 与 PDF 是独立但配对的方向分布。当前任务不重新讨论这一产品边界，只把尚未遵守它的 source viewer 与 reference backends 收敛到同一合同。
