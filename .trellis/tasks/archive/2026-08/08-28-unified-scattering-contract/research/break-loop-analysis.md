# Bug Analysis：统一接口存在但 source renderer 绕开 estimator owner

## 1. Root Cause Category

- **Category**：B - Cross-Layer Contract，同时伴随 D - Test Coverage Gap 与 E - Implicit Assumption。
- **Specific Cause**：项目已经定义 `INclsScatteringBackend.prepare → State.evaluate/sample/pdf`，但旧 `ReferencePathTracer` 仍按 source family 分别拼装 evaluator、generic proposal、PDF 与 MIS gate。接口形状存在，却没有 descriptor fail-closed、renderer 静态边界和跨入口数值恒等式来保证真实调用图遵守它。旧实现还隐含假设 reflection-only 的正余弦、event label 能决定 ray-origin side，以及 `finite` 足以证明尖锐材质收敛。

### Bayesian diagnosis

| Hypothesis | 初始 prior | 判别证据 | 最终判断 |
|---|---:|---|---|
| H1：evaluate/sample/pdf estimator 错配 | 50% | 旧 reference integrator 明确含 family branch、固定 GGX/cosine fallback 与 LayerStack `pdf=0`；MDL target code 自带三件套 | 对旧架构为高置信根因；通过 canonical migration 消除 |
| H2：correct estimator 下的真实 HDR 窄高光与有限样本方差 | 30% | 同一 target-code sample/pdf 通过 GPU 恒等式后，高值仍只集中在 HDR 环境反射位置；4-sample NEE 在不 clamp 的情况下把 MDL mean RSE 降低约 27–31% | 对迁移后残余“白亮点”为高置信解释 |
| H3：NaN、错误 ray offset 或自相交漏光 | 20% | 所有 EXR/contract query finite；direct-only/one-bounce 判别显示主峰来自环境 bounce。event-driven offset 仍是结构风险，但没有证据证明它是本次主峰来源 | NaN 假设被排除；ray-origin 风险预防性修复，不宣称是现场主因 |

判别证据优先级为 GPU 数值恒等式与 raw EXR 空间结构，其次是 capture manifest/RSE，再次才是显示图观感。最终结论不把所有高值都归为 bug：随机分散且 sample/PDF 不匹配的是 estimator 缺陷；与 HDR 光源映射一致、随 spp 稳定成片的是正确信号但可能方差较高。

## 2. Why Fixes Failed

1. **MDL viewer adapter 单点修复**：让 MDL viewer 调用 target-code sample/pdf，修复了一个 family 的 estimator，但保留 viewer-only API、runtime/query capability 混淆，以及其他 source family 的 generic proposal；属于 incomplete scope。
2. **只检查 finite 或提高 spp**：NaN/Inf 检查发现不了极端但有限的 `f·cos/pdf`；提高 spp 会让真实 HDR 高光逐步显现，也可能让错误长尾产生更多白点，因此没有判别力。
3. **把高值统一当 firefly**：没有先区分空间聚集、direct/one-bounce 来源与 environment peak，容易用 clamp 删除真实 signal；属于 mental model anchoring。
4. **把上游三入口强制改成同一 float32 数值路径**：OpenPBR 的 official sample/pdf 数学上属于同一 proposal，但极窄 coat 在掠射反射时，sampler 内部仍持有高精度 half-vector，returned `wi` 已经发生方向舍入。用该 `wi` 再调 `openpbr_pdf()` 会因切向量相消把约 `10^9` 的 sample-path PDF 错算成约 `10^-8`；用 independent eval/PDF 重建 weight 随即从 native `10^-2–10^-1` 爆到 `10^14`。因此正确边界是保留不可拆分的 native sample tuple，并单独验证 independent evaluate/pdf；“canonicalize tuple”本身是本轮发现并删除的 implementation defect。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | 五个 source backend 直接实现 canonical state；`SceneReferenceProgram` 只做 sum dispatch，integrator 不识别 family | DONE |
| P0 | Compile-time / descriptor | `ReferenceProgramDescriptor` 缺 `PREPARE/EVALUATE/SAMPLE/PDF` 任一项即拒绝 | DONE |
| P0 | Test Coverage | GPU 验证 sample→pdf、weight 恒等式、event/finite 与 mixture 概率质量；unit 扫描旧符号和 renderer family branch | DONE |
| P0 | Native API tuple identity | OpenPBR sample 原样发布 official direction/event/PDF/weight；independent evaluate/pdf 仍由同一 prepared state 提供，追加 car paint logarithmic grazing 与 float32 capture 回归 | DONE |
| P0 | Error semantics | invalid/null sample 终止当前 path；不允许 generic fallback 或 radiance/throughput clamp | DONE |
| P1 | Variance control | source/package PT 都使用 4-sample environment NEE；MIS 两侧统一 `4·p_light` | DONE |
| P1 | Geometry robustness | ray origin 根据 actual direction 选 geometric-normal side，不依赖 event label | DONE |
| P1 | Documentation | core/viewer/data spec 分离 runtime scattering 与 provider query capability，并加入 tail 诊断清单 | DONE |

## 4. Systematic Expansion

- **Similar Issues**：neural package 的 host ABI 虽然已经干净，仍需持续保留静态调用扫描；新增 source family、deferred sampling 扩展和 environment sampler 数量变更都可能重新产生同类错配。
- **Design Improvement**：统一接口的强度来自“唯一调用路径 + fail-closed + 数学恒等式”，不是来自结构体命名。heterogeneous source 只允许在 composer 内做类型分派，所有散射数学留在 concrete backend。
- **Process Improvement**：现场白点先建立 estimator mismatch、HDR signal、几何漏光三个竞争假设；用 sample/PDF oracle、bounce 隔离、空间聚集与 RSE 选择判别证据，不从单张 tone-mapped 图下结论。

## 5. Knowledge Capture

- [x] 更新 `.trellis/spec/core/shared-slang-backend.md` 的完整 scattering 合同与错误矩阵。
- [x] 更新 `.trellis/spec/viewer/conventions.md`、`mdl-reference.md` 与 `path-surface.md`。
- [x] 更新 `.trellis/spec/data/mdl-reference.md`，分离 query/runtime capability plane。
- [x] 更新 `.trellis/spec/guides/cross-layer-thinking-guide.md` 与 index。
- [x] 新增 unit/GPU/static/capture 证据；不另建后续 issue，因为本任务已完成 root migration。
- [x] 项目没有 `src/templates/markdown/spec/`，不存在可同步的模板副本。
