# Bug Analysis：MDL car paint / glazed ceramic firefly 不收敛

## 1. Root Cause Category

- **Category**：B（Cross-Layer Contract）+ D（Test Coverage Gap）+ E（Implicit Assumption）。
- **Specific Cause**：MDL bridge 已在同一 `surface.scattering` target function 中生成 matched `sample/evaluate/pdf`，但 viewer 只接入 evaluate，随后用固定 roughness `0.2` 的 cosine/GGX mixture 采样所有 MDL closure，并把该 PDF 用于环境光 MIS。flakes 与 glazed coat 的窄峰和这个 proposal 不匹配，导致极端 `response / pdf` 权重持续进入累计器。
- **Bayesian evidence**：初始先验为 proposal/PDF 错配 50%、纹理解码坏点 25%、累计公式错误 25%。累计器是标准加权均值且同一白点位置随 rare sample 出现，降低累计错误概率；52 万正式 GPU query 中 generic 最大权重为 `3747/7018`，同方向 MDL PDF 权重为 `75/17`，再加上 matched capture 消除空间孤立点，使 proposal/PDF 错配置信度超过 99%。

## 2. Why Fixes Failed

1. 原始 finite smoke：只断言 EXR 全 finite、1024 spp 与 identity；`4112` 仍是有限数，因此错误 estimator 被误判为可运行。
2. 原始视觉验收：只检查三种材质“看起来不同”，没有检查尖锐 closure 的空间孤立点和权重尾部。
3. 第一次真实 scene 复跑：standalone adapter probe 能访问 MDL 事件宏，但跨 dynamic-module import 后宏不可见。最终由 adapter 返回解码后的 `transmission` 字段，避免 path tracer 依赖 MDL 私有宏。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | MDL viewer 的 direction、`bsdf_over_pdf` 与 MIS PDF 全部来自同一 target code | DONE |
| P0 | GPU test | sampled direction 上断言 sample PDF 等于 formal PDF，weight 等于 evaluate/PDF | DONE |
| P0 | Integration | car paint 与 glazed ceramic 真实 1024 spp capture 检查 high tail 与局部孤立点 | DONE |
| P1 | Documentation | `.trellis/spec/viewer/mdl-reference.md` 禁止 fixed-GGX/MDL evaluate 配对和 clamp 修补 | DONE |
| P1 | Review guide | source PT 检查 sampler/weight/MIS PDF 是否属于同一 estimator | DONE |

## 4. Systematic Expansion

- **Similar Issues**：任何 multi-lobe、delta-like 或纹理驱动粗糙度的 source closure 都不能仅凭 evaluate finite 就假定 generic proposal 可用；MERL 与 MaterialX 的 generic proposal 应在各自出现相同症状或扩展尖锐资产时单独审计。
- **Design Improvement**：把“公开 source capability”与“viewer 为正确积分而消费的内部 target-code transport”分开，避免为了不承诺公共 sampler 而退回错误 estimator。
- **Process Improvement**：新 source PT 的 smoke 除 finite/identity 外，至少选一个高动态范围资产检查 sample-weight high tail、MIS PDF 和空间孤立点。

## 5. Knowledge Capture

- [x] 更新 `.trellis/spec/viewer/mdl-reference.md` 的 matched transport、错误矩阵、测试和 wrong/correct。
- [x] 更新 `.trellis/spec/viewer/index.md` 与 `.trellis/spec/guides/index.md` 的质量门。
- [x] 更新稳定文档与 viewer 用户说明，区分内部 transport 与公共 source capability。
- [x] 记录诊断脚本和修复前后 capture；运行产物保存在 ignored `artifacts/`。
