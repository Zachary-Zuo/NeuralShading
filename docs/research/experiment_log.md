# 实验注册表

每个正式 run（标准档及以上）一行；快速档 smoke 不入表。可比性、结论强度与对照要求见 [`experiment_framework.md`](experiment_framework.md) §7。详细数值与逐项报告留在 `artifacts/`，本表只保留能回答「现在做到哪了」的最小信息。

v1 基准（P0）生效前本表为空；迁移前结果的适用边界已汇总在 [`model_candidates.md`](model_candidates.md) §9，不重复登记，也不为其保留当前 pipeline/config 入口。

| 日期 | run ID | 候选+档位 | 数据版本 | 预算档 | seeds | 方向 L1 (med/p95) | 能量误差 (med/p95) | 结论 | artifacts |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-25 | `4ce8bd54ddd0b7c27d279cc4ece426c200488f3aab15b366a9d841c41719a27b` | M1 FiLM S | LayerStack P1 v1 `0513d0c8…` | 25k（实际 25k） | `20260824` × 1 | 0.0506 / 0.1196 | 0.0116 / 0.2169 | 最快的稳健外观端点；median 略高于 0.05 参考线，保留为效率优先默认 | `artifacts/runs/p1-film-s-seed-20260824/` |
| 2026-08-25 | `ec1ceb5a745b6cf17e71d5c526c9125fd85f9e039fd7bd62f47d2f4e7d8a16f2` | M1 FiLM M | LayerStack P1 v1 `0513d0c8…` | 25k（实际 25k） | `20260824` × 1 | 0.0452 / 0.1180 | 0.0126 / 0.1551 | 三条主参考线均通过；S→M 质量差异 CI 跨零，作为 best observed quality 端点，不取代 S 的效率位置 | `artifacts/runs/p1-film-m-seed-20260824/` |
| 2026-08-25 | `58369ce13fa8011e0728bfd118f908040bd09878815524fe72123657d3f66710` | M1 FiLM L | LayerStack P1 v1 `0513d0c8…` | 25k（实际 25k） | `20260824` × 1 | 0.0766 / 0.2119 | 0.0195 / 0.2495 | 相对 M 的 median/p95 都显著变差且成本更高；淘汰该 L 配置 | `artifacts/runs/p1-film-l-seed-20260824/` |
| 2026-08-25 | `e15dfefb915d779c57ac51241bd8c38be0fb39b96fac9bf8ae66c3c7239a225d` | M2 analytic residual S | LayerStack P1 v1 `0513d0c8…` | 25k（早停 7.5k） | `20260824` × 1 | 0.0182 / 0.5862 | 0.0123 / 0.7510 | 相对 M1-S，median 显著更好但 p95 显著更差；当前 core 路径实测也更慢，不作稳健默认 | `artifacts/runs/p1-analytic-residual-s-seed-20260824/` |
| 2026-08-25 | `71bd9e733e768a2bf711f3bf0e9b950f9f4997ba9a1121db1bb7f301976437d5` | M2 analytic residual M | LayerStack P1 v1 `0513d0c8…` | 25k（早停 6k） | `20260824` × 1 | 0.0237 / 0.5618 | 0.0191 / 0.5534 | 相对 M2-S 无显著容量收益、成本更高；淘汰该 M 配置 | `artifacts/runs/p1-analytic-residual-m-seed-20260824/` |
| 2026-08-25 | `8c4168dbbc594d7da20f553b75338065cb4f5de07593003ce09cc1370a8428f1` | M2 analytic residual L | LayerStack P1 v1 `0513d0c8…` | 25k（早停 5.5k） | `20260824` × 1 | 0.0260 / 0.5342 | 0.0281 / 0.6875 | 相对 M2-M 无显著容量收益、成本最高；淘汰该 L 配置 | `artifacts/runs/p1-analytic-residual-l-seed-20260824/` |
| 2026-08-25 | `cd18d7bc0dabf66d093799becbbd7050bc029d8beab7dd7a6f181954e514e52a` | T per-state teacher L | LayerStack P1 v1 `0513d0c8…` | 25k（实际 25k） | `20260824` × 1 | 0.0775 / 0.3749 | 0.0184 / 0.3559 | 当前预算下未形成可引用的 high-capacity teacher；不支持把 M1 误差归因为共享 latent 瓶颈 | `artifacts/runs/p1-teacher-l-seed-20260824/` |

## P1 v1 阶段结论（2026-08-25）

- 数据：正式 corpus manifest 为 `artifacts/corpus/layer-stack-p1-v1.json`，`data_id=0513d0c837b109f74cbf6fd4f811e05c6bc68c02226bd6d443f3225ef5dd64b7`；69/69 shards 完整并通过 corpus/hash/split/role 校验。dense audit 为 `artifacts/corpus/layer-stack-p1-v1-dense-audit.json`，没有 state 需要提升到 16,384 directions。reciprocal 与 diagnostic high-noise 行保留 moments/SE，但不参与模型质量宣称。
- 评测：test、adversarial probe、dense slice 共 21 份报告均使用 `quality-v1` suite hash `3cf0db5e35ff06a55aeb43c5da342241f83495a1341ee8c9315f944c6ca758d1`，全部 `valid=True`。M1-S 三个 role 的 med/p95 分别为 `0.0506/0.1196`、`0.0512/0.1219`、`0.0522/0.1170`；M1-M 为 `0.0452/0.1180`、`0.0434/0.1096`、`0.0478/0.0990`。
- 容量与成本：RTX 4090 PyTorch benchmark 中，M1-S 为 `1.888 ms/query`、`7.662 µs/direction@256`，M1-M 为 `2.574 ms`、`10.447 µs`，M1-L 为 `4.389 ms`、`19.118 µs`。S→M 的 test median/p95 差异 95% CI 分别为 `[-0.0104, 0.0014]`、`[-0.0357, 0.0207]`，没有显著质量收益；M→L 两项都显著变差。因此 P2 以 M1-M 作为通过全部主参考线的质量起点，同时保留 M1-S 作为效率优先 Pareto 端点；不追加 seed。
- 机制对照：M2-S 相对 M1-S 的 test median 差异为 `-0.0325`，95% CI `[-0.0552,-0.0114]`，但 p95 差异为 `+0.4665`，CI `[0.0580,0.5704]`；解析 core 只改善多数 state，显著损害困难尾部，且当前实测 `4.777 ms/query` 慢于 M1-S。M2-M/L 没有显著容量收益。
- M3：`artifacts/oracles/layer-stack-p1-v1-m3.json`（report hash `4dbfb29eca55048fc78e0fc432c4ee6af6638bd3294ff821a6a55561e3a4d411`）中，top-2 字典在 matched bytes 下显著落后 PCA：K16 state response median/p95 `0.294/0.756` 对 `0.096/0.404`，paired CI `[0.071,0.200]`；不把简单字典提升为 P2 主路径。
- seed 决策：P1 主搜索只用统一 deterministic seed `20260824`。M1-S/M 的质量差异本身不显著，而 M 的运行成本确定更高；保留两个 Pareto 端点已经足以决策。额外 seed 不会改变当前效率默认，因此不做机械三次重复。
- 部署轨道：M1-M 已导出 `film-m1-direct-neural@1` frozen-state diagnostic MethodBundle，method ID `4fa39178cf0e094f3ec38a6e972c86602332261bf25fa0a661ac09b65292443a`。代表状态 `6324e3b2…` 是四层 diffuse stack，test directional L1 `0.02937`；Python/C++ LayerStackIR hash 均为 `563d08a7…`，viewer load-time GPU parity 通过。RTX 4090、320×240 输出（方法半屏 160×240）、单方向光下实测 prepare `82.52 ms`、lighting `112.72 ms`；因此它证明 neural evaluator 已正确部署，但明确不满足快速/实时目标。证据位于 `artifacts/exports/p1-film-m-best-6324e3b2-v2/` 与 `artifacts/captures/p1-film-m-headless-v4/`。
