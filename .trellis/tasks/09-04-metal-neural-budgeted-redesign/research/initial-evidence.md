# 初始证据：旧 full profile 与重构动机

## 结论摘要

`metal_fused_full_v1` 当前同时暴露质量和成本问题，不能再作为“先把 full 做好、以后自然可以缩小”的默认研究载体。旧结果只能证明该具体大结构在当前训练下的行为，不能证明相同机制缩到部署预算后仍保留质量。因此，本任务需要先冻结有意义的主 profile 预算，再从最低必要机制重新设计。

以下均为规划前已观察事实或代码审计结果，不是新任务的 hard gate。

## 需求变化来源

- 2026-08-29 的原始需求明确要求先融合 NTC-style texture、NVIDIA neural appearance、typed 参数编译等优秀组件，优先追求质量，再通过消融删除无效组件；旧 full profile 按这一授权实现。
- 2026-09-04 用户根据 20k 图像和运行时差距修正了研究边界：当前 full profile 已大到无法支持对低预算同构模型的推断，应重新全面分析并实现更合理的模型。
- `trigger`：用户本轮明确变更；`invalidated evidence`：旧 full profile 即使后续取得高质量，也不能单独支撑 compact/deployment 结论；`scope impact`：停止把旧结构的缩模消融作为默认下一步，新增预算冻结、受控表达力、matched runtime 和新模型实现；`rerun required`：新 method/profile 使用新的 pilot、checkpoint 与正式结果身份，旧 v4 run 只保留为历史对照。

## 质量观察

- `artifacts/viewer/metal-step00020000-tungsten/viewer-window.png` 中，reference 一侧可见的细刷痕在 neural 一侧基本消失，neural 高光呈现明显绿色/黄色偏色。
- 对应 package 的量化 Python validation 预期输出已经带有显著通道失衡，因此该偏色不能只归因于 Slang 或 viewer parity；仍需用同一 probe 分离 eager、量化和 Slang 的误差来源。
- 20k 附近 validation appearance 没有给出稳定单调改善证据；“只训练了 20%”不足以推断继续训练会自动恢复微细节或颜色。

## 训练目标观察

- 旧训练日志中的 total loss 为实数负值，不是复数。负值主要来自 proposal 的连续密度 NLL：当 PDF 密度大于 1 时，`-log(pdf)` 可以为负。
- proposal 梯度与共享 appearance 参数之间做了 detach，因此负 proposal 数值不会简单抵消 appearance 梯度；但 total loss 仍会掩盖 appearance 的真实进度。
- `joint-coarse-to-fine` 的 phase step 主要调节 proposal 权重，appearance 路径没有真正的空间频率或高光 coarse-to-fine curriculum。
- 当前亮峰和 proposal 目标较依赖 luminance，训练与 validation 缺少直接暴露逐通道偏色、色度、峰值尾部和空间高频损失的观测。

## 表示观察

- 旧 codec 的高分辨率 grid 仅在半分辨率工作，低分辨率 grid 约为八分之一分辨率，并在 QAT 中量化。
- per-slot 特征会进入聚合/平均的 `structured` 表示；semantic/normal head 虽有辅助监督，但 evaluator 的 runtime spatial state 并不直接消费该受监督输出。
- 现有 topology 包含 typed attention、宽 U-Net、多级 latent bank 和多 lobe head。这些模块是否对 Metal 的高频方向性结构和 RGB 高光保真具有净贡献，尚无 matched 消融支持。

## 成本观察

静态描述中的旧 full profile 约为：

| 项目 | `metal_fused_full_v1` | NVIDIA faithful evaluator 对照 |
|---|---:|---:|
| `prepare` MAC | 2,416,000 | 需 matched 重测并补齐口径 |
| `evaluate` MAC | 185,088 | 9,664 |
| 首次 `prepare+evaluate` MAC | 2,601,088 | 需 matched 重测并补齐口径 |
| prepared state | 2816 B | 96 B |
| 固定随机读取 | 106 | 需按 faithful ABI 统一统计 |

旧 full profile 仅 evaluator 就约为 NVIDIA faithful 对照的 19 倍，prepared state 约为 29 倍，且 `prepare` 进一步占主导。历史 viewer 整帧时间也显示 neural 明显慢于 reference，但该数据不能替代本任务要求的 matched kernel 测量。

## 对新设计的约束含义

1. 新主 profile 应从目标成本反推状态、读取和网络拓扑，不能先构造大模型再寄希望于后续剪枝。
2. 大 teacher 只能帮助区分“数据/训练不足”与“主 profile 容量不足”，不能承担目标方法的结论。
3. 单材质受控过拟合、细节/颜色观测和 matched runtime 必须先于多材质长训。
4. 若主 profile 在冻结 pilot 中得到低 observed quality，应按正常 empirical outcome 报告或回到经用户确认的新 planning；不得自动扩宽模型直到过门。
