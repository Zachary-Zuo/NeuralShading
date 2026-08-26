# Baseline 复现反复的根因与修订

## 它是什么

本文记录 2026-08-26 对 `08-25-03-neural-baseline-and-candidate` 的回溯。结论不是“模型还差一点过线”，而是任务把三个不同问题混成了一个循环：方法是否正确、训练是否收敛、最终质量是否达到人为阈值。旧流程只要最后一个数值不满足，就回到实现和训练重跑，因此无法形成稳定终点。

## 已发生的运行

- `formal-direct-v1` 完成 6,750 steps，训练约 61 分钟；只走 mollification groups，没有进入 base-v5 阶段。
- `formal-direct-v2` 完成 20,000 steps，训练约 78 分钟；包含 mollification 与 base-v5 groups。
- 两次 run 都产生了有限 checkpoint 和可评测结果，但使用的是当前自定义 direct 形态，不是已经逐项审计完成的 NVIDIA 原方法。因此它们只能说明训练链路能运行，不能证明 baseline 已复现。
- 进程记录中存在约 7 小时 38 分钟没有训练工作的等待区间；随后 watcher 又排队了 core、paper、缩模 direct 和 sampler runs。等待链把“任务在推进”和“进程仍存在”混为一谈。

运行中还出现两类真实实现问题：

1. SlangPy Torch wrapper 会按公开 callable 身份缓存首次观察到的 active-gradient mask。冻结 evaluator 路径先 warm 后，sampler-only 阶段若复用 callable 身份，目标参数可能没有 `grad_fn`。这需要 role-specific wrapper identity，已经形成长期 spec；反复重启训练不能修复。
2. direct evaluator 后来加入最低输出比例 floor，说明训练诊断仍在改变函数形态。该改动也不是 NVIDIA 原方法的一部分，不能在变动中的实现上继续用质量门判断“复现是否成功”。

## 旧验收为什么不成立

旧 Q1 使用的 `0.045 / 0.10 / 0.013 / 0.15` 来自早期 P1 run 的观察值和已知 state 结果，其中部分数字甚至比被引用 run 的 observed p95 更严格。`docs/research/experiment_framework.md` 原本明确把质量线称为可修订参考而非 kill gate，后续计划却把它提升成跨 state 的硬验收。这是 spec drift，不是由不同材质的物理或感知特点推导出的合同。

旧任务还同时要求：

- 原规模论文结构只作 `diagnostic`；
- 另做 `≤2k MAC` 缩模形态作为正式 baseline；
- 缩模结果必须过上述绝对质量门；
- 未过门就继续在冻结预算内修改和重训。

这会系统性地产生循环：缩模形态并不等于原方法，过不了质量线又不能放宽或改变预算；即使训练正常完成，也既不能证明忠实复现，又不能结束任务。

## 对只读二手复现的结论

`D:\01_Workspace\Real-Time Neural Appearance Models` 提供了有用的结构对照：默认 latent 8、两个由 latent 提取的 frame、`z + T wi + T wo` decoder、`3×64` 最大 preset、`3×32 → 9` sampler，以及较长的正式训练配置。它只能作为二手线索，不能直接作为 correctness oracle。

只读审计发现其中至少有这些边界需要独立验证：layered BRDF GT 是单层 Cook–Torrance 的加权和，不是本项目 LayerStack random-walk reference；方向 softening 与已生成 target 的配对存在疑点；`dual_batch` 配置没有形成实际训练数据流；sampler objective 与 evaluator detach 边界需要重新核对；quick start 的短 schedule 和 Mitsuba sphere render 也不等于本项目正式 viewer 生命周期。因此本项目只参考其结构，不复制其 GT、loss、训练结论或 viewer 证据。

## 修订后的终点

本任务继续使用原目录，不创建新任务。复现状态只由以下证据决定：

1. method-correspondence 逐项证明实现对应原方法或明确登记为 adaptation；
2. loss、梯度、权重有限，validation 相对初始化改善，后期无可信发散，多 seed 判断一致；
3. checkpoint 可恢复，SlangPy/Falcor/packed asset 是同一实现；
4. sampler 的 PDF/null/sample 数学正确。

directional/energy/visual quality、sampler 方差、时间和内存继续完整报告，并按材质结构分组比较，但不再决定“复现成功”。原规模 baseline 必须进入 MethodBundle/viewer；超过软成本线只改变成本分类，不触发缩模替换。

## 进程处置

需求修订后已停止尚未开始产出、会继续训练缩模 direct/sampler 的两个 watcher；没有删除任何 artifacts。随后 `formal-core-v2` 在 3,750 steps 以 `status=failed` 结束，排队的旧 `formal-paper-v2` 没有启动；当前已无 03 正式训练进程。core 失败产物保留用于诊断，但现有 paper pipeline 也已被一手 correspondence 证明不忠实，因此不会继续训练。后续只有在原规模 baseline 结构、joint lifecycle 与 convergence report 冻结后才启动新的 formal run。
