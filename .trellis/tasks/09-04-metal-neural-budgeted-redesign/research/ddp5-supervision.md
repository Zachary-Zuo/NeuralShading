# DDP5 持续监督记录

## 范围

本文件记录 GPU 5–9 上 hybrid/direct matched pair 的里程碑、实现缺陷、修复身份与实验解释。完整 stdout/stderr、checkpoint、metrics和机器采样保存在 `artifacts/metal-budgeted-pilot/ddp5-20260905/`，不提交到根仓库。

## Attempt 1：hybrid fresh step 0

- 时间：2026-09-05 01:44（Asia/Shanghai）
- 命令入口：`scripts/run_falcor_python.sh --gpus 5,6,7,8,9`
- resolved plan：hybrid DDP5，物理 GPU `[5,6,7,8,9]`
- 已完成：五个 rank 均创建 `NVIDIA RTX A6000` Vulkan device；train-only calibration完成；rank 0写出225,247 B checkpoint、SHA-256 sidecar与summary；进程退出后GPU全部释放。
- 失败：最终review读取0 B metrics时抛出`ValueError: training review requires metric rows`，Gloo控制面把rank-0错误传播给全部rank，torchrun整体退出1。
- 分类：training lifecycle implementation defect，不是NCCL reducer、reference、模型数值或资源问题。`stop_at_step=0`按冻结协议不执行optimizer step，因此空metrics合法；最终review路径错误地把正step约束用于step 0。
- 修复：metric loader新增默认关闭的`allow_empty`；CLI只有在final checkpoint `global_step == 0`时启用。正step空metrics仍失败，不改变formal readiness、method/data/query identity或训练数值。
- 回归：`test_training_review.py`覆盖step-0显式允许与默认拒绝；相关engine/launcher测试共同通过。

## Attempt 2：hybrid step 0 → 8

- 时间：2026-09-05 01:49（Asia/Shanghai）
- 恢复边界：Attempt 1修复后的hybrid DDP5 step-0 checkpoint。
- 已完成：五rank恢复相同checkpoint并取得第一步online evaluator/sampler batch；错误由所有rank一致报告，进程组退出后GPU释放，step-0 checkpoint未被部分覆盖。
- 失败：通用`validate_objective_outputs()`报告11个required component output缺失，包括asset/compiler/direction/evaluator trace与proposal identity。
- 分类：budgeted method implementation defect，不是DDP reducer或模型数值失败。objective已计算对应值，但只返回带`trace/`、`proposal/`前缀的诊断metric；descriptor要求的component output alias没有进入mapping。原unit只检查标准loss和gradient，没有执行通用conformance gate。
- 修复：保留原诊断metric，并为descriptor要求的component outputs增加detached alias；method unit直接调用`validate_objective_outputs()`。修复修改`metal_budgeted.py`并改变implementation identity，因此Attempt 1的两份step-0 checkpoint标为superseded，hybrid/direct都必须fresh重跑。

## Milestone 8：implementation `6b9f81c`

- hybrid/direct均从fresh DDP5 step 0开始，calibration的scale/P95/epsilon、reference execution plan与query stream一致；不同calibration hash只来自各自training config hash。
- 两侧0→8均完成，所有metric有限，六个required parameter group均已有finite/nonzero gradient与update，rank0-only checkpoint/review及teardown通过。
- peak allocated memory均为739,713,536 B/rank。hybrid/direct review median分别约2.928/2.903 step/s；首步含compile约3.6–3.9秒，后续tqdm显示约2–3 step/s。
- step 1→8 appearance：hybrid `3.6547→2.6996`，direct `4.7860→3.8209`；spatial gradient分别`0.3325→0.2541`与`0.3324→0.2534`。这是极短早期诊断，不形成结构选择。
