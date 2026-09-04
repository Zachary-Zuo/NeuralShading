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
