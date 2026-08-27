# 规划审阅记录

## 审阅结论

2026-08-27 已完成 `prd.md`、`design.md`、`implement.md` 与当前忠实度审计的逐项收敛。任务边界与用户最新授权一致，允许按 task-scoped continuous execution 直接启动。

## 冻结摘要

- **目标**：把当前只保留 decoder/sampler 骨架的 NVIDIA 实现升级为作者公开完整方法的 functional reproduction，并在统一 pipeline 中让 neural package 真实进入 PT/deferred。
- **包含**：encoder、hierarchical z8、filter/mollification、两阶段 lifecycle、独立双 65k route、300k formal、two-lobe sampler、typed latent resources、generic binding、双 slot与 neural PT。
- **不包含**：作者未公开资产/训练代码/tensor-core intrinsic、论文图像逐数值复刻、UE或额外研究候选。
- **验收来源**：用户明确需求、一手论文/补充材料、统一 pipeline/scattering/viewer合同与数学不变量；observed quality/time/memory只报告。
- **作者未公开项**：全部进入 versioned recipe identity；不得当作者事实，也不得在看到 formal 结果后静默调整。
- **source 边界**：MaterialX standard_surface 提供 spatial/texture/footprint功能证据，LayerStack提供显式1×1适配；两者保持各自 native GT，不反演成对方表示。
- **执行顺序**：correspondence/math → data/source → lifecycle → package/binding → neural PT → formal run → full gate/归档/commit，每段都有独立 rollback point。

## 连续执行核对

`task.json.meta` 已记录：

- `execution_mode=continuous`
- `continuous_authorized=true`
- `authorization_parent=08-27-faithful-nvidia-neural-materials`
- `commit_policy=preauthorized-scoped-local-no-push`

本次细化没有离开冻结范围。context manifest校验已通过；inline模式不做 jsonl curation。按 workflow continuous exception 可直接 `task.py start`。
