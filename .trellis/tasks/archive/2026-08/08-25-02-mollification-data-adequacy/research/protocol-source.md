# Directional Mollification 协议来源与现状检查

## NVIDIA 原始定义

官方 supplemental `nvidia_neural_materials_author_supplemental.pdf` 的 Training Procedure Details 给出：前 `M=20,000` 次迭代使用

```text
mollification cone angle = 0.5 * 10° * (1 + cos(i*pi/M))
```

并在 cone 内采样 256 个 `ωo` 估计模糊 BRDF。主论文进一步说明 cone 以原 `ωo` 为中心，训练初期平均多个方向样本，随后收敛回 reference。来源：

- https://research.nvidia.com/labs/rtr/neural_appearance_models/
- https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_paper.pdf
- https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_supplemental.pdf

论文没有把 cone 边界遇到 reflection-only hemisphere 时的条件分布、离线 corpus 离散层数或随机序列作为公共数据合同。因此本任务在不改变论文 `ωo`、10° schedule 和 256 samples 三个核心事实的前提下，显式版本化 upper-hemisphere 截断与四个离线 curriculum level；报告必须把后者称为有限 corpus 适配。

## P1 v5 实际布局

只读检查 `artifacts/corpus/layer-stack-p1-v1.json` 和对应 HDF5 得到：

- corpus 有 69 个 shard，selection 为 30 个 state；每个 state 具备 train/validation/test/adversarial/dense 五个 role。
- train 的 `(view_count,direction_count)` 分别为 W `48×512`、G `64×1024`、S `96×2048`；每个 `wo` 的 peak-aware `wi` 表独立。
- dense slice 为四个固定 `wo` × 8,192 uniform `wi`；四个 `wo` 为 `(-0.88947326,-0.25029257,0.38234919)`、`(0.69375205,-0.35007387,0.62940955)`、`(-0.17043296,0.45028126,0.87646985)`、`(-0.36717224,-0.91983861,0.13813573)`。
- v5 保存逐 pair response、proposal PDF 与 solid-angle weight，但没有 `wo` cone、jitter group 或 curriculum level；`wo` proposal density 也不落盘。

因此 v5 在字段上能训练 point evaluator，却不能仅凭 schema 宣称忠实表达 `ωo` cone average。是否可由邻域插值近似必须经过 support 和 fresh matched 数值 gate；结构观察不替代正式结果。

## 代表状态来源

- 纯 diffuse `1b6dc2…c49f` 提供方向平滑控制。
- 单层 anisotropic conductor `3d9258…ce5b5` 的 `alpha=(0.02306,0.00594)` 提供窄峰控制。
- `bd6de2…/6fff05…/4ebd92…/179606…` 来自 `docs/research/p1_v2_plan.md` 的四个既有失效尾部，覆盖三/四层 conductor、diffuse 与 sheen。
- 四个 dense `wo` 中最后一个 `z≈0.138` 明确覆盖 grazing，不另选结果驱动方向。
