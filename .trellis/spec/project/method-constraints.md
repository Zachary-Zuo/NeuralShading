---
name: project-method-constraints
description: 新候选方法与 backend 的根本约束：着色器预算内的单一最大形态；四条接口硬约束；单一 Slang 源与主线形态
paths:
  - src/ncls/learning/pipelines/**
  - src/ncls/learning/models/**
  - src/ncls/bundle/**
  - shaders/ncls/backends/**
  - configs/learning/**
  - docs/research/**
---

# 方法探究的根本约束

> 目标空间是能进 GPU shader 热路径的形态。容量上界由着色器预算定义：每个候选实现一个形态——预算内能容纳的最大容量——先在它上面追求质量，再在形态内提速。

## 预算

数值只维护两处：研究期软线在 `docs/research/experiment_framework.md` §0.1，硬线在 `docs/contracts/method_bundle.md`「成本信息」，两者一致，本文只引用。工况：RTX 4090、1080p、材质着色 2 ms/帧、每像素 1 次 `prepare` + ≤ 4 次 `evaluate`。硬线由 `MethodBundle.cost_claims` 校验（`src/ncls/bundle/manifest.py`）；`tests/unit/test_deployment_budget.py` 机械判定 `deployment_candidate=True` 的 pipeline 满足全部软线。

- 形态约束是硬的：`prepare/evaluate` 的循环次数、函数级数组（≤ 64 元素）、状态与权重大小全部静态定长。满足这一条的形态才进入注册表。
- MAC / state / 资产数值是软的：超软线的 run 可以跑、可以登记，descriptor 写 `runtime.deployment_candidate=False`，`experiment_log.md` 标「非部署候选」，默认配置与 D 部署轨道只取满足软线的候选；注册时一句话说明缩回软线时换哪个部件。

## 接口硬约束（注册的前提）

1. 有界执行：latent/state 大小、网络结构、单次 `evaluate` 成本静态可界定，与源材质图深度、层数和随机状态无关；单次 `(state, wo, wi)` 查询读取固定个 latent / 权重块，与分辨率和历史查询无关。
2. `evaluate()` 输出线性 RGB `f`；几何余弦只在 loss/metric 与 renderer 内各乘一次。
3. `prepare()` 结果可被同一着色点的多个 `wi` 复用；选择下一方向的随机数全部由 `sample()` 消费。
4. 声明 `ScatteringSampling` 的候选提供与 `evaluate` 共用 state、密度可计算且精确匹配的 `sample/pdf`。

## 实现形态

- **单一 Slang 源**：模型前向只写一份 Slang，训练（SlangPy）、GPU 测试（Falcor Python）、viewer 三处 `#include` 同一文件；Torch 只作 parity oracle。见 `core/shared-slang-backend.md`。
- **主线**：`prepare` 输出 lobe 参数、`evaluate` 解析求和、精确 `sample/pdf`（`docs/research/p1_audit.md` §5.1；实现 `shaders/ncls/backends/lobe_residual/`）。direct neural evaluator（§5.2）在 cooperative vector 工具链验证后作同预算对照。
- **残差参数化**：非负 lobe + 有界乘性修正（`p1_audit.md` §4.2）。

## 注册时纸面检查

按 `experiment_framework.md` §8 逐条过；`parameter_costs()` 返回 `B_asset`、`B_shared`、`B_evaluate_weights`、`C_prepare_macs`、`C_eval_macs`、`state_bytes_per_pixel`，material-static 调制参数计入 `C_prepare` 或烘焙后计入 `B_asset`，二选一声明；每 run 记录候选自己的配置轴（如 lobe-residual 的 `K`、`correction`）与实际成本。
