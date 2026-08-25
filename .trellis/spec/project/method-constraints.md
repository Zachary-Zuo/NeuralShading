---
name: project-method-constraints
description: 新候选方法与 backend 的根本约束：必须具备硬件部署可能；探索期可以超软线验证表达力，但不可违反硬约束；注册时静态检查清单
paths:
  - src/ncls/learning/pipelines/**
  - src/ncls/learning/models/**
  - src/ncls/bundle/**
  - shaders/ncls/backends/**
  - configs/learning/**
  - docs/research/**
---

# 方法探究的根本约束

> 所有候选都朝着"能进 GPU shader 热路径"设计。允许用较大开销先验证表达能力，但不允许毫无部署可能的形态。数值依据：`docs/research/experiment_framework.md` §0.1（研究期软线）与 `docs/contracts/method_bundle.md`「成本信息」（硬线），两者数值一致。

## 两级线

| 量 | 软线 | 说明 |
|---|---|---|
| `C_eval`（每 `wi`） | ≤ 2e3 MAC（标量 ALU） | cooperative vector 工具链在远程机验证后放宽到 1e4 |
| `C_prepare`（每像素） | ≤ 1e4 MAC | inline prepare（`state_stride=0`）按灯数折算 |
| state bytes / 像素 | ≤ 64 B，或 inline | 1080p 下 1 KB state 就是 2.1 GB 缓冲 |
| `B_asset` | 均匀材质 ≤ 512 B；空间变化 ≤ 32 B/texel；含全部烘焙的 material-static 参数 | 不只计 latent |
| `B_shared` | `evaluate` 权重 ≤ 32 KB fp16；bundle 共享权重 ≤ 512 KiB | |
| 环境光 / 面光 | 解析或预滤波积分 capability；否则固定 ≤ 4 次 `evaluate`/像素 | |
| 实现 | `prepare/evaluate` 无数据相关循环、无 > 64 元素的函数级数组 | `bounded_execution` |

工况：RTX 4090、1080p、材质着色 2 ms/帧、每像素 1 次 `prepare` + ≤ 4 次 `evaluate`。硬线由 `MethodBundle.cost_claims` 校验（`src/ncls/bundle/manifest.py`）；viewer loader 侧校验是 `docs/research/p1_v2_plan.md` V4.2 待办。

## 探索期规则

1. **可以超软线**：为验证表达能力可以跑更大的模型；但 descriptor 必须 `runtime.deployment_candidate=False`，`docs/research/experiment_log.md` 登记为"非部署候选"，不得成为默认配置、不得进 D 部署轨道。`tests/unit/test_deployment_budget.py` 机械判定 `deployment_candidate=True` 的 pipeline 必须满足全部软线。
2. **必须写明收敛路径**：超线候选在 `docs/research/model_candidates.md` 或阶段计划里说明"哪个部件在部署档换成什么"；说不出来的候选不注册。
3. **不可协商的硬约束**（违反即不允许注册，无论多准）：
   - `bounded_execution`：latent/state 大小、网络结构、单次 evaluate 成本和内部循环都可静态界定，不随源材质图深度、层数或未受控随机状态增长；
   - 单次 `(state, wo, wi)` 查询读取量固定：有限个 latent / codeword / 权重块，与分辨率和历史查询无关；不把完整方向表藏进 `prepare()`；
   - `evaluate()` 输出线性 RGB `f`，不含几何余弦；余弦只在 loss/metric 内和 renderer 内各乘一次；
   - `prepare()` 结果可被同一着色点的多个 `wi` 复用，且不消费用于选择下一方向的随机数；
   - 声明 `ScatteringSampling` 就必须提供与 `evaluate` 共用 state、密度可计算且精确匹配的 `sample/pdf`；只能输出方向不能算 pdf 不算满足；
   - material-static 调制参数要么计入 `C_prepare`，要么烘焙后计入 `B_asset`，注册时二选一声明；
   - 残差参数化不得含死区：禁止 `clamp(core + signed Δ, 0)`（`docs/research/p1_audit.md` §4.2 实测在 4 个多层 state 截断 44–98% 残差），改用非负 lobe + 有界乘性修正。
4. **单一 Slang 源**：模型前向只写一份 Slang，训练（SlangPy）、GPU 测试（Falcor Python）、viewer 三处 `#include` 同一文件；Torch 只作 parity oracle。见 `core/shared-slang-backend.md`。
5. **主线形态**：`prepare` 输出 lobe 参数、`evaluate` 解析求和、精确 `sample/pdf`（`p1_audit.md` §5.1；实现 `shaders/ncls/backends/lobe_residual/`）。direct neural evaluator（§5.2）只在 cooperative vector 工具链验证后作同预算对照。不要再提议扩大 MLP 容量来解长尾。

## 注册时静态检查（`experiment_framework.md` §8，纸面检查，不需先写 shader）

- [ ] `LearningPipelineDescriptor`（`src/ncls/learning/pipelines/base.py`）的 `data / model / fitting / runtime` 四组字段精确齐全，`runtime.deployment_candidate` 为 bool；
- [ ] `parameter_costs()` 返回 `B_asset`、`B_shared`、`B_evaluate_weights`、`C_prepare_macs`、`C_eval_macs`、`state_bytes_per_pixel`，与软线逐项对照；
- [ ] 每 run 报告 signed 能量比、`E_core/E_ref`（有 core 时）、achieved reference SE；
- [ ] 面向 deferred 的候选声明环境光 / 面光积分方式（解析、预滤波或固定 query 预算）；
- [ ] 写明 `evaluate` 的实现路径（标量 ALU / cooperative vector）与依赖的工具链版本；
- [ ] 容量按 `S/M/L` 档位报告，成本每 run 必录。

## 反例

- 把 256 宽 FiLM MLP（M1-M：`C_eval` 8.6e5 MAC、state 1 KB、烘焙资产 19.5 KB）当默认候选或 P2 起点——它只是 P1 v1 的容量曲线记录。
- 先用 Torch 写"参考模型拿质量信号"再翻译成 Slang——会被扔掉的重复代码，已被否决（`p1_v2_plan.md` §6）。
- 靠放宽误差约束或解除预算来掩盖长尾表达力不足。
