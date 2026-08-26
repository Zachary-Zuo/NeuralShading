---
name: data-reference-and-corpus
description: reference provider 与 corpus 的执行规则：ReferenceProvider 只暴露原生 state / query / response、矩形 shard、五个 query role、噪声预算分层、reciprocal 只作 scorecard、reference shader 单一源、不保留旧 reader
paths:
  - src/ncls/data/**
  - shaders/ncls/reference/**
  - shaders/ncls/data/**
  - configs/corpus/**
  - references/**
---

# reference 与 corpus 执行规则

## ReferenceProvider

- 公共 `BaseProvider`（`src/ncls/data/providers/base.py`）只暴露原生 state、surface query、direction query 和 reference response；新族实现自己的 state 解析、难度分级（`W/G/S` + `T/M`）和 proposal adapter，再加对应 `CorpusPlan`。
- `ReferenceDescriptor`（`src/ncls/data/contract.py`）声明 `family_id`、`reference_id`、`native_schema_id`、`incident_domain`（`upper-hemisphere` / `full-sphere`）、`deterministic`、`capabilities`、`implementation_sha256`（`implementation_hash()` 对实现文件求 hash）。
- 参数式族（LayerStack、OpenPBR 常量资产）从源参数推难度；资产式族（MERL、MaterialX）用 `difficulty-probe-v1` 实测分级，结果随资产 manifest 版本化。
- 每个 GPU dispatch ≤ 4,096 条 query；更多时 provider 自动切 tile。

## shard 与 corpus（`docs/contracts/reference_dataset.md`）

- 一个 `reference-shard` v5 固定一个结构 family、一个 query role、一个 `direction_count`——训练 batch 因此永远是矩形，不 padding。`ReferenceDataset.open()` 只接受 v5，没有旧格式探测或转换。
- 五个 role：`train / validation / test / adversarial_probe / dense_slice`；每个 state 必须五个 role 齐全，各 role 独立 seed 与方向表，validator 拒绝同一 state 跨 role 的方向 hash 碰撞。
- `responses/` 保存 `mean / variance / replica_mean_a|b / sample_count / valid / event_flags / reference_pdf` 和 reciprocal 三元组；`queries/` 逐 query 落盘 proposal PDF 与 `1/(N·pdf)` 权重，训练分布与均匀立体角指标因此能分开算。
- `reference-corpus` v1 内嵌完整 `CorpusPlan` 与 `plan_sha256`；v2 再内嵌版本化 `corpus-selection`（P1 的 30-state selection：`configs/corpus/layer-stack-p1-v1.selection.json`）。validator 从基础计划重新枚举 state 集合，不信任 manifest 自报。
- `corpus_id` 由语义内容计算，不依赖文件名、时间或容器字节布局。

## v1 已冻结的噪声预算（`docs/data.md`「Reference 噪声与 reciprocal pair」）

下表只验证当前 v1 corpus 的 reference estimator 精度，不是所有后续语料自动继承的理论常数，也不是 neural 方法质量或任务完成门。新 corpus 若把精度值作为 hard gate，必须按 `project/research-execution.md` 记录来源、适用角色、为何必须为硬门以及达 cap 后的处置，并在 formal 采集前由用户确认。

| 用途 | 目标 SE p95 | 最终 group 门 | cap | 达 cap 未达标 |
|---|---|---|---|---|
| validation / test 主 response | 0.04 | 0.10（个别 state 在 CorpusPlan 显式晋升） | 262,144（晋升可到 1,048,576） | shard 失败，不写入 |
| train 主 response | 0.06（v1 审计参考） | 0.25（v1 审计参考） | formal plan 冻结的足够 SPP；v1 cap 262,144 | sample count 不满足 plan 则失败；SE 审计证明 plan 不足则停止发布并返回 planning |
| adversarial / dense | 0.08 | 0.50 只作参考 | 262,144 | 无条件落盘 |
| reciprocal（全部 role） | 0.20（train 0.50） | 0.999 只作参考 | 65,536（train 4,096） | 无条件落盘 |

- 只有排名 GT 的门是硬失败；有实测失败证据的 state 才在 `reference_budget.state_sample_promotions` 逐 state、逐 role 晋升，不整体翻倍。
- reciprocal pair 只进 source-aware reciprocity scorecard，不进训练、方向 L1 或能量主指标；高噪声 reciprocal 不得支撑质量结论。
- dense slice 默认 `4×8192`，只把 dense audit 证明峰邻域不足的 state 晋升到 `4×16384` 并独立成 shard；这两个值之外的密度不属于 v1。

## 正式采集只执行一次冻结规则

- pilot 与 formal 分离。pilot 要在代表性难点上测量噪声随 SPP 的下降、吞吐和成本，其 HDF5 不进入正式 corpus；pilot 结束后先冻结唯一 `CorpusPlan` / collection lock 与足够大的 train SPP，再启动 formal。
- formal plan 必须在任何正式 response 产生前声明每个 role 的方向分布、seed、每个 direction 的 SPP（或完整的自适应增量 / 停止规则）、最大 cap、失败行为和预计成本。按这份规则在一次 run 内逐 target 自适应增加 SPP 是合法的“采集一次”；看到 formal 结果后再提高 SPP/cap、换 threshold 或改 accumulator 不属于同一次采集。
- formal shard 必须达到 plan 声明的 sample count；SE/variance 用来审计预设 SPP 是否确实把 Monte Carlo 噪声压到可训练范围。若 formal 审计证明 SPP 仍不足，则停止、保留 failure report 且不发布 corpus；扩大 SPP 或改变 estimator 要回到 planning，连续执行授权不包含这个决定。
- direction count / proposal 控制参数空间与方向域覆盖；SPP 控制每个已选 direction 的 reference response 噪声。不得用增加方向数代替 SPP，也不得用极高 SPP 掩盖方向覆盖不足。
- 实现 bug 使 response 无效时，旧输出标 diagnostic/invalid，新实现使用新 identity 从冻结 plan 重新采集；不得把 bug 修复前后的 shard 拼成正式 corpus。
- 序列化 / manifest 导出只消费冻结 response，并且是确定性的。导出失败可以从相同 response 重试；reference query 不应随导出重试再次执行。

## Scenario: directional mollification supplement

### 1. Scope / Trigger

- 当冻结 audit 证明 base v5 不能重建训练早期的 directional mollification target 时，使用 `mollified-reference-shard/corpus` v1；它只补充固定 `wo` cone average，不替换 base v5。

### 2. Signatures

- 验证入口：`ncls data validate-mollification <manifest>` / `validate_mollification_supplement(path)`。
- 学习入口：`MollificationCurriculumStore(entry_path).batch(state_ids, view_indices, light_indices, training_progress=t)`；调用方只传冻结 entry，不扫描目录寻找最新 manifest。

### 3. Contracts

- 每 shard 固定一个 state，布局为 `8×4×64`；四个正半径 level 为 `10°/8.5355339059°/5°/1.4644660941°`，每 target 使用 256 个 deterministic upper-cap `wo`，`wi` 保持不变。
- composite corpus 必须覆盖 selection 的 30 个唯一 state；每个 shard 自带产生它的 budget plan/collection lock URI 与 hash。当前已经发布的 `f6931474…e4f3` 仍按其 v6/v7/v8 provenance 验证和读取，但这种历史组合只说明产物可追溯，不是未来 formal acquisition 的模板。
- training entry 显式绑定 base corpus ID、audit report hash、supplement corpus URI/ID 和 curriculum；`t<0.875` 取最近的正半径 level，`t≥0.875` 必须读取该 shard 内冻结的 base-v5 `source_response`（0°）。

### 4. Validation & Error Matrix

- state 缺失、重复或不等于 selection → corpus 拒绝。
- shard semantic/file hash、base/protocol/anchor/reference identity 或自身 plan/lock 不一致 → corpus 拒绝。
- validator 必须从 replica mean 复算 mean、variance 与 relative-SE `p95/max`；stored/manifest 摘要、sample count 或 plan identity 不一致 → corpus 拒绝。mollification supplement 属于 train target：SE 超出 pilot 对 frozen SPP 的预期表示 acquisition plan 不充分，停止发布并返回 planning；不得在同一次 formal 中据此自动提高 cap 直到过门。
- entry 与 supplement corpus identity 不一致、reader 遇到未知 state/越界 view/light → 训练加载或 batch 立即失败，不允许 fallback。

### 5. Good/Base/Bad Cases

- Good：pilot 在代表性难点上确定足够的 train SPP，随后冻结一个能覆盖全部 state 的 acquisition policy；formal 按固定 SPP 或预先声明的自适应规则完成后一次发布 corpus。
- Base：`t=0.875` 返回 `target_source=base-v5`、`radius_degrees=0`。
- Bad：formal 达 cap 后根据失败值自动生成 v2/v3 并继续，最后把多个结果驱动的 plan 拼成“原本就冻结”的 corpus；或在 reader 中按目录名猜测最新语料。

### 6. Tests Required

- unit：protocol/schema 严格字段、连续 `sample_offset` 与 CPU float64 Welford merge、摘要/tamper、curriculum `.874/.875` 边界和未知 state fail-stop。
- GPU/reference：通过 `scripts/run_falcor_python.ps1` 覆盖固定 batch reference 路径。
- corpus：同时运行 base v5 与 composite validator，并用真实 training entry 做 reader smoke。

### 7. Wrong vs Correct

```python
# Wrong：目录发现与隐式 fallback 会让训练数据随磁盘状态漂移。
store = MollificationCurriculumStore(find_latest_manifest())

# Correct：训练配置只消费版本化 entry；entry 再显式绑定 composite manifest。
store = MollificationCurriculumStore(frozen_entry_path)
batch = store.batch(state_ids, views, lights, training_progress=progress)
```

## reference shader（`shaders/ncls/reference/`）

- 随机游走 reference 已是单一源：`random_walk_reference.slang` 同时服务采集（`shaders/ncls/data/reference_layer_stack.cs.slang`）与 viewer。新族的 reference shader 同样只写一份，采集入口与 viewer 都 `#include` 它。
- `interfaces.slang` 提供四种界面的 evaluate / pdf / sample 三件套；纯 frame、Fresnel、cosine、GGX/VNDF 在 `shaders/ncls/scattering/` 单一拥有。`sampling.slang` 只保留 PCG/reference RNG、体相函数，以及“恰好一次 `nextFloat2()` 后转调公共 VNDF”的薄包装。改这些文件必须保持现有 reference GPU 测试、固定 seed 随机数消费与已采集结果的 hash 不变（`p1_v2_plan.md` S2.2 的验收）。
- reference 可以在 `evaluate()` 内部用随机样本、逐像素累积；它不参与 MethodBundle 成本比较，也不生成通用 packet。

## 反例

- 给采集器加 `--profile` 之类可拼接参数绕过 `CorpusPlan`。
- 把两个 `direction_count` 混进同一 shard。
- 用 train 的放宽噪声数据当排名 GT。
- 为"整体提升质量"把全 corpus 的 cap 翻倍。
- 把 pilot shard 晋升为正式数据，或把 formal 失败值持续写回 PRD 后自动提高 cap，直到某个自定数值门通过。
- 保留迁移前 HDF5 的 reader / converter；需要历史结果时用对应 Git 提交。
