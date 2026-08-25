# 02 Directional Mollification 数据充分性设计

## 1. 数据流与所有权

```text
frozen protocol config + base corpus manifest
                    ↓ freeze（只读旧 corpus）
              anchor-lock.json + hash
                    ↓ run（此后才允许 fresh reference）
 old-v5 kNN reconstruction ↔ matched cone reference
                    ↓ binary decision
          reuse-v5  或  generate supplement
                    ↓
      frozen training-data-entry.json → 03
```

`src/ncls/data/mollification.py` 拥有协议解析、anchor lock、cone 方向、旧数据重建、报告/decision 与 training data entry；`src/ncls/data/mollification_collection.py` 独立拥有 budget/collection lock、mollified shard、composite manifest 和 validator。LayerStack fresh evaluation 通过现有 provider/reference shader，不复制随机游走。`src/ncls/learning/data.py` 只增加读取冻结 data entry 与 curriculum target 的薄 reader，不拥有协议或阈值。

## 2. Freeze/Run 状态机

CLI 采用显式两阶段：

```powershell
python -m ncls data mollification-freeze --config <protocol> --output <anchor-lock>
scripts/run_falcor_python.ps1 -m ncls data mollification-audit --config <protocol> --anchor-lock <lock> --output <report>
```

- `freeze` 验证 base corpus、protocol schema、六个 state 和 dense/train role，解析精确 `wo/wi`，把 base `corpus_id`、dataset hashes、protocol SHA-256 和所有方向写入 canonical JSON。
- `audit` 在创建 GPU evaluator 前重新计算全部 hash；任何不一致立即失败。report 写 `protocol_frozen_at` 和 `first_reference_result_at`，并要求前者严格更早。
- fresh evaluator 的 query seed 只由 protocol seed、state ID、anchor/level index 派生；dispatch tile 不影响语义 seed。
- `decision` 是 report 的派生字段，不能由命令行覆盖。

## 3. Matched Query 布局

对每个 state 的每个 locked `wo`：

1. 取四个 fixed `wi`；
2. 对四个 curriculum level 生成同一规则的 256 个 `ωo`；
3. 一个 reference block 表示 256 个 view × 4 个 light，共 1,024 pair query，低于 4,096 dispatch 上限；
4. 每个 pair 保存 replica A/B moments，先在 cone 维平均再估 target variance；
5. 旧 corpus reconstruction 对相同 1,024 pairs 求 kNN，再按同一顺序平均。

总审计规模固定为 `6×4×4×4×256=98,304` pair directions。重跑复用已有 raw matched shard，只有 hash 完全一致时可读；不覆盖不一致文件。

## 4. 旧 v5 重建

对每个 state 构造 6D unit-vector embedding `x=[s_o·wo, s_i·wi]`，其中 scale 由固定 angular support 转成 chord 尺度，不由数据拟合。使用 deterministic `cKDTree` 查询 32 个候选，然后以精确角度重新排序和 Shepard-2 权重归一。相同距离以原 shard dataset ID、group index、direction index 排序。

重建会同时输出：

- cone mean RGB；
- 每个 pair 的最近 `wo/wi` 角距离；
- support fraction 和 kNN effective sample size；
- normalized L1 的 state/level/anchor 分解。

support 门独立于数值门：宽 diffuse 偶然数值相近不能证明窄峰数据语义充分。

## 5. Supplement 物理布局

新 supplement 不修改 `reference-shard` v5。它使用专用格式：

```text
mollified-reference-shard v1
  attrs: protocol/base/reference hashes, response_measure, jitter contract
  states: state_id + native/source identity
  anchors: source dataset/group, wo, wi[64]
  curriculum: progress[4], radius_degrees[4]
  responses: mean/variance/replica_mean_a/replica_mean_b/sample_count

mollified-reference-corpus v1
  base_corpus_uri/id
  protocol/anchor-selection hashes
  state selection + shard manifests/hashes
  totals + corpus_id
```

一个 shard 固定一个 structure family 和 64 个 `wi`，保持矩形。生成写临时文件，完成 schema、finite、noise 与 semantic hash 后原子 rename；已存在文件只有全部 identity 一致时复用。

## 6. Curriculum Reader

训练 data entry 的 schema 固定两种合法 variant：

- `base-v5-neighborhood-v1`：仅在 audit 全过时指向 base manifest 和 frozen reconstruction contract；
- `base-v5-plus-mollification-v1`：同时指向 base manifest 与 supplement manifest。

reader 输入 normalized training progress：`t<0.875` 选择最近的四个 stored level；`t≥0.875` 返回 base v5 0° target。batch 必须返回 `mollification_progress/radius/target_source`，让 `03` 的实验 provenance 能看见适配，而不是把 target 静默伪装成普通 v5 response。

## 7. Validation And Error Matrix

| 失败 | 行为 |
|---|---|
| protocol/lock/base hash 不一致 | 在 GPU 查询前拒绝 |
| representative state 或 role 缺漏 | freeze 失败 |
| support/noise/numeric/repeat 任一不通过 | decision 固定为 supplement |
| raw audit shard identity 不一致 | 不覆盖，要求新路径 |
| supplement 任一 state/level 缺失 | manifest 不发布 |
| response 非有限、measure 非 `f·|cos|` | shard 失败 |
| 达 sample cap 仍超过 train hard limit | corpus 失败，03 保持阻塞 |
| reader data entry 与 manifest hash 不一致 | 训练加载前拒绝 |

## 8. 测试设计

- 纯 CPU：Hammersley cone 的确定性、solid-angle 统计与 hemisphere 截断；selector tie-break；kNN/support；metric/decision 边界；schema/hash/tamper；curriculum routing。
- HDF5：矩形布局、semantic identity、原子写/续采、不完整 state/level 拒绝、base provenance。
- GPU/reference：小型 fixed-state block 的 fresh mean 与 direct LayerStack reference 一致；dispatch tiling 不改 hash；完整 audit 与必要 supplement collection。
- 全量：`validate-corpus` 继续通过原 v5；新 supplement validator 和 learning reader 通过；锁定 upstream 干净。

## 9. Rollback Points

- audit 代码与 supplement schema 分开提交/测试；audit 失败是有效实验结果，不回退阈值。
- supplement 生成中断只留下显式 `.tmp` 或未登记 shard，manifest 不发布；续采只复用 verified file。
- 如果固定规模无法达到 reference noise 门，只能在 PRD cap 内倍增；不得降低门、减少 256 jitter 或让 `03` 在线猜测。

`reference-se-v1` 已证明原 512 cap 不足。后续 cap refinement 不回写原 protocol，而是新增 `mollification-supplement-budget` plan 与 collection lock；shard identity 和最终 corpus manifest 同时绑定这两个 hash。v2 cap 4096 来自冻结失败 p95 的平方反比采样量推导，写入全新目录，旧未发布 shard 保留且不得复用为 v2。

高预算不能把完整 cap 放进单次 shader 的 float32 `sum`：v5 的 65536 结果与 v4 在报告精度上不变，单 target 差分诊断又触发 `DXGI_ERROR_DEVICE_REMOVED`。v6 保持 cap 65536，但固定每 GPU call 最多256 samples，以连续 `sample_offset` 生成独立前缀，并在 CPU float64 用 parallel Welford 合并 A/B moments；accumulator 源码 hash与当前 reference implementation hash进入 plan/collection lock。

v7/v8 使用逐 state rollback：已通过 shard 的 HDF5 identity 继续绑定产生它的 plan/lock，新的 composite collection lock 只冻结 reuse evidence 和待采 state partition。最终 manifest 的每个 shard 都携带自己的 budget plan/lock URI 与 hash；validator 逐项加载并检查该 shard 自己的 gate，而不是拿最终 524288 cap 重新解释全部 30 个 shard。`corpus_id` 覆盖完整多预算 provenance。
