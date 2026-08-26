# Learning pipeline 与评测

P1 已注册第一批外观建模候选：M1 conditioned shared evaluator 的 S/M/L 三档、matched M2 analytic residual 的 S/M/L 三档，以及无共享 latent 瓶颈的 per-state teacher。M3 response-space oracle 使用独立的直接拟合入口，只做 canonical probe 上的字典可压缩性诊断，不伪装成可连续查询的 runtime 候选，也不进入 quality-v1 排名。仓库不为迁移前候选保留 registry 别名。

P1 v1 已完成正式单-seed 比较。M1-M 是通过全部主参考线的 best observed quality 候选，M1-S 是实际查询更快的 Pareto 端点；P2 以 M1-M 做质量起点，同时保留 S 做效率对照。M1-L、M2-M/L 和当前 per-state teacher 配置停止扩张；M2-S 只保留“中位强、困难尾部弱”的机制对照；简单 M3 top-2 字典不进入主路径。完整 run、置信区间和成本见 [`experiment_log.md`](research/experiment_log.md)。

## Pipeline 身份

候选使用短而可读的名称，例如 `film-evaluator-s-v1`。descriptor 不再把所有选择拼进一个长 ID，而是分成四组结构字段：

- `data`：reader、partition、source adapter；
- `model`：representation、architecture、latent；
- `fitting`：gradient/direct-fit/hybrid 路径和 loss；
- `runtime`：compiler 与 exporter。

名称用于人读，descriptor 的 SHA-256 才是精确实现身份。`capacity` 可省略，P1 v1 历史配置保留其 S/M/L 字符串。schema 见 [`learning_pipeline_v1.schema.json`](../src/ncls/learning/schemas/learning_pipeline_v1.schema.json) 和 [`training_config_v1.schema.json`](../src/ncls/learning/schemas/training_config_v1.schema.json)。

## 数据读取

训练和评测的 `--data` 可以指向单个 `reference-shard`，也可以指向完整 `reference-corpus` manifest。corpus reader 建立全局 state 索引，但每个实际 batch 只从一个矩形 shard 读取，因此 W/G/S、普通 role 和 dense slice 的不同方向数不会被混装或 padding。

Directional mollification 由 `MollificationCurriculumStore` 读取冻结的 `mollification-training-data-entry`，训练配置不得自行选择 corpus、半径或阈值。reader 输入 normalized training progress：`t<0.875` 选择距离最近的 `{0,0.25,0.5,0.75}` level，并返回 `mollification_progress`、实际 level progress、radius 与 `target_source=mollified-reference`；`t≥0.875` 返回同一 anchor 锁定的 base-v5 0° response，并标记 `target_source=base-v5`。这是一组有限离线 target，不宣称与连续在线 schedule 无误差等价。

`parametric-v1` 使用 source split 与同名 query role 的交集；`target-visible-v1` 只按 query role 切分，允许 encoder 或 bounded refinement 读取 source-test state 的 train response；`workflow-v1` 用于资产式族的固定工作流批量评测。三者必须在候选 descriptor 中明确选择。

## 统一散射 evaluator 与 sampler（03）

03 的 NVIDIA evaluator 与 `core-frame-neural-v1` 各自拥有一份 Falcor-free Slang 生产前向，分别位于 `shaders/ncls/backends/nvidia_neural_appearance/` 与 `shaders/ncls/backends/unified_neural/`。SlangPy 训练、Falcor parity、MethodBundle 和 viewer 编译各方法自己的同一份 core；Torch 只承载 loss、optimizer、统计与独立数值 oracle，不提供第二套生产前向。参数 offset 从实际模型反射并随 packed asset 版本化，不能在 Python、Slang 与 viewer 中各维护一份布局。

### NVIDIA 复现身份与训练协议

论文方法包含两层不同要求，必须同时满足后才能称为忠实复现：

1. **方法形态**：`z8`、只依赖 latent 的两个非正交 learned frame、`20→64→64→64→3` evaluator、`exp(raw-3)` cosine-weighted response，以及 `11→32→32→32→9` 的 GGX9 sampler；
2. **训练生命周期**：训练时在 GPU 上在线生成 source-material query 并立即求 reference BRDF；论文报告 300k iterations，每次分别处理两个 65k batch，共接近 400 亿个在线样本。

当前正式入口已改名为 `nvidia-frame-two-lobe-layer-stack-budget-adapted-v1`，只对齐第一层的主要网络形态。它使用冻结的 LayerStack HDF5/mollification entry、25k steps、每 step 16 个 query group；每组 64 个方向，因此每条训练 route 每 step 只有 1,024 个方向查询。它是 **LayerStack 上的离线、预算适配诊断**，不是论文训练协议的忠实复现，也不能用 `paper baseline` 描述其训练结论。旧 run/artifact 中的 `nvidia-frame-two-lobe-paper-v1` 只按原 hash 作为历史诊断 provenance 保留；pipeline registry、config 与 exporter 不提供旧身份 alias。

该 25k run 的 validation 相对初始化显著改善、数值有限且后期没有发散，但 best checkpoint 正好落在最后一步，最后 25 个 validation 记录的 normalized slope 为 `-0.168`，95% bootstrap CI 为 `[-0.265,-0.099]`。这说明训练仍在可信改善，证据只支持“有限、改善、未发散”，不支持“已经训练到论文规模或已经饱和收敛”。其 test directional L1 state median/p95 为 `0.680/1.205`，能量相对误差 median/p95 为 `0.408/4.372`；这些数值说明当前 checkpoint 本身存在显著底色/能量误差，不是 viewer packing 或 shader parity 单独造成的。

现有离线 curriculum 读取 `artifacts/corpus/layer-stack-p1-mollification-training-v1.json`（entry `47ef2013…5a89`）：前 20k step 由 reader 在四个冻结半径和 base-v5 之间路由，之后只读 base-v5。它可以继续用于 matched 的离线 LayerStack 候选比较，但必须使用独立的 `budget-adapted` 身份；若目标是论文复现，则需实现 GPU-resident online reference query 与 joint evaluator/sampler lifecycle，不能只把离线 step 数提高到 300k。

sampler 正确性仍不能用自己的 `sample/pdf` 自证：正式协议对 30 state × 4 个含 grazing 的 view 使用独立 CPU quadrature/PDF oracle，并检查 normalization/null、sample→pdf、histogram 和相对同一冻结 evaluator 的 MC 无偏性。packed checkpoint 还要在 SlangPy 与 Falcor 执行完整 evaluator+PDF parity；这些门只证明数学实现与打包一致，不证明模型已经学对 source material。

方法对应、训练状态和质量比较必须分开登记。绝对 quality 数值不决定 prior-art 方法形态是否实现正确，但训练预算与 online/offline lifecycle 属于 method correspondence，不能以“未设置绝对质量门”为由省略。候选比较只在共享数据角色、预算类别和评测协议的 matched cell 间报告 paired bootstrap 与成本；不得恢复旧的 Q1 绝对阈值或默认 winner 规则。

## 固定四层 quality-v1

候选只返回线性 RGB `f`。harness 在指标内部乘一次 `|cos θi|`，与 HDF5 的 response measure 对齐；candidate 不能替换 metric suite。报告以可读名称 `quality-v1` 和 [`quality-v1.json`](../configs/evaluation/quality-v1.json) 的 SHA-256 共同标识协议，比较器会拒绝 suite hash 不一致的报告。

1. sanity：数据/hash、role 完整性、输出 finite、颜色范围、checkpoint 可恢复、fitted state 只来自 train。只有这一层会把报告标成无效。
2. 主指标：solid-angle weighted normalized L1，先聚合到 state 再报告 median/p95；半球或整球积分能量相对误差按 `(state, wo)` 报告 median/p95。checkpoint 固定按 validation 的 directional state-median 选择，同值用 state-p95 决胜。
3. scorecard：log-domain error、95% 峰高支持集角距、峰高比、top-5% energy recall、source-aware reciprocity deviation，以及 difficulty/tag/family/structure/cohort/state breakdown。它们解释长尾，不单独否决候选。
4. diagnostics：绝对误差、reference SE 与两者比值。`model error / reference SE` 不再是 kill gate。

`ncls learn audit-p1` 对冻结 checkpoint 另行计算逐 state / 通道 / role 的 signed 能量比、M2 的 clamp 死区比例与 `E_core/E_ref` core coverage、achieved reference SE（group p95 与 integrated ratio）以及 30-state p95 的 bootstrap CI / leave-one-state-out 范围，输出 `p1-audit-report`；它不改动 `quality-v1` 的主排名。

reciprocity 使用语料中落盘的 reciprocal paired response，比较“模型互易偏差与 source 自身互易偏差之差”；因此非严格互易的 source 不会被错误地强制为零。

每份 checkpoint report 另记录 `B_asset`、`B_shared`、`C_prepare`、`C_eval`、state bytes 和参数数目。成本用于 Pareto，并按 `docs/research/experiment_framework.md` §0.1 的部署软线在注册表标注是否为部署候选；超线 run 不淘汰，但不能成为默认配置。

研究端 query benchmark 通过 Torch CUDA event 计时，但实际 forward 仍执行同一 SlangPy core；`single_query` 测一个 `(state, wo, wi)` 的串行延迟并额外落盘 30-state matched timing，`coherent_packet` 在同一 `(state, wo)` 下批量求多个 `wi`，把一次 `prepare` 的成本摊薄后报告每方向时间。它用于当前硬件上的相对成本曲线；最终 viewer 时间仍由 05 的部署轨道负责，不能拿研究进程的调度开销代替 viewer 结论。

## 命令

注册候选后，训练只接受完整 `training-config-v1`：

```powershell
conda run -n neural-shading python -m ncls.cli learn train `
  --data artifacts/corpus/layer-stack-v1.json `
  --config configs/learning/film-evaluator-s-v1.json `
  --run artifacts/runs/film-evaluator-s-v1-seed-1
```

P1 主搜索以效率和准确外观优先：全部候选先使用同一 deterministic seed、同一子语料和同一最大 step 预算；训练配置从 4,000 step 后启用 validation patience，曲线停止改善时允许提前结束。只有 matched 候选差距接近、训练轨迹异常或结论准备升级时才自适应追加 seed，不对每个候选机械重复三次。state-block paired bootstrap 仍用于量化同一 run 在冻结 state 集上的外观差异；它不冒充 seed 方差估计。

M3 直接拟合在 dense slice 的冻结 `wo×wi` probe 上分别构造 per-state 全响应和 per-`(state, wo)` 方向轨迹，运行 K-means++ top-2 闭式凸混合，并按相同总 bytes 配一条 PCA 对照。每个 K 对 matched unit 的 linear relative L1 做 1,000 次 paired bootstrap；报告还给出由 HDF5 standard error 推出的 one-SE 参考噪声地板。逐 unit relative L1 的分母使用“非零 unit L1 中位数的 1%”作为下限并记录实际 floor，避免零能量 view 把均值放大到无意义量级。置信区间跨零或差异低于噪声量级时不作结构胜负结论。P1 只有 30 个 state，因此只执行小于实际 unit 数的 K；更大的 256/1024 codebook 留到 P2 全语料，不用重复原型制造虚假低误差：

```powershell
conda run -n neural-shading python -m ncls.cli learn oracle-m3 `
  --data artifacts/corpus/layer-stack-p1-v1.json `
  --output artifacts/oracles/layer-stack-p1-v1-m3.json `
  --codebook-sizes 8 16 32 64
```

显式读取 validation、test、adversarial probe 或 dense slice：

```powershell
conda run -n neural-shading python -m ncls.cli learn evaluate `
  --data artifacts/corpus/layer-stack-v1.json `
  --checkpoint artifacts/runs/film-evaluator-s-v1-seed-1/checkpoints/best.pt `
  --split test `
  --output artifacts/runs/film-evaluator-s-v1-seed-1/quality-test.json
```

matched 对照必须使用同一 `data_id` 和完全相同的 test state；比较入口以 state 为 block 重采样每个 state 的完整方向主指标和 `(state, wo)` 能量行，要求至少 20 个 matched state，并做至少 1,000 次、95% 置信度的配对 bootstrap。P1 冻结子集含 30 个 state，置信区间宽度会如实反映这个规模，不再被面向全语料的 50-state 旧门槛阻断：

```powershell
conda run -n neural-shading python -m ncls.cli learn compare `
  --baseline artifacts/runs/baseline/quality-test.json `
  --candidate artifacts/runs/candidate/quality-test.json `
  --output artifacts/comparisons/baseline-vs-candidate.json
```

比较器要求 baseline 与 candidate 的 `steps / seed / dataset_selection` 完全一致，不一致直接拒绝；候选之间的差异只体现在 pipeline 与 model。

冻结 checkpoint 后测实际 query 成本：

```powershell
conda run -n neural-shading python -m ncls.cli learn benchmark `
  --data artifacts/corpus/layer-stack-p1-v1.json `
  --checkpoint artifacts/runs/film-evaluator-s-v1-seed-20260824/checkpoints/best.pt `
  --packet-size 256 `
  --output artifacts/runs/film-evaluator-s-v1-seed-20260824/query-benchmark.json
```

历史 profile、训练 schema、gate 和 pipeline 不设 reader、converter 或 registry 别名。历史证据只从对应 Git 提交读取；正式新 run 必须使用当前 corpus 与 quality-v1。
