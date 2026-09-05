# 代码实施与验证计划

## 1. 当前交付与实施基线

研究定位、五张 viewer 原图检查、R8/R9 要求、模型审计已完成；基线为架构提交 `ea2d743`、HEAD `e3f1c21`。用户批准开始实施，当前状态为 `in_progress`；实际进展见 [progress.md](progress.md)。历史与当前锚点见 [post-architecture-code-plan.md](research/post-architecture-code-plan.md)。用户新增的 UV 分组要求以 design §9 为准，替换以下清单中的全资产两张平面、24 维 prepare 输入等单组假设。

环境重新确认为完整 Windows：RTX 4090、`neural-shading`、锁定 Falcor Windows Python 构建存在。实施已运行 unit 与 GPU witness，命令和结果记录于 progress.md；Linux/NCCL 不在本机证据范围内。

实施开始时读取 `trellis-before-dev`，复核 HEAD 和工作树，再读 project method-constraints/unified-pipeline/research-execution、learning online-training/pipeline-and-evaluation/deployment、data online-pipeline、core shared-slang-backend。只因后续新改动重查受影响部分，不重复旧图像研究或恢复旧入口。

## 2. 按依赖排序的代码清单

本次实际检查点见 [implementation-validation.md](research/implementation-validation.md)。勾选对应实现及已具备的 witness，不代表三个实际 source 的 D0 或 D1 已完成。`spatial_cook.py`、`spatial_asset.py`、`test_metal_spatial_cook.py`、`test_metal_spatial_runtime.py` 承接下文原计划放入旧 `asset_cook.py`/runtime 测试的职责；旧 profile 单测只保留历史数值回归。C6 fixture 已通过，真实其它 source 的质量仍待 D1。

### P1：D0 输入、过滤和通用资源关联（AC9/AC10/AC13）

- [x] 修改 `src/ncls/learning/source_adapters.py` 的返回合同；新增 `learning/conditioning_resources.py`，扩展 `batches.py` 的资源/binding 与 select/concat/lease。
- [x] 修改 `learning/producer.py` 的 conditioning 构建、packed evaluator、paired rejection、accepted batch 和 method-sampler 返回；`methods/nvidia/data.py` 只改空资源返回封装。通用 engine/session 不识别 Metal 名称。
- [x] `methods/metal/native_assets.py` 提供固定解码的原始 mip0 tile；按已有 role/domain 声明分组，复用 host/residency。将 raw decode 与有原生依据的 normal 处理分开，保留所有合法值域。
- [x] `methods/metal/data.py` 生成 CPU 预定 source/tile cohort、GPU row binding、UV/derivatives/pair/filter random；移除新路径的逐 query summary patch 和粗略 LOD 依赖。
- [x] `references/query.py` 增加可选 `filter_random` 并上传现有 meta 字段；producer 转发它，缺省保持 0.5。使用已有 footprint averaging，不另造 reference 公式。
- [x] 新增 `tests/unit/test_conditioning_resources.py`；扩展 `test_training_batch.py`、`test_online_training_producer.py`、`test_online_data_session.py`：两资产/重试/空选择/重排绑定、引用释放、异常/stop/resume、Nvidia 空资源回归。
- [ ] 新增 `tests/unit/test_metal_spatial_inputs.py` 和 `tests/gpu/test_metal_spatial_reference.py`：绝对 roughness 范围、图外统计不变、文件编码、非对称 UV/通道、原生 normal 顺序、point/footprint 独立求值。失败先定位 source/adaptation，不能按视觉误差最小选择翻转。

P1 完成后应有正确的原始输入和 reference witness；没有 CNN 拟合质量结论。共享 conditioning 的接口变更是第一处需独立审阅的公共风险点。

### P2：raw encoder、真实读取与 encoder-only 编译（AC9–AC11）

- [x] 新增 `methods/metal/spatial_encoder.py` 与 `asset_read.py`；按 design §8.2 实现五类 stem、融合/trunk、共享 mip block、Detail/Context heads、全局 stride phase 与 halo 推导。
- [ ] `asset.py` 接入共享 raw tile，取消新主路径的 `variant_scale_bias`。修正后的 summary 仅作为独立 control，使用同一个 read-plan 和量化/部署条件。
- [x] `asset_cook.py` 共用 encoder hierarchy，按层流式生成最终 DDS；训练主/paired 只编码一次；禁止把 raw coarse mip 或过期 learned feature 代入。
- [x] `model.py`、`evaluator.py`、`runtime.py` 更新 prepare 输入的 `wo3+Jacobian4+fracLOD`、FP16 program/prepared STE、真实量化四邻点读取，保持 prepare/evaluate 网络宽度。
- [x] `method.py` 更新 descriptor 的资源依赖、parameter groups、objective/phase、state export/restore 和 runtime 参数清单。移除 fixed asset count；C6 允许已支持 schema 的未见 source snapshot cook。
- [x] 新增 `tests/unit/test_metal_spatial_encoder.py`、`test_metal_asset_read.py`，扩展 `test_metal_budgeted_asset_cook.py`、`test_metal_budgeted_method.py`、`test_metal_budgeted_runtime.py`：完整图/tile、奇数/非方/1×N、repeat/clamp seam、不同 slot 分辨率与变换、感受野、亚 texel、量化前后、Context H(l+2)、未见 source 无优化。
- [ ] 扩展 GPU `test_metal_resident_sampling.py`、`test_metal_budgeted_model.py`：有效 stem/trunk/head 的 response 梯度、本 step 复用、下一步不使用 stale graph、资源预算和异常释放。动态缺图的 DDP unused group 行为沿用 engine 机制；无 Linux 实机则只登记待验。

P2 使用新方法 identity 和 fresh 初始化；不能把旧 checkpoint reshape 成新网络。不以质量先提高为原始 encoder 的实现前置。

### P3：C1–C6、Slang 与 layout（AC12/AC15）

- [x] `compiler.py` 修复 C1，以 initial+零 delta 保持第 0 步等价；control domain 和 delta 半径显式给出。
- [x] Python `evaluator.py` 与 `metal_budgeted_common.slang` 使用 design §8.5 的连续正 Z frame；在 P2 新训练前完成 C4。
- [x] `metal_budgeted_common.slang` 稳定 softplus；`metal_budgeted_evaluator.slang` 与外层 `metal_budgeted.slang` 传播有效性，C2/C3 分别有独立 oracle 和失败注入 witness。
- [x] `evaluator.py`/`sampler.py`/`model.py` 实现独立 proposal parameter head 与 proposal frames；对应 `metal_budgeted_asset.slang`、`metal_budgeted_sampler.slang`、common/pack 同步。C5 的 reverse 密度与独立反向 prepare 对照。
- [x] 新增 `learning/abi/metal_budgeted_layout_v2.json`，更新 `profile.py`、`runtime.py`、`tools/learning/generate_metal_budgeted_layout.py` 和生成的 Slang。按 design §9 登记 176 B、7,664 prepare MAC、11,392 evaluate MAC、每 UV 组最多六次 reads（全资产上限 54）；program 导出的 REVERSE_PDF 与 descriptor 一致。
- [x] 新增 `tests/unit/test_metal_model_correctness.py`，扩展 GPU `test_metal_budgeted_sampler.py`、`test_metal_budgeted_runtime_package.py`。softplus 正常数尾部不能被全局 atol 吞掉；frame 用独立正交/连续/主轴不变量；validity 比较标志；reverse 不能只用 frozen-state 双方 parity。
- [x] `test_metal_budgeted_profile.py` 覆盖新状态字段非重叠、真实成本和生成文件同步；既有 Metal 当前入口/测试迁到新配置。历史 v3–v6 的原始 identity 和报告不改写为新结果。

实施顺序为 P1 → P2/P3 → 新训练；P2 的构造/读取测试可以先于 P3 shader 完成。C4 必须在新表示训练前，C2/C3/C5 必须在部署或 sampler 结论前，C6 必须在 encoder-only 生命周期交付前；C1 在任何对应 control 前。不把 C5 的质量调优设为 D1 evaluator 研究前置。

### P4：配置、诊断与阶段收尾（AC4/AC14）

- [x] 更新 `configs/training/methods/metal.yaml` 的当前 profile/context；新增 `recipes/metal-spatial-pilot.yaml` 与三个 `runs/metal-spatial-probe-{tungsten,bronze-scratched,steel-painted-cracked}.yaml`。source locator 复用当前三个 data 文件，以新配方显式覆盖 filtering/adapter identity；不沿用旧实验 recipe 名称。
- [ ] 新增 matched summary-control 配方/三个 run，所有 source/query/下游/量化/预算与 encoder 一致；trace 记录 feature、latent、prepared、core/gate/correction。
- [x] 新增 `runs/metal-spatial-stage-eval.yaml`，继承一个完整的新 run 配置，仅覆盖 `hooks.visual_eval.enabled=true`、reference_spp=128 和当前场景显示设置。独立 eval 只取该文件的视觉设置，source/model 仍来自 checkpoint。
- [ ] 在 `method.py` 与 `evaluator.py` 中提供分支观测及 readout 冻结职责；D2 positive/signed head 按 design §8.7，共用公共 phase/objective/checkpoint 生命周期，不写第二套训练入口。
- [x] 日常训练/验证/导出继续走 `python -m ncls`。一次性 D0 观测、信号读出或成本分析脚本只放本任务 `scratch/`，调用公开 Method/reference/session API，GT 在线，禁止保存 batch。
- [ ] 按冻结 diagnostic 配方记录配置、种子、真实工作量、指标、图像和当前 checkpoint。通过 RunPaths 写 `outputs/`；跨 run 分析放任务 `research/` 或 `artifacts/`，不改旧成果。
- [ ] 阶段末一次 package/Slang/同场景 viewer 与查询成本检查；保留完整 ready capture 身份、线性输出和显示设置。更新当前规范中的必要输入/资源/ABI 合同与根 TESTING.md，单次结果不写入 PRD。

## 3. 第一轮诊断的冻结计划

以下是待执行的 diagnostic 计划，不是 formal 或本轮实测。首次代码实施的强制目标是 P1–P3 正确性和生命周期；质量实验达 cap 即收尾记录，不自动跑连续变体。

| 项目 | 预定配置与工作量 | 用途/停止条件 |
|---|---|---|
| D0 | Tungsten、划痕青铜、开裂钢各一个当前默认 state；非对称合成 source 作独立正确性 witness；0/1/4 texel、整数与预定 fractional LOD、point/16/64 footprint 点 | 不训练；坐标/值域/目标/读取差异阻塞受影响比较 |
| D1 首轮 | 每 source 两个配置：raw encoder、修正 summary；每 run 512 optimizer steps、B=128、direction_count=1、主/paired，共 6 runs；FP32 master，部署 QAT 从比较起点启用；无自动 sampler 质量训练 | 分辨 encoder 效果；proposal head 参与结构正确性，不用 sampler loss 混入表示归因 |
| D1 采样 | balanced-four-mode 方向；footprint 0/1/4 分层；evaluation_samples=1、footprint_samples=16；同 pair 共用方向/seed/filter random | 每 run 已接受 target 对应 512×128×2×16=2,097,152 次点求值调用；calibration/validation/rejection 单列实际数 |
| D1 优化 | Adam，betas=(0.9,0.99)、eps=1e−8、weight_decay=1e−6；cosine 3e−4→5e−5；model/train seed 2026090501，validation 2026090502，final query 2026090503 | 所有 matched 配置一致；不继承旧 optimizer/calibration |
| calibration/validation | train-only calibration 16,384 行，当前 median/p95 appearance 校准；每128步数值 validation 8 batches，末次32 batches | 校准只用于 loss，不标准化 raw texture。held-out tile 与训练 tile 的完整 receptive field 分离，pair/halo 不跨 split |
| tile/residency | 等尺寸默认 core128/halo64，实际 halo 由网络/坐标推导；单 source cohort，每步最多两个 tile bundle；共享 residency 8192 MiB；ready=1、reference_batch_steps=1 | smoke 观测峰值/吞吐和 ETA；超预算停止分类，不能悄悄缩 raw 感受野 |
| D1b 分支 | raw/summary/feature/latent/prepared 读出解释信号；必要时 encoder latent 初始化的 bounded refinement、固定 decoder 的自由 code control 各最多256步/冻结 source | 不作为 encoder-only 必经步骤。执行前记录分支和 CPU tile 集；无收益不等于带宽不足 |
| D2 分支 | 正确输入上检查 loss/correction；划痕青铜、开裂钢各一对 positive/signed readout，最多256步/配置，其余 D1 条件固定 | head 外权重和 gate 输出冻结；R 由 train-only residual calibration 固定；无收益停止，不自动放开 trunk 或增大 R |

512 步、16 footprint 点和 tile/residency 数值是首轮诊断的成本选择，未从旧质量数字推导成功门槛；formal 比较需要另行冻结收敛、工作量和评测合同。D1b/D2 是预列条件分支，不全部列为代码验收前置，不自动排成长训练队列。

每 source 的独立统计单元为冻结 UV tile/query block，方向/通道是块内样本；共享 texture-set 的 source 保持分组。报告 appearance/spatial/peak、符号/差分幅度、分支 RMS、SNORM 占用、梯度与饱和、真实 cook/prepare/evaluate 成本。单 seed CI 不覆盖训练 seed 方差。未见 tile/方向/参数状态/texture-set 分开命名，六个单 source run 不能宣称通过多材质泛化。

## 4. 实施后执行的真实命令

以下保留计划命令；本次实际执行的子集、真实路径与结果见 research/implementation-validation.md。尚未运行的项目不计为通过，不能换回旧 CLI 冒充新 profile 验证。Python 使用 `neural-shading`，GPU 测试通过当前 Python runtime launcher。

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_conditioning_resources.py tests/unit/test_training_batch.py tests/unit/test_online_training_producer.py tests/unit/test_online_data_session.py -q
conda run -n neural-shading python -m pytest tests/unit/test_metal_spatial_inputs.py tests/unit/test_metal_spatial_encoder.py tests/unit/test_metal_asset_read.py tests/unit/test_metal_model_correctness.py -q
conda run -n neural-shading python -m pytest tests/unit/test_metal_budgeted_asset_cook.py tests/unit/test_metal_budgeted_model.py tests/unit/test_metal_budgeted_method.py tests/unit/test_metal_budgeted_runtime.py tests/unit/test_metal_budgeted_profile.py -q
conda run -n neural-shading python -m tools.learning.generate_metal_budgeted_layout --check
conda run -n neural-shading python -m ncls.runtime --device 0 -- -m pytest tests/gpu/test_metal_spatial_reference.py tests/gpu/test_metal_resident_sampling.py tests/gpu/test_metal_budgeted_model.py tests/gpu/test_metal_budgeted_sampler.py tests/gpu/test_metal_budgeted_runtime_package.py -q
```

共享 batch/API 修改后，收尾运行一次全量 unit，并按当前 TESTING.md 补 Nvidia GPU/公共 query 回归；已经通过的范围无新变化时不反复全量跑。

```powershell
conda run -n neural-shading python -m pytest tests/unit -q
conda run -n neural-shading python -m ncls train 0 --config configs/training/runs/metal-spatial-probe-bronze-scratched.yaml --stop-at-step 0
```

从命令返回的实际 run 目录取得 `checkpoints/latest.pt`，赋给 `$spatialCheckpoint`；必须为上述输出的真实路径，不选择任意“最新”或旧 artifacts checkpoint。依次检查初始化导出、0→2 step 恢复、数值验证与当前图像入口：

```powershell
conda run -n neural-shading python -m ncls export $spatialCheckpoint --material-index 0
conda run -n neural-shading python -m ncls train 0 --config configs/training/runs/metal-spatial-probe-bronze-scratched.yaml --resume $spatialCheckpoint --stop-at-step 2
conda run -n neural-shading python -m ncls validate $spatialCheckpoint --batches 1 --device 0
conda run -n neural-shading python -m ncls eval $spatialCheckpoint --config configs/training/runs/metal-spatial-stage-eval.yaml --device 0
```

新配置中间训练默认关闭 visual hook，阶段收尾用带 enabled=true override 的独立 eval；没有 override 的 eval 会沿用 checkpoint 中关闭的设置，不能把正常返回 0 当作已经出图。检查同步 hook 生命周期时只在 smoke 开启。0/2 step 图像只证明导出链路，不参加质量比较。诊断 run 用同一 `train 0 --config ...`，按 §3 cap；本轮不启动。

shader/layout 修改后阶段末构建并核对 overlay 干净：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
git diff --check
git -C external/Falcor status --short
```

Linux/NCCL 的资源生命周期和缺失语义组梯度须目标实机验证；本机不能用 mock 宣称完成。待运行命令与已验证范围写入根 TESTING.md，不为本任务启动跨机队列。

## 5. 回退与完成判据

- P1 公共 conditioning 与 Metal 网络分别形成可审阅修改单元。只回退本任务对应 hunks，不重置架构提交、其它任务或未跟踪资料；本轮不提前创建 worktree/branch 或修改任务指针。
- P2/P3 更改输入/target/数学/ABI 后使用新 identity 和 fresh run，旧成果原位保留。不得放宽 parity tolerance 掩盖读法差异。
- correctness/protocol/resource defect 先停止并修复受影响路径；正常 empirical outcome 直接报告。预算、source 或模型实质扩大时回 planning，不能凭新文件名扩大授权。
- 实施修改覆盖本任务所列源码、测试、配置与必要规范；不改其它任务或来源不明文件，不 push。commit 按有效任务授权与质量门处理，不凭本条推定已获预授权。
- 规划收尾检查 PRD 需求/验收映射、本地链接、任务 JSON、陈旧架构阻塞和未完成占位符。代码交付时以 AC9–AC15 逐项附实际证据；未执行实验/平台单列，不能把规划勾选当模型验收通过。

本轮规划静态检查：7 份 Markdown、76 个本地链接目标均存在，未发现未完成占位符或行尾空白；task.json 可解析，状态为 planning，当前与历史基线分别保存。PRD 已按目标/背景/需求/验收/范围重新收敛并通读，R1–R9 和原 AC1–AC8 映射保留，代码 AC9–AC15 全部未勾选。已跟踪源码无 diff；这些检查只证明文档状态，不证明模型正确性或性能。
