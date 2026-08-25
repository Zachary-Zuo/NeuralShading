# P1 v2 实施计划：lobe-residual 候选、单一 Slang 后端与 PT 对照

本文是 [`p1_audit.md`](p1_audit.md) §5–§7 与 [`experiment_framework.md`](experiment_framework.md) §0.1 部署预算落地为工程任务的计划。它回答：新候选长什么样、每一步改哪些文件、每一步用什么测试判定完成、哪些决定还没做。所有行号对应 commit `9a458fa`。

## 0. 目标与完成判定

**目标**：在 LayerStack P1 v1 子语料（30 state，`data_id=0513d0c8…`）上，用一个满足 §0.1 全部软线的候选，同时做到：

| 判定项 | 门 | 依据 |
|---|---|---|
| Q1 质量 | test directional L1 state-median ≤ `0.045`（M1-M 水平）且 p95 ≤ `0.10`；15 个单层 state ≤ `0.013`；M2 失效的 4 个多层 state（`bd6de2…/6fff05…/4ebd92…/179606…`）各 ≤ `0.15` | `experiment_log.md` P1 v1；`p1_audit.md` §4.2 |
| Q2 尾部诚实 | 报告附 bootstrap CI、leave-one-state-out、最差 state 清单、signed 能量比、`E_core/E_ref` | framework §7、`p1_audit.md` §7 |
| C1 成本 | `C_eval ≤ 2e3` MAC、`C_prepare ≤ 1e4`、state ≤ 64 B、`B_asset ≤ 512 B`、evaluate 权重 ≤ 32 KB；由单元测试机械判定 | framework §0.1 |
| C2 实测 | RTX 4090、640×360 benchmark preset 下 prepare + lighting ≤ 1 ms；1080p 单灯 ≤ 2 ms | §0.1 工况 |
| S1 sampler | `sample/pdf` 与 `evaluate` 一致性、白炉、pdf 归一化在 GPU 测试中通过 | `scattering_backend.md` 一致性测试节 |
| E1 单一源 | 训练（SlangPy）、GPU 测试（Falcor Python）、viewer 三处 `#include` 同一份 `lobe_residual.slang`；Torch 实现只作 parity oracle | `p1_audit.md` §6.2 |
| V1 对照 | viewer 新增「PT + 源材质 vs PT + method material」比较模式，4 个失效 state 与 `6324e3…` 各出一张 capture | `p1_audit.md` §5.4 |

**非目标**：P2 全语料、spatial latent texture、cooperative vector、MERL/OpenPBR 族、`integrate_*` 专用积分器。这些等 P1 v2 通过 Q1/C1 后按 framework §6 推进。

## 1. 候选定义：`lobe-residual`（注册名 `lobe-residual-k2-v1`，family `m2b`）

### 1.1 数学形态

```text
z ∈ R^16                                   每 state 一个 autodecoder latent（P1 沿用 target-visible）
h, θ_1..θ_K = Prepare(z, φ(wo))            ≤64 宽 2 层 MLP；h ∈ R^8 供修正项，θ_k 为 K 个 LTC lobe 参数
f(wo, wi) = [ f_top(wo, wi; IR_0) + Σ_k A_k · D_k(wi; θ_k) / cos θ_i ] · exp(Δ(h, ψ(wi)))
```

- `f_top`：顶层界面精确 microfacet，参数直接来自 `LayerStackIR.interfaces[0]`，不学习；实现复用 `shaders/ncls/reference/interfaces.slang:191` `nclsEvalInterface`（Torch 侧 `torch_eval.py:249` `eval_direct_top_bsdf`）。只取反射分支（`NCLS_SAMPLE_REFLECTION`）；进入 coat 的能量由 residual lobe 表示。
- `D_k`：LTC 余弦分布，参数 `(inverseScaleX/Y, shearX/Y/Z, angle)` + RGB 幅值 `A_k`，即现有 `NclsLegacyLtcK2Lobe` 的 9 个有效 float；解码沿用 `torch_eval.py:38` `decode_ltc_residual`（`softplus / exp(clamp ±3) / 3·tanh / π·tanh`），**幅值非负由 softplus 保证，全程无 clamp**。
- `Δ`：可选乘性 log 修正，`ψ(wi)` 为 6 维（`wi`、half-slope 2 维、`wi.z`），网络 `(8+6)→32→3`，输出 `clamp(·, −2, 2)`。配置轴 `correction ∈ {none, log32}`；`none` 即纯 Adapter 形态，`log32` 保留「evaluate 由 MLP 直接补全」的项目定位（[`../realtime_material_compilation.md`](../realtime_material_compilation.md) §「Neural evaluator 是目标表示的主体」），两者 matched 对照后决定是否保留。
- `sample/pdf`：混合 proposal `p = w_0 · p_top + Σ_k w_k · p_k`，`p_top` 为顶层 VNDF 反射采样（`sampling.slang:92` `nclsSampleGgxVisibleNormal` + `interfaces.slang:144` `nclsInterfacePdfLocal`），`p_k = D_k` 自身（LTC 逆变换：`q̂ ~ cosine`，`wi = normalize(M_k q̂)`，pdf 即 `det·max(q_z,0)/(π|q|⁴)`，也就是 `legacy_ltc_k2.slang:40` 已算出的 basis）。权重 `w` 由 `prepare` 按各项估计能量归一。Δ 有界，所以 pdf 不含 Δ 仍无偏。

### 1.2 成本自检（对照 framework §0.1）

| 量 | K=2, `none` | K=2, `log32` | K=3, `log32`（研究消融） | 软线 |
|---|---:|---:|---:|---|
| `C_prepare` | `(16+7)×64 + 64×64 + 64×(8+18) ≈ 7.2k` MAC | 同 | `≈ 7.8k` | ≤ 1e4 ✓ |
| `C_eval` | `f_top ≈ 150` FLOP + `2 × 60` | `+ (14×32 + 32×3) ≈ 550` MAC | `≈ 700` | ≤ 2e3 ✓ |
| state | slot(4) + 2×9 half(36) + flags(8) = 48 B | + h 8 half(16) = **64 B** | 82 B ✗ | ≤ 64 B |
| `B_asset` | z 64 B + top IR 64 B = 128 B | 同 | 同 | ≤ 512 B ✓ |
| `B_shared` | ≈ 7.5k 参数 = 30 KB fp32 / 15 KB fp16 | ≈ 8k = 16 KB fp16 | — | evaluate 权重 ≤ 32 KB ✓ |

结论：**部署档固定 K=2**；K=3 只作研究消融，注册表标「非部署候选」。顶层 IR 不进 state，state 只存 `materialSlot`，evaluate 时从 `CompiledMaterial` 缓冲随机读取一块 64 B（合同允许的固定读取数）。

### 1.3 与 P1 v1 的差异清单

| | M2（`analytic-residual`） | `lobe-residual` |
|---|---|---|
| 残差 | signed MLP 输出 + `clamp(core+Δ, 0)` | 非负 LTC lobe + 有界乘性修正 |
| evaluate | 256 宽 FiLM trunk（`8.6e5` MAC） | 解析求和（≤ `7e2` MAC） |
| state | `float[256]` = 1 KB | 64 B |
| sample/pdf | 无 | 混合 VNDF + LTC，精确 pdf |
| 条件化 | condition 向量烘焙 19.5 KB | latent 直接进 prepare，无烘焙 |
| Slang | 手写复刻 + 手工偏移 | 单一源，Python 生成 layout |

## 2. 现状中可复用与必须新建的部件

复用（已存在、有测试）：`interfaces.slang` 四种界面的 evaluate/pdf/sample 三件套（`:91/:144/:223`）；`sampling.slang` 各向异性 VNDF（`:92`）；`legacy_ltc_k2.slang` 的 LTC evaluate 与 `INclsScatteringBackend` 合同实现范例（`:103-160`）；`torch_eval.py` 的 `decode_ltc_residual/eval_ltc_residual/eval_direct_top_bsdf`；`source_adapters/layer_stack_direct_top.py` 的 `fit_direct_top_state`；`tests/gpu/kernels/legacy_ltc_k2.cs.slang:10-20` 的 `ISampleGenerator` 测试实现；`p1_audit.py` 的 signed 能量 / 死区 / tail 诊断；`film_m1.py:44-71` 的 `_canonical_json/_write_json/_git_commit`。

蓝本但不可直接用：`p1_compiler.slang`（零调用者、GRU 数据相关循环、手工偏移）——只借用 25 维 IR 特征函数与 18 维 raw→lobe 解码。

必须新建（按 §3 分期）：LTC lobe 的 sample/pdf（Slang + Torch 两侧都没有）；`NclsRng`↔`ISampleGenerator` 适配；Falcor-free 的后端核心文件；SlangPy 训练路径（仓库零依赖、零 `[Differentiable]` 代码）；通用 bundle exporter；viewer 的 realtime 加载路径、泛型 pass、PT+method family；`sample/pdf` 的任何测试（目前零覆盖）。

## 3. 分期任务

每期列出改动文件、验收测试、执行地点（本地 = 静态分析/编写，远程 = 运行）。任务前缀：P=Python、S=Slang、V=viewer、D=文档。

### Phase 1 — Torch 参考实现与质量信号（先回答「形态对不对」）

Torch 版只作 parity oracle，Phase 3 之后不再承担训练。它几乎全由现成件拼成，所以先用它拿质量信号，避免把 SlangPy 0→1 的工程风险挡在质量问题前面。

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| P1.1 | 新 pipeline 家族 `m2b`：`LobeResidualPipeline`，descriptor `model.representation="analytic-core-lobe-residual-v1"`、`architecture="lobe-residual-prepare-mlp-v1"`、`data.source_adapter="layer-stack-direct-top-v1"`；`predict_f` 返回 `f_top + Σ lobe`（`log32` 时再乘 `exp(Δ)`），**无 clamp**；`fit_training_state` 复用 `fit_direct_top_state`；`training_loss` 复用 `p1-appearance-v3` 四项（`pipelines/p1_evaluator.py:255-304`），去掉 `:263` 的 clamp | `src/ncls/learning/pipelines/lobe_residual.py`（新）、`pipelines/__init__.py:5` 注册 | `tests/unit/test_lobe_residual_pipeline.py`：duck-typed Store + monkeypatch `direct_top_bsdf`（照 `test_pipeline_contract.py:112-175`）；断言输出非负、无 clamp 路径、descriptor sha 稳定 |
| P1.2 | 模型 `LobeResidualModel`：`nn.Embedding(30,16)` + prepare MLP `(16+7)→64→64→(8+9K+3K)` SiLU + 可选 Δ 网 `(8+6)→32→3`；lobe evaluate 调 `eval_ltc_residual`，core 调 `eval_direct_top_bsdf` | `src/ncls/learning/models/lobe_residual.py`（新） | 同上；`parameter_costs()` 返回与 §1.2 一致的 `C_prepare_macs/C_eval_macs/state_bytes_per_pixel/B_asset/B_shared` |
| P1.3 | Torch 侧 LTC lobe `sample/pdf` + 混合 proposal（为 Phase 2 parity 与 S1 准备） | `src/ncls/core/representations/legacy_ltc_k2/torch_eval.py` 追加 `sample_ltc_lobe / ltc_lobe_pdf / mixture_pdf` | `tests/unit/test_ltc_sampling.py`：MC 估计 `∫pdf dω = 1 ± 1e-2`；`pdf(sample(u)) == sample.pdf`；importance 估计的 lobe 能量与求积一致 |
| P1.4 | 部署预算单元门：遍历注册表中 `deployment_candidate=True` 的 pipeline，断言 `parameter_costs()` 满足 §0.1 全部软线 | `tests/unit/test_deployment_budget.py`（新）；descriptor 增加 `runtime.deployment_candidate` 需同步 `schemas/learning_pipeline_v1.schema.json` 与 `base.py:36-59` 的精确字段集校验 | M1/M2 三档必须被标 `False` 且测试通过；`lobe-residual-k2` 为 `True` 且通过 |
| P1.5 | checkpoint tail guard：`TrainingConfig` 增字段 `checkpoint_selection: "median_then_p95" \| "tail_guard"`（默认旧值，`from_dict` 拒绝未知字段所以要同步 `training_config_v1.schema.json`）；runner `:277` 的 `(median, p95)` 元组改为按策略分支；新增 `configs/evaluation/quality-v2.json`（只改 `checkpoint_selection` 块，指标定义不变，`quality.py:49-53` 校验分支化） | `training/config.py`、`training/runner.py:276-302`、`evaluation/quality.py:17-65` | `tests/unit/test_training_config.py` 增策略往返；用 P1 v1 M2-S 的 validation history 回放断言选到 step 7500 而非 4500 |
| P1.6 | 审计工具泛化：`p1_audit.py:540-560` 的 `is_m2` 改为探测 pipeline 可选方法 `core_f(model, batch, store, device)`，有则算 `E_core/E_ref`；死区分支只在 pipeline 声明 `has_signed_residual` 时执行 | `evaluation/p1_audit.py`、`pipelines/base.py` 增可选协议 | `tests/unit/test_p1_audit.py` 增 `core_f` 探测用例 |
| P1.7 | 配置：`configs/learning/lobe-residual-k2-v1.json`（`correction:none`）、`lobe-residual-k2-log32-v1.json`、`lobe-residual-k3-log32-v1.json`（研究）、`smoke/lobe-residual-k2-p1-smoke.json`；训练调度沿用 S 档（bs 16、lr 3e-4、25k、patience 6、`tail_guard`） | `configs/learning/` | `test_training_config.py:70-76` 自动覆盖 |
| P1.8 | 远程：三个配置各跑 seed `20260824`；`ncls learn evaluate` test/adversarial/dense；`audit-p1` 出 signed 能量与 core coverage；`compare` 对 M1-M | 远程 | Q1 判定。若 K=2 `none` 未达 p95 ≤ 0.10 而 `log32` 达到 → 修正项保留；两者都未达 → 看最差 state 的 `E_core/E_ref` 与 lobe 承担能量，决定是否需要 K=3 或 lobe 型别（GGX 型 vs LTC 型）扩展，再进 Phase 2 |

Phase 1 出口：Q1 通过或明确的失败归因；`tests/unit` 全绿；注册表登记三个 run（标注部署候选与否）。

### Phase 2 — 单一 Slang 后端（evaluate/sample/pdf）与 GPU parity

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| S2.1 | **Falcor-free 核心**：`lobe_residual_core.slang` 定义 `NclsLobeResidualState`（64 B 布局见 §1.2）、`NclsLobeResidualParams`（权重偏移表结构，由 Python 写入 cbuffer/StructuredBuffer，Slang 不再手写偏移常量）、`nclsLobeResidualPrepare(z, woLocal, params)`、`EvaluateF(state, top, woLocal, wiLocal, params)`、`Pdf(...)`、`Sample(state, top, woLocal, float3 u, params)`（随机数由调用者传入）。只 `#include` `contracts/layer_stack_ir.slang`、`reference/interfaces.slang`、`reference/sampling.slang` | `shaders/ncls/backends/lobe_residual/lobe_residual_core.slang`（新） | GPU 编译冒烟 + parity（S2.5） |
| S2.2 | `interfaces.slang` 的 `nclsSampleInterface(:223)` 增加接受预抽随机数 `float3 u` 的重载，原 `inout NclsRng` 版本改为薄包装；`sampling.slang:92` 同样加 `float2 u` 重载 | `shaders/ncls/reference/interfaces.slang`、`sampling.slang` | 现有 reference GPU 测试不变（`tests/gpu`、`tests/integration/reference`）；random walk 采集结果 hash 不变 |
| S2.3 | LTC lobe 采样与 pdf：`nclsLtcLobeSample(lobe, float2 u)`（上三角矩阵闭式求逆）、`nclsLtcLobePdf(lobe, wiLocal)`；混合 proposal 的选择与 pdf 合成 | `lobe_residual_core.slang` | S2.5 中与 P1.3 Torch 版逐点 parity |
| S2.4 | **合同包装**：`lobe_residual.slang` `#include` core 与 `contracts/scattering_backend.slang`，实现 `struct LobeResidualBackend : INclsScatteringBackend`（`CompiledMaterial = {NclsLayerInterfaceIR top; float latent[16];}`，`State` 为 64 B 结构 + context），`evaluate/sample/pdf` 从 `ISampleGenerator` 取 `sampleNext2D/1D` 后调 core | `shaders/ncls/backends/lobe_residual/lobe_residual.slang`（新） | `tests/gpu/test_scattering_contract_gpu.py` 同型的编译冒烟：在锁定 Slang 2024.1.34 上 conform 成功 |
| S2.5 | GPU parity 与 sampler 测试：kernel 提供 `evaluateDirect / evaluateThroughContract / pdfThroughContract / sampleThroughContract` 四个入口；Python 打包 state 与 params | `tests/gpu/kernels/lobe_residual.cs.slang`、`tests/gpu/test_lobe_residual_gpu.py`（照 `test_legacy_ltc_k2_gpu.py` 样板） | evaluate parity `rtol 2e-5`；`pdf(sample.wi) == sample.pdf`；白炉：`A_k` 设为使 lobe 反照率为 1、`f_top` 关闭时，`E[f·cos/pdf] = 1 ± 1e-2`；pdf MC 归一化；grazing/退化方向无 NaN |
| S2.6 | Python 侧 state/params 打包与 layout 生成：`pack_state`（64 B，`struct` 校验）、`build_params_layout(model) -> (np.ndarray, layout_json)`；layout JSON 是权重偏移的唯一来源，Slang 从 params 缓冲读偏移 | `src/ncls/core/representations/lobe_residual/{state.py,layout.py}`（新） | `tests/unit/test_lobe_residual_layout.py`：往返 + 与 core.slang 结构字段名一致（正则抽取，照 `test_film_m1_bundle.py:30-45`） |
| S2.7 | 生成合同若需新事件位（无预期）走 `abi/scattering_contract_v1.json` + `abi_layout.py`；本期预计不改 | — | `test_scattering_contract.py:111` 逐字节断言保持 |

Phase 2 出口：S1 通过；Torch 与 Slang 在 evaluate/pdf/sample 三条路径 parity；一份 core 源。

### Phase 3 — SlangPy 训练路径（同一源做梯度）

这是 0→1 的部分，先做一个独立 spike 决定可行性，再接入 runner。

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| P3.0 | **Spike（远程）**：在 `neural-shading` env 安装 `slangpy`（版本写入 `environment.yml` pip 段），用它编译 `lobe_residual_core.slang`（加 `[Differentiable]` 标注、`no_diff` 标记 IR/方向输入），对随机 batch 调 `bwd_diff(evaluate)` 得到对 params 与 latent 的梯度，与 Torch autograd 比 | `scripts/spike_slangpy_autodiff.py`（一次性，放 `scripts/`） | 梯度 `rtol 1e-3`；吞吐 ≥ Torch 版 0.5×（batch 16 group × 256 方向）；记录 slangpy 携带的 slang 版本与 2024.1.34 的语法差异清单 |
| P3.1 | 若 spike 通过：`LobeResidualPipeline.predict_f` 增 `backend="torch" \| "slangpy"` 开关；slangpy 分支把 `batch["wo"/"wi"/"state_index"]` 交给 SlangPy 模块，返回 `torch.Tensor`（带自定义 autograd Function 包装 `bwd_diff`）；loss/optimizer 不动 | `src/ncls/learning/pipelines/lobe_residual.py`、`src/ncls/learning/slang/session.py`（新，SlangPy 模块加载与缓存） | `tests/gpu/test_lobe_residual_slangpy.py`（marker `slangpy`）：同一 checkpoint 两个 backend 的 `predict_f` parity `rtol 1e-4`，一次训练 step 后权重差 `≤ 1e-4` |
| P3.2 | 训练配置增 `backend` 字段；远程用 slangpy backend 复跑 `lobe-residual-k2-*`，结果与 Torch 版 run 同表登记 | `configs/learning/*`、`training/config.py` | 两版 test median/p95 差异在 paired bootstrap CI 内 |
| P3.3 | 语法约束探针：`tests/gpu/test_lobe_residual_gpu.py` 中的 Falcor 编译 + P3.1 的 SlangPy 编译共用同一文件；任一失败即阻止提交 | 同上 | `p1_audit.md` §6.2 要求的「两个编译器同一源」探针 |

若 spike 失败（语法/性能/Torch 互操作任一不可接受）：记录原因到 `experiment_log.md`，Phase 3 降级为「Torch 训练 + 生成式 layout + GPU parity」，单一源目标保留在 evaluate/sample/pdf 与 viewer，训练梯度暂留 Torch。这是本计划最大的技术风险，由 spike 尽早暴露。

### Phase 4 — Bundle 与 viewer：realtime 加载、泛型 pass、PT + method 对照

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| P4.1 | 通用 exporter：`bundle/exporter.py` 抽出 manifest 组装、文件写入、hash、parity probe（`ncls.backend-parity-probe`）；`bundle/lobe_residual.py` 只提供 descriptor、params/layout、`compiled_materials/`（30 个 state 各 128 B，键为 LayerStack IR hash）与 `cost_claims`；`runtime_class="realtime"` 需 `manifest.py:68-69` 的 `is_complete_realtime_backend`（caps `1\|2\|4\|8\|16`）成立 | `src/ncls/bundle/{exporter.py,lobe_residual.py}`、`cli.py` 增 `bundle export-lobe-residual` | `tests/unit/test_bundle_export.py`（新）：export → `MethodBundle.open` → 校验通过；realtime 硬线由 Python 侧 `cost_claims` 校验函数判定（同一份数值来自 `method_bundle.md` 硬线表） |
| V4.2 | loader 去硬编码：`MethodBundle.cpp:72-73` 接受 `realtime`；`:76-94/:126-138/:192` 的 backend 专属检查移入按 `backend_id` 查表的 `BackendRegistry`（表项：shader 相对路径、prepare/evaluate/sample/pdf 入口、state stride 校验、layout 格式名）；realtime 时按硬线表校验 `cost_claims`；`kRequiredCapabilities` 对 deferred 为 `1\|2\|16`，启用 PT 模式再要求 `4\|8` | `apps/viewer/MethodBundle.{h,cpp}` | film-m1 diagnostic bundle 与新 realtime bundle 都能加载；硬线超标的 bundle 报具体原因 |
| V4.3 | 泛型 pass：`Prepare/Approximation/Parity.cs.slang` 改为 `#include NCLS_METHOD_BACKEND_HEADER` + `typedef NclsMethodBackend`，通过 `INclsScatteringBackend` 调 `prepare/evaluate`；`NclsViewer.cpp:319-327` 改为按选中方法用 defines 创建 pass（照 `:607-624` reference pass 的做法）；state 缓冲按 descriptor stride 分配（`:537-543` 已支持）；`CMakeLists.txt:27` 增 `shaders/ncls/backends/lobe_residual/*.slang` 与 `contracts/`、`reference/` 依赖 | `apps/viewer/shaders/*.cs.slang`、`NclsViewer.cpp`、`CMakeLists.txt` | 加载期 parity probe 通过；`scripts/benchmark_viewer.ps1` 自动挑 realtime bundle 的路径（`:53-66`）首次可用 |
| V4.4 | PT + method material：`ReferencePathTracer.cs.slang` 增 family 4，四个 dispatch 点（`:276/:523/:587/:632`）调 `NclsMethodBackend`；`NclsRng` 适配为 `ISampleGenerator`（`struct NclsRngSampleGenerator : ISampleGenerator { NclsRng rng; [mutating] uint next(); }`）；C++ 增第二组累积纹理与 `gFamilyOverride`，`Composite.cs.slang` 增 mode 4「PT reference / PT method split」；`allMaterialsSupportedBy(:752-766)` 改为「所有 slot 的 IR hash ∈ bundle `compiled_materials`」 | `apps/viewer/shaders/{ReferencePathTracer,Composite}.cs.slang`、`NclsViewer.cpp:607-650, 752-766, 1008-1020` | 同灯、同反弹深度、同 spp 下左右差图只含表示误差 + MC 噪声；对 `6324e3…` 与 4 个失效 state 各出 capture |
| V4.5 | benchmark：`configs/viewer-benchmark-v1.json` 三机位跑 realtime bundle；另加 1080p 单灯 preset | `scripts/benchmark_viewer.ps1`、`configs/` | C2 判定；结果 `(method_id, scene, device)` 进 `artifacts/`，注册表登记 |
| D4.6 | 合同文档与代码命名对齐：`scattering_backend.md:189-191` 的 `state_storage_mode/supported_capabilities/supported_material_ir` 改为代码名 `state_storage/capabilities/supported_ir_ids`；`method_bundle.md:60` 删除不存在的 `compiler.weight_files`；`:161` 的「尚未实现校验」改为实现事实 | `docs/contracts/*.md` | 文档评审 |

### Phase 5 — P1 v2 收口

| # | 任务 | 验收 |
|---|---|---|
| 5.1 | 注册表：`lobe-residual-k2-{none,log32}`、`k3` 三行 + Slang/viewer 实测行；结论列写明部署候选与 Q1/C2 判定 | `experiment_log.md` |
| 5.2 | `model_candidates.md` §3 M2 改写为 `lobe-residual` 的正式定义（§1.1），删除 signed-residual 描述；§1.5 增「部署档」定义 | 文档 |
| 5.3 | `p1_audit.md` §7 关闭已完成项；`experiment_framework.md` §6 表 P1 行更新主要候选 | 文档 |
| 5.4 | 记忆/TESTING：`TESTING.md` 增本计划各期命令（unit / gpu / slangpy / viewer benchmark） | 远程按 TESTING.md 复核 |

## 4. 依赖关系与并行性

```text
P1.1–P1.7 (Torch, 本地可写) ──► P1.8 (远程) ──► Q1 判定
        │
        ├─► S2.1–S2.6 (Slang core + parity)  可与 P1.8 并行；S2.3 依赖 P1.3 的 Torch 对偶
        │            │
        │            └─► P3.0 spike (远程)  ──► P3.1–P3.3
        │
        └─► P4.1 exporter  ──► V4.2 loader ──► V4.3 泛型 pass ──► V4.4 PT family ──► V4.5 benchmark
                                                    (V4.2–V4.4 依赖 S2.4 的合同包装存在)
```

- 本地（静态）可直接推进：P1.1–P1.7、S2.1–S2.4、S2.6、P4.1、V4.2–V4.4 的代码与 D4.6。
- 必须远程：P1.8、S2.5、P3.0–P3.3、V4.5，以及每期的 pytest/GPU 回归。
- 规模：Phase 1 小（现成件拼装，主要是 pipeline 与测试）；Phase 2 中（sampler 与 parity 测试是新内容）；Phase 3 spike 小、接入中，但风险最高；Phase 4 大（viewer 五处硬编码、PT family、第二累积链）；Phase 5 小。

## 5. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| SlangPy 携带的 slang 与 Falcor 8.0 的 2024.1.34 语法不兼容 | E1 单一源无法同时训练与部署 | P3.0 spike 先行；core 文件只用 2024.1.34 已验证的写法（固定数组、`typedef` 绑定 associated type、`[unroll]`），新特性一律不用；P3.3 双编译探针常驻 |
| LTC 型 lobe 表达不了 M2 失效的 4 个 state（coat 下 base 的多次散射形态） | Q1 p95 不达标 | Phase 1 用 `E_core/E_ref` 与 lobe 承担能量分解失败原因；备选：lobe 型别扩展为 GGX-VNDF 型（`interfaces.slang` 三件套直接可用）或 K=3 研究档 |
| 64 B state 在 `log32` 下刚好卡线，half 打包引入精度误差 | parity 容差 | S2.5 用 half 打包后的 state 做 parity，容差按 half 量级单列；`none` 配置 48 B 留余量 |
| viewer 改造范围大，容易把 film-m1 diagnostic 路径改坏 | 回归 | V4.2 保留 film-m1 表项，加载测试双 bundle；泛型 pass 先在 film-m1 上验证再切新后端 |
| PT + method 的第二累积链把左右噪声关联/去关联处理错 | 差图误判 | 左右使用独立 seed 流、相同 spp；capture manifest 记录两侧 `estimated_mean_relative_standard_error` |
| 30-state 上的 p95 本身不稳定（framework §7） | Q1 判定过拟合到 P1 子集 | 报告 CI 与 leave-one-out；Q1 的 p95 门只作 P1 selection 判定，P2 用 ≥ 50 state 重判 |

## 6. 已定事项（2026-08-25）

1. Phase 1 保留 Torch 参考实现，作为 Phase 2/3 的 parity oracle；Phase 3 后不再承担训练。
2. `correction` 默认 `none`（48 B state），`log32` 只作 matched 对照。
3. PT + method 做完整第二累积链（V4.4），不做单帧折中。
4. SlangPy 固定版本写入 `environment.yml`；不升级 Falcor 的 slang。
5. 代码风格：实现以简洁为先——复用现成件（`torch_eval.py`、`interfaces.slang`、`p1-appearance-v3` loss 抽函数而非复制），只暴露 `K` 与 `correction` 两个配置轴，不为未来阶段预留抽象；每个新文件控制在 200 行内。
