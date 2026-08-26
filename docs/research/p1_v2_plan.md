# P1 v2 实施计划：lobe-residual 候选、单一 Slang 后端与 PT 对照

本文是 [`p1_audit.md`](p1_audit.md) §5–§7 与 [`experiment_framework.md`](experiment_framework.md) §0.1 部署预算落地为工程任务的计划。它回答：新候选长什么样、每一步改哪些文件、每一步用什么测试判定完成、哪些决定已经做了。所有行号对应 commit `9a458fa`。

## 0. 目标与完成判定

**目标**：在 LayerStack P1 v1 子语料（30 state，`data_id=0513d0c8…`）上，用一个满足 §0.1 全部软线的候选，同时做到：

| 判定项 | 门 | 依据 |
|---|---|---|
| R1 复现 | 方法结构/方向/输出/loss 与对应一手定义或已登记 adaptation 一致；训练全程有限、validation 相对初始化改善、后期无可信发散，checkpoint 可恢复；最终绝对质量只报告 | `08-25-03-neural-baseline-and-candidate` PRD 与 method correspondence |
| Q2 尾部诚实 | 报告附 bootstrap CI、leave-one-state-out、最差 state 清单、signed 能量比、`E_core/E_ref` | framework §7、`p1_audit.md` §7 |
| C1 成本 | `C_eval ≤ 2e3` MAC、`C_prepare ≤ 1e4`、state ≤ 64 B、`B_asset ≤ 512 B`、evaluate 权重 ≤ 32 KB；由单元测试机械判定 | framework §0.1 |
| C2 实测 | RTX 4090、640×360 benchmark preset 下 prepare + lighting ≤ 1 ms；1080p 单灯 ≤ 2 ms | §0.1 工况 |
| S1 sampler | `sample/pdf` 与 `evaluate` 一致性、白炉、pdf 归一化在 GPU 测试中通过 | `scattering_backend.md` 一致性测试节 |
| E1 单一源 | 训练（SlangPy）、GPU 测试（Falcor Python）、viewer 三处 `#include` 同一份 `lobe_residual_core.slang`；不写第二套模型前向，现有 `torch_eval.py` 只作 parity oracle | `p1_audit.md` §6.2 |
| V1 对照 | viewer 新增「PT + 源材质 vs PT + method material」比较模式，4 个失效 state 与 `6324e3…` 各出一张 capture | `p1_audit.md` §5.4 |

**非目标**：P2 全语料、spatial latent texture、cooperative vector、MERL/OpenPBR 族、`integrate_*` 专用积分器。这些等 P1 v2 完成方法正确性、稳定收敛与 C1 后按 framework §6 推进。

## 1. 候选定义：`lobe-residual`（注册名 `lobe-residual-k2-v1`，family `m2b`）

### 1.1 数学形态

```text
z ∈ R^16                                   每 state 一个 autodecoder latent（P1 沿用 target-visible）
h, θ_1..θ_K = Prepare(z, φ(wo))            ≤64 宽 2 层 MLP；h ∈ R^8 供修正项，θ_k 为 K 个 LTC lobe 参数
f(wo, wi) = [ f_top(wo, wi; IR_0) + Σ_k A_k · D_k(wi; θ_k) / cos θ_i ] · exp(Δ(h, ψ(wi)))
```

- `f_top`：顶层界面精确 microfacet，参数直接来自 `LayerStackIR.interfaces[0]`，不学习；实现复用 `shaders/ncls/reference/interfaces.slang:191` `nclsEvalInterface`（Torch 侧 `torch_eval.py:249` `eval_direct_top_bsdf` 只作 parity oracle）。只取反射分支（`NCLS_SAMPLE_REFLECTION`）；进入 coat 的能量由 residual lobe 表示。
- `D_k`：LTC 余弦分布，参数 `(inverseScaleX/Y, shearX/Y/Z, angle)` + RGB 幅值 `A_k`，即现有 `NclsLegacyLtcK2Lobe` 的 9 个有效 float；解码沿用 `p1_compiler.slang:203-211` / `torch_eval.py:38` 的 `softplus / exp(clamp ±3) / 3·tanh / π·tanh`，**幅值非负由 softplus 保证，全程无 clamp**。
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
| 实现 | Torch 前向 + 手写 Slang 复刻 + 手工偏移 | 一份 Slang（SlangPy 训练、Falcor 部署），layout 由反射生成 |

## 2. 现状中可复用与必须新建的部件

复用（已存在、有测试）：`interfaces.slang` 四种界面的 evaluate/pdf/sample 三件套（`:91/:144/:223`）；`sampling.slang` 各向异性 VNDF（`:92`）；`legacy_ltc_k2.slang` 的 LTC evaluate 与 `INclsScatteringBackend` 合同实现范例（`:103-160`）；`torch_eval.py` 的 `decode_ltc_residual/eval_ltc_residual/eval_direct_top_bsdf`（只作 parity oracle）；`source_adapters/layer_stack_direct_top.py` 的 `fit_direct_top_state`；`tests/gpu/kernels/legacy_ltc_k2.cs.slang:10-20` 的 `ISampleGenerator` 测试实现；`p1_audit.py` 的 signed 能量 / 死区 / tail 诊断；`film_m1.py:44-71` 的 `_canonical_json/_write_json/_git_commit`。

蓝本但不可直接用：`p1_compiler.slang`（零调用者、GRU 数据相关循环、手工偏移）——只借用 25 维 IR 特征函数与 18 维 raw→lobe 解码。

必须新建（按 §3 分期）：LTC lobe 的 sample/pdf（Slang 侧）；`NclsRng`↔`ISampleGenerator` 适配；Falcor-free 的后端核心文件；SlangPy 训练路径（仓库零依赖、零 `[Differentiable]` 代码）；通用 bundle exporter；viewer 的 realtime 加载路径、泛型 pass、PT+method family；`sample/pdf` 的任何测试（目前零覆盖）。

## 3. 分期任务

每期列出改动文件、验收测试、执行地点（本地 = 静态分析/编写，远程 = 运行）。任务前缀：P=Python、S=Slang、V=viewer、D=文档。

### Phase 1 — SlangPy 可行性 spike（先排除最大风险）

单一 Slang 源的前提是 SlangPy 能对该源求梯度并与 Torch 互操作；仓库目前零 slangpy 依赖、零 `[Differentiable]` 代码，所以第一件事是 spike，不是写模型。

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| P1.0 | **远程 spike**：`environment.yml` pip 段固定 `slangpy==<版本>`；写最小 `[Differentiable]` 核（一个 64 宽 MLP + `nclsLegacyLtcK2LobeResponseCos` 型 evaluate），用 SlangPy 调 `bwd_diff` 得到对权重与 lobe 参数的梯度，与 Torch 对照：lobe 部分用 `torch_eval.py:117` `eval_ltc_residual` 的 autograd，MLP 部分用有限差分 | `scripts/spike_slangpy_autodiff.py`（一次性） | 梯度 `rtol 1e-3`；batch 16 group × 256 方向前向+反向吞吐 ≥ P1 v1 M1-S Torch 的 0.5×；记录 slangpy 携带的 slang 版本与 2024.1.34 的语法差异清单 |
| P1.1 | **备选，不默认执行**：spike 失败时改用 Torch 模型做训练路径（形态同 §1.1），Slang 只做部署 | — | 只有 spike 明确失败才启用，原因登记到 `experiment_log.md` |

### Phase 2 — 单一 Slang 后端与 SlangPy 训练接入

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| S2.1 | **Falcor-free 核心**：`lobe_residual_core.slang` 定义 `NclsLobeResidualState`（64 B 布局见 §1.2）、`NclsLobeResidualParams`（权重缓冲；偏移由 SlangPy 反射出的 layout 决定，Slang 内不写偏移常量）、`[Differentiable] nclsLobeResidualPrepare(z, woLocal, params)`、`[Differentiable] nclsLobeResidualEvaluateF(state, top, woLocal, wiLocal, params)`、`Pdf(...)`、`Sample(..., float3 u)`（随机数由调用者传入）；IR 与方向输入标 `no_diff`。只 `#include` `contracts/layer_stack_ir.slang`、`reference/interfaces.slang`、`reference/sampling.slang` | `shaders/ncls/backends/lobe_residual/lobe_residual_core.slang`（新） | Falcor 与 SlangPy 两个编译器编译同一文件（P2.7） |
| S2.2 | `interfaces.slang:223` `nclsSampleInterface` 与 `sampling.slang:92` 增加接受预抽随机数（`float3 u` / `float2 u`）的重载，原 `inout NclsRng` 版本改为薄包装 | `shaders/ncls/reference/{interfaces,sampling}.slang` | 现有 reference GPU 测试与采集结果 hash 不变 |
| S2.3 | LTC lobe 采样与 pdf（上三角矩阵闭式求逆）+ 混合 proposal，Slang 侧实现；测试里用 `torch_eval.py:117` 的 basis 项做 pdf 对照，**不新增 Torch 生产代码** | `lobe_residual_core.slang`、`tests/gpu/test_lobe_residual_gpu.py` | `pdf(sample.wi) == sample.pdf`；白炉 `E[f·cos/pdf] = 1 ± 1e-2`；pdf MC 归一化；grazing 无 NaN |
| S2.4 | **合同包装** `lobe_residual.slang`：`struct LobeResidualBackend : INclsScatteringBackend`（`CompiledMaterial = {NclsLayerInterfaceIR top; float latent[16];}`），`evaluate/sample/pdf` 从 `ISampleGenerator` 取随机数后调 core | `shaders/ncls/backends/lobe_residual/lobe_residual.slang`（新） | 锁定 Slang 2024.1.34 编译冒烟（照 `tests/gpu/test_scattering_contract_gpu.py`） |
| P2.5 | **Python 接入**：`src/ncls/learning/slang/session.py`（SlangPy 加载 core、反射 layout、`torch.autograd.Function` 包装 `bwd_diff`）；`pipelines/lobe_residual.py`（family `m2b`；descriptor `model.representation="analytic-core-lobe-residual-v1"`、`architecture="lobe-residual-prepare-mlp-v1"`、`data.source_adapter="layer-stack-direct-top-v1"`、`runtime.deployment_candidate`）；`create_model` 返回只持有 params 与 latent 张量的薄 `nn.Module`，`predict_f` 调 Slang，**无 Torch 前向**；`fit_training_state` 复用 `fit_direct_top_state`；loss 从 `p1_evaluator.py:255-304` 抽成 `p1_appearance_loss` 共用并去掉 `:263` 的 clamp；`parameter_costs` 按 §1.2，返回 key 与现有 pipeline 一致并加 `state_bytes_per_pixel` | 新文件、`pipelines/__init__.py:5`、`pipelines/p1_evaluator.py` | `tests/gpu/test_lobe_residual_slangpy.py`（marker `slangpy`）：给定 lobe 参数时 evaluate 与 `torch_eval` parity `rtol 1e-4`；一次 step 后梯度与有限差分一致 |
| P2.6 | state/params 打包：`pack_state`（64 B）用 SlangPy 反射布局；测试只做往返 | `src/ncls/core/representations/lobe_residual/state.py`（新）、`tests/unit/test_lobe_residual_layout.py` | 往返一致 |
| P2.7 | 双编译探针：GPU 测试同时用 Falcor（`falcor.ComputePass`）与 SlangPy 编译 core，evaluate 数值一致 | `tests/gpu/kernels/lobe_residual.cs.slang`、`tests/gpu/test_lobe_residual_gpu.py` | `rtol 2e-5`；任一编译器失败即阻止提交 |
| P2.8 | **框架接入（纯 Python，无 Slang 依赖，可最先做）**：部署预算单元门 `tests/unit/test_deployment_budget.py`（遍历注册表，`deployment_candidate=True` 的 pipeline 必须满足 §0.1；M1/M2 七个标 `False`），descriptor 增 `runtime.deployment_candidate` 需同步 `schemas/learning_pipeline_v1.schema.json` 与 `base.py:36-59` 的精确字段集校验；tail guard：`TrainingConfig.checkpoint_selection ∈ {"median_then_p95","tail_guard"}`（默认旧值，同步 `training_config_v1.schema.json`）、`runner.py:277` 的 `(median, p95)` 元组按策略分支（`tail_guard` = 先剔除 validation p95 > 该 run 至今最小 p95 × 1.25 的 checkpoint 再取 median 最小）、`configs/evaluation/quality-v2.json` 只改 `checkpoint_selection` 块、`quality.py:49-53` 接受 v1/v2；`p1_audit.py:540-560` 的 `is_m2` 改为探测 pipeline 可选方法 `core_f(model, batch, store, device)` 与属性 `has_signed_residual`；配置 `lobe-residual-k2-v1`（`none`，部署候选）、`lobe-residual-k2-log32-v1`、`lobe-residual-k3-log32-v1`（研究，`deployment_candidate=False`）、`smoke/lobe-residual-k2-p1-smoke`，调度沿用 S 档（bs 16、lr 3e-4、25k、minimum 4000、patience 6、seed 20260824）+ `tail_guard` | `training/{config,runner}.py`、`evaluation/{quality,p1_audit}.py`、`configs/learning/`、`configs/evaluation/` | `tests/unit` 全绿；用 P1 v1 M2-S 的 validation history 回放断言 tail guard 选 step 7500 而非 4500 |
| P2.9 | 远程：三个配置按冻结 seed 集运行；先生成 implementation/convergence report，再读取一次 test/adversarial/dense；`audit-p1`；`compare` 对 M1-M | 远程 | 复现状态与质量比较分开。实现正确且稳定收敛后登记全部质量；按最差 state 的 `E_core/E_ref` 与 lobe 承担能量做结构归因，不以某个绝对 p95 决定是否复现 |

Phase 2 出口：R1 与 S1 通过或明确归因；一份 core 源同时被 SlangPy、Falcor 测试编译；注册表登记全部正式 run。

### Phase 3 — 已并入 Phase 2

保留编号，Phase 4/5 的引用不变。

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
| 5.1 | 注册表：`lobe-residual-k2-{none,log32}`、`k3` 三行 + Slang/viewer 实测行；结论列分开写implementation/convergence、质量比较与C2成本分类 | `experiment_log.md` |
| 5.2 | `model_candidates.md` §3 M2 改写为 `lobe-residual` 的正式定义（§1.1），删除 signed-residual 描述；§1.5 增「部署档」定义 | 文档 |
| 5.3 | `p1_audit.md` §7 关闭已完成项；`experiment_framework.md` §6 表 P1 行更新主要候选 | 文档 |
| 5.4 | 记忆/TESTING：`TESTING.md` 增本计划各期命令（unit / gpu / slangpy / viewer benchmark） | 远程按 TESTING.md 复核 |

## 4. 依赖关系与并行性

```text
P1.0 spike (远程) ──► S2.1–S2.4 core/合同 ──► P2.5 SlangPy 接入 ──► P2.9 训练 (远程) ──► R1 implementation/convergence
                        │                          ▲
P2.8 框架接入 (本地, 无依赖) ───────────────────────┘
S2.1 ──► P2.7 双编译探针 / S2.3 sampler 测试 (远程)
S2.4 ──► P4.1 exporter ──► V4.2 loader ──► V4.3 泛型 pass ──► V4.4 PT family ──► V4.5 benchmark
```

- 本地（静态）可直接推进：P2.8 全部；S2.1–S2.4、P2.5–P2.6、P4.1、V4.2–V4.4、D4.6 的代码。但 S2.1 与 P2.5 的 API 细节依赖 P1.0 spike 给出的 slangpy 版本与语法清单，所以 **spike 是第一个动作**，P2.8 可与之并行。
- 必须远程：P1.0、S2.3/P2.7 的 GPU 测试、P2.9、V4.5，以及每期 pytest/GPU 回归。
- 规模：Phase 1 小；Phase 2 大（core、sampler、SlangPy 接入、框架四块）；Phase 4 大（viewer 五处硬编码、PT family、第二累积链）；Phase 5 小。

## 5. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| SlangPy 携带的 slang 与 Falcor 8.0 的 2024.1.34 语法不兼容，或 Torch 互操作/吞吐不可接受 | E1 单一源无法同时训练与部署 | P1.0 spike 先行；core 只用 2024.1.34 已验证的写法（固定数组、`typedef` 绑定 associated type、`[unroll]`）；P2.7 双编译探针常驻；失败则启用 P1.1 备选并登记 |
| 质量信号要等 SlangPy 接入后才有 | implementation/convergence 证据晚于 Torch-first 方案 | 接受：避免写一套会被扔掉的 Torch 模型；P2.8 框架部分先行，接入后立即可跑 |
| LTC 型 lobe 表达不了 M2 失效的 4 个 state（coat 下 base 的多次散射形态） | 某些材质结构上的quality较低 | 用 `E_core/E_ref` 与 lobe 承担能量分解失败原因；备选：lobe 型别扩展为 GGX-VNDF 型（`interfaces.slang` 三件套直接可用）或 K=3 研究档；不把低quality写成复现失败 |
| 64 B state 在 `log32` 下刚好卡线，half 打包引入精度误差 | parity 容差 | parity 用 half 打包后的 state，容差按 half 量级单列；`none` 配置 48 B 留余量 |
| viewer 改造范围大，容易把 film-m1 diagnostic 路径改坏 | 回归 | V4.2 保留 film-m1 表项，加载测试双 bundle；泛型 pass 先在 film-m1 上验证再切新后端 |
| PT + method 的第二累积链把左右噪声关联/去关联处理错 | 差图误判 | 左右独立 seed 流、相同 spp；capture manifest 记录两侧 `estimated_mean_relative_standard_error` |
| 30-state 上的 p95 本身不稳定（framework §7） | 单一汇总会掩盖材质结构差异 | 报告分组CI与leave-one-out；p95只作描述，P2扩大state后重新比较，不作跨材质复现门 |

## 6. 已定事项（2026-08-25）

1. **Slang 优先，不写 Torch 参考模型。** 现有 `torch_eval.py` 只作 evaluate/pdf 的 parity oracle；Torch 训练路径（P1.1）只是 spike 失败时的备选。
2. `correction` 默认 `none`（48 B state），`log32` 只作 matched 对照。
3. PT + method 做完整第二累积链（V4.4），不做单帧折中。
4. SlangPy 固定版本写入 `environment.yml`；不升级 Falcor 的 slang。
5. 代码风格：实现以简洁为先——复用现成件（`torch_eval.py`、`interfaces.slang`、`p1-appearance-v3` loss 抽函数而非复制），只暴露 `K` 与 `correction` 两个配置轴，不为未来阶段预留抽象；文件按语义单一职责划分，不设行数硬限（规则见 `.trellis/spec/project/code-organization.md`）。
