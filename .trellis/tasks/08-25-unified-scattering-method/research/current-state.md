# 当前状态审阅

## 结论摘要

1. 右侧 viewer 目前不能在通用接口下执行 path tracing。它只接受一个硬编码的 `film-m1-direct-neural` diagnostic bundle，右侧执行的是 raster G-buffer 后的 deferred approximation。
2. 公共散射合同已经定义 `prepare/evaluate/sample/pdf` 的方向、测度、事件和 capability；缺口主要在真实 backend、MethodBundle loader、viewer shader specialization 与 path-tracing method adapter，而不是从零发明公共语义。
3. 当前 `lobe_residual` 不是可运行方法：训练 pipeline 是注册占位，sampler/PDF 是固定失败占位，也没有 `INclsScatteringBackend` 包装。
4. `reference-shard` v5 已保存 sampler 训练可用的 `(wo, wi, f·|cos|)`、采集 proposal PDF 和 solid-angle weight，点查询 schema 足以表达当前反射域。NVIDIA baseline 还使用 directional mollification 和 sampler-vs-current-evaluator KL；实施前必须审计冻结 corpus 的密度/分组是否足以忠实适配这些训练机制。若不足，可能需要保持 v5 schema 但生成新 corpus identity，或版本化增加 mollification 分组；不能继续笼统宣称一定不重采。
5. 当前 LayerStack reference descriptor 只声明 upper-hemisphere `evaluate + monte-carlo-moments`，落盘 `reference_pdf` 为 0。若首个方法要求外部透射、delta 或蒸馏 source reference 的 sampler/PDF，则现有语料不够，必须另行扩展合同和重采。
6. `p1_audit.md` 中关于 P1 v1 失效机制、单一 Slang 源、matched `sample/pdf` 和 PT-vs-PT 对照的判断仍可复用；把纯 lobe 参数化定为目标主线、把 method 当成 reference 的“第五个 family”、保留 Film M1 兼容路径等内容与当前目标冲突。
7. 用户已确认首个 sampler 使用 learned tractable distribution head：网络从共享 state 预测解析 proposal 参数；`sample/pdf` 严格匹配。裸方向 MLP 只有在使用可逆球面变换并能计算 Jacobian 时才可能满足合同，不作为本任务起点。
8. 用户已确认首个方法限定为 upper-hemisphere、reflection-only、non-delta LayerStack。现有 v5 response 字段足以支撑该事件域；是否重采由 NVIDIA baseline 的训练密度/mollification adequacy 审计决定，而不是由 `reference_pdf=0` 决定。
9. 用户已确认归零边界：旧 `legacy_ltc_k2` 方法身份和路径删除，但 LTC/VNDF/sample/PDF 等正确数学原语迁移为通用 Falcor-free Slang 组件，并保留一个显式 analytic control/oracle；其余旧方法路径彻底删除。
10. 方法设计必须优先保证表达能力与研究价值：完整吸收 P1 审计、v2 计划和 `lobe_residual` 骨架的有效结论，在 `design.md` 冻结具体结构后才实现，不能以 lifecycle smoke 为目标选择粗糙方法。
11. NVIDIA 原始论文不是泛泛 prior，而是本项目当前目标的结构 baseline：two learned frames + direct BRDF MLP；sampler 预测 9 个参数，组成 tilted-cosine diffuse 与 non-centered anisotropic GGX specular proposal；sampler PDF 对当前 learned BRDF 做 KL，且 latent 从 KL 梯度 detach。论文 learned-frame ablation 明确优于同参数量 vanilla MLP。
12. 本项目需要做 matched 2×2：evaluator `{NVIDIA direct, exact-core positive residual}` × sampler `{NVIDIA diffuse+GGX9, LTC-K2}`。LTC 仍是有价值候选和公共原语，但不能未经实验替换论文 sampler；最终部署选择必须不劣于 budget-matched NVIDIA baseline。

## 证据

### Viewer

- `apps/viewer/MethodBundle.cpp:72-86` 只接受 `runtime_class=diagnostic`、`backend_id=film-m1-direct-neural`，并硬编码 `nclsFilmM1Prepare/nclsFilmM1EvaluateF`。
- `apps/viewer/shaders/Prepare.cs.slang:33`、`Approximation.cs.slang:49`、`Parity.cs.slang:16-17` 直接调用 `nclsFilmM1*`，没有通过 `INclsScatteringBackend`。
- `apps/viewer/shaders/ReferencePathTracer.cs.slang:5-21` 的编译期 mask 只有四种 source reference；`nclsSampleReferencePath` 与 `nclsReferencePdfPath` 位于该 source-specific 分派中，没有 MethodBundle backend 分支。
- `docs/viewer_spec.md` 明确写明右侧目前是 deferred 实时结果，并且没有 path-traced 全局传输。

### 当前方法实现

- `src/ncls/learning/pipelines/lobe_residual.py:100-125` 的 `create_model/predict_f/training_loss` 都抛 `NotImplementedError`。
- `shaders/ncls/backends/lobe_residual/lobe_residual_core.slang:125-147` 的 `pdf()` 固定返回 0，`sample()` 固定返回 false。
- `shaders/ncls/backends/lobe_residual/` 不包含合同包装 `<backend>.slang`；当前只有 core、MLP 原语和 pack 骨架。
- `docs/realtime_material_compilation.md:72,110` 要求目标方法由小型 `EvaluateMLP` 直接补全散射；解析 core 可以稳定极窄响应或提供 proposal，但不能把目标退化成只在 `prepare()` 预测固定 closure/lobe。

### 数据合同

- `docs/contracts/reference_dataset.md:8,37,49`：v5 保存 `f·|cos|`、采集 proposal PDF、solid-angle weight、event flags 与 reference PDF。
- `src/ncls/data/providers/layer_stack.py:114-118`：当前 LayerStack 只声明 upper-hemisphere `evaluate/monte-carlo-moments`。
- `src/ncls/data/providers/layer_stack.py:463-472`：LayerStack shard 的 `reference_pdf` 为全 0；这表示 source reference 没有声明可蒸馏 sampler，不影响从 response 训练另一个有解析密度的 method proposal。
- 磁盘上 `data/reference-responses/layer-stack-p1-v1/` 是 v5；根目录的 E1/E2 文件是 v4，四个 pilot 文件是 v3，当前 reader 按规范不会读取旧格式。

## 可复用与应淘汰的内容

### 可复用

- 生成 ABI 的 scattering contract、方向/余弦/PDF 测度、capability 机制。
- `reference-shard` v5 的 response 与 importance weight 字段；是否重采由最终方法事件域决定。
- P1 v1 对错误 FiLM、signed residual clamp 死区、成本记账和 Film M1 硬编码的审计证据。
- `legacy_ltc_k2` 中经测试的解析公式与 VNDF/sample-generator 适配，可作为 optimized-code control、proposal 原语和合同 oracle；不能作为目标 neural 表示。
- `lobe_residual` 中 Falcor-free 文件划分、定长 state/weight 形态和非负/有界参数化思路；占位实现和“纯 lobe 即目标方法”的结论不保留。

### 应淘汰

- `film_m1` 的训练模型、bundle exporter、viewer 硬编码路径、配置和测试；它只证明过 frozen-state diagnostic 生命周期。
- P1 v1 的错误 M1/M2/T pipeline/config 与相应 run/export/capture/audit/comparison/oracle。
- 未完成却注册成 deployment candidate 的 `lobe_residual-k2-v1` 占位，以及 sampler/PDF TODO。
- 把 method 加成 source-reference family 的 viewer 方案；右侧应是通用 MethodBundle backend specialization，reference 继续保留族专属语义。
- v3/v4 HDF5 和 smoke 数据；它们不属于当前 v5 corpus。

## 清理分级（待用户确认）

### 现在即可安全删除的未版本化内容

- `data/reference-responses/` 根目录下 11 个 v3/v4 HDF5：`layer-stack-e1-*`、`layer-stack-e2-*`、`layer-stack-evaluator-pilot-v3.h5`、`materialx-spatial-pilot-v3.h5`、`merl-representative-pilot-v3.h5`、`openpbr-representative-pilot-v3.h5`。
- `data/reference-responses/smoke/`。
- 若不再需要复现 P1 v1：`artifacts/runs/`、`artifacts/exports/`、`artifacts/captures/`、`artifacts/audits/`、`artifacts/comparisons/`、`artifacts/oracles/`、`artifacts/configs/`、`artifacts/diagnostics/`。这些都是旧方法产物，可由 Git 中的代码和 reference corpus 重跑，但删除后本机现有 checkpoint/capture 不可恢复。

### 暂时保留

- `data/reference-responses/layer-stack-p1-v1/` 与 `artifacts/corpus/layer-stack-p1-v1.json`：当前证据表明它们仍是合法 v5 response corpus，可用于首个反射方法；若最终决定支持透射/delta 或改变 response/query 合同，再整体替换。
- `assets/`、`external/`、`references/`：它们是 source/reference provenance，不是错误模型数据，不能随 P1 清理删除。
- `build/` 中的 locked reference/viewer 构建输出不属于模型数据；可以按磁盘需要重建，但不作为本任务“错误方法归零”的语义目标。

### 任务内删除

- 所有被 Git 跟踪的旧模型、配置、shader、bundle、viewer 旁路、测试与过时研究文本。删除前先把仍承担公共 oracle/control 的解析原语迁到明确位置，并确保没有 fallback 或字符串硬编码残留。
