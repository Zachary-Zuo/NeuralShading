# 统一材质 Reference 训练数据回路设计

## 1. 冻结边界

本任务统一的是 scattering query 的调用合同与训练数据流，不是把源材质改写成同一种表示。

- LayerStack 的 `SourceFamilyDefinition`、`SourceSnapshot`、随机游走 reference、`NclsLayerStackReferenceBackend`、原生参数编辑与 package/viewer 能力全部保留。
- 删除 LayerStack 专用 collector、HDF5 corpus、`LiveReferenceBatchSource` 与训练 CLI 分支；MaterialX/MDL 等现有 source-specific live producer 也由同一个 generic producer 取代。
- 删除全部 offline/replay/recorded-batch 能力，不提供 reader、migration、compatibility shim 或新磁盘 schema。
- source reference 与 method runtime 都遵守 `ncls.scattering-backend@1`：`prepare/evaluate/sample/pdf`。source 的 `sample/pdf` 用于 reference PT 与数值验证，但不是 NVIDIA sampler 的 teacher。
- NVIDIA evaluator 直接学习线性 RGB BRDF `f`。`f·|cosθi|` 只在渲染 response adapter 或 sampler 目标密度中显式构造，不再作为 evaluator target 或持久化 response measure。
- NVIDIA 当前只声明 LayerStack 与 MaterialX source adaptation；generic reference dispatcher 必须支持全部正式 source family，但方法不支持某个 source 时仍由 `MethodDescriptor` fail closed。统一 query ABI 不伪造统一的 native feature schema。

## 2. 所有权与数据流

```text
source locator
  → SourceFamilyDefinition.load_snapshot()
  → SourceSnapshot
  → ReferenceProgramDefinition.compile_runtime()/compile_material()
  → ReferenceQueryDispatcher.prepare/evaluate/sample/pdf
                         │
                         └─ evaluate().f ───────────────┐

MethodDefinition + source adaptation ─→ query/context ─┼→ OnlineTrainingProducer
                                                       │     ├─ EvaluatorBatch(target_f)
                                                       │     └─ MethodSamplerBatch(sample_u)
                                                       ↓
                                                  TrainingRunner
                                                       ↓
                                      NVIDIA evaluate_f / learned sample+pdf
```

各层唯一职责如下：

| 层 | 所有权 | 明确不做 |
|---|---|---|
| source family | 从 locator 构造、验证、编辑原生 `SourceSnapshot` | 不执行训练、不定义通用 BRDF |
| reference program | 把 snapshot 编译为 canonical backend module 与 typed resources | 不生成训练方向、不写 HDF5 |
| reference dispatcher | 在 GPU 上调用同一 concrete state 的 `prepare/evaluate/sample/pdf` | 不识别 family 名称、不实现 closure fallback |
| method source adaptation | 生成论文需要的 native feature、UV/LOD/footprint conditioning | 不计算 reference response |
| online producer | 组合 route RNG、conditioning 与 dispatcher result | 不按 family 分支、不持久化 batch |
| method objective | 解释 typed batches，并执行 learned evaluator/sampler loss | 不读取 source provider 或 HDF5 |
| runner | lifecycle、optimizer、route 调度、resume 与 lease | 不解释 evaluator/sampler 数学 |

## 3. Source 与 Reference 绑定合同

### 3.1 Source locator

`TrainingConfig@3` 用一个 generic `source` 对象取代 `batch_source`：

```json
{
  "source": {
    "family_id": "materialx.document@1.39.4",
    "materials": [
      {"locator": {"kind": "catalog-asset", "asset_id": "american_walnut_veneer"}}
    ]
  }
}
```

`SourceFamilyDefinition.load_snapshot(locator)` 负责解释 family-native locator。CLI 只查 source registry 并调用该接口；不得再比较 option 字段集合或按 LayerStack/MaterialX/MDL 分支。locator 是源资产入口，不是训练数据或 recorded batch。

reference registry 新增按 `(family_id, source_contract_version)` 的唯一查找，并验证：

1. source descriptor 的 `reference_program_id` 与 `program_key@version` 一致；
2. snapshot、source descriptor 与 reference descriptor 的 family/version 一致；
3. capabilities 至少包含 `PREPARE|EVALUATE|SAMPLE|PDF`；
4. 同一 source contract 只能注册一个 canonical reference program。

任一条件不成立即在训练构造期失败，不能退回 data provider 或 generic proposal。

### 3.2 Runtime 与 material payload

`RuntimePayload` 与 `MaterialPayload` 是 dispatcher 唯一 binding 输入。现有 payload descriptor 需要补齐为可执行 typed binding：

- buffer descriptor 的 `usage` 是实际 shader symbol；`dtype/shape/stride/alignment` 决定 GPU buffer 创建与校验；
- texture descriptor 使用版本化的 typed texture dtype，明确维度、format、颜色空间与 mip 语义；
- sampler 作为 typed resource 显式提供，不依赖 provider 私下创建；
- MDL generated target code 使用 `slang-module-source@1` descriptor，`usage` 是 module name；argument block、RO segment、2D/3D texture 与 sampler 同样通过 typed binding；
- binder 按 descriptor dtype 分派资源类型，不读取 source family/program key。

MDL 的 `ReferenceProgramDefinition` 改为导出 canonical `mdl.slang` module closure；旧 `mdl_query.slang` 的 query 入口删除。动态 `NclsMdlGenerated` 仍来自同一个锁定 SDK artifact，只是由 generic binder 注入。

## 4. Generic GPU Scattering Query

### 4.1 Query kernel

新增一个 program-specialized、family-agnostic Slang query module。host 通过 define 注入 reference program header；该 header 必须导出：

```slang
NclsPackageBackend nclsCreatePackageBackend();
NclsPackageCompiledMaterial nclsLoadPackageMaterial(uint index);
typedef ... NclsPackageState;
```

query kernel 从公共 buffer 构造完整 `NclsScatteringContext`，只执行以下形式：

```slang
let backend = nclsCreatePackageBackend();
let state = backend.prepare(context, nclsLoadPackageMaterial(materialIndex));
state.evaluate(wi, sg);
state.sample(sample, sg);
state.pdf(wi);
```

不得包含 source family 条件、另一个 BRDF 函数、固定 cosine/GGX fallback 或 response clamp。

### 4.2 Python dispatcher

`ReferenceQueryDispatcher` 负责：

- 从 reference program 与 snapshots 构造 concrete program/binding；
- 按 typed descriptors 创建并校验 GPU resources；
- 为 route 保留至少两个 shared-buffer slots；
- 把 context/query 输入从 CUDA tensor 复制到 Falcor shared resources，显式执行 CUDA→Falcor 同步；
- 返回映射到同一 CUDA device 的 tensor view，并用 lease 阻止 slot 在 consumer 完成前复用；
- 在 `close()` 时拒绝仍有 active lease 的静默释放。

训练正式路径不得调用 `to_numpy()`、HDF5 或 host response readback。CPU readback只允许独立 parity test 与显式诊断工具。

### 4.3 Operation result

dispatcher 暴露三个 typed operation：

- `evaluate(query, wi, evaluation_samples)` → `f`、forward/reverse PDF、event、valid；
- `sample(query, sample_seed)` → source-native direction/event/PDF/weight/eta/valid tuple；
- `pdf(query, wi)` → forward/reverse PDF。

`sample()` 的 tuple 原样返回；不得用舍入后的 `wi` 调 `evaluate/pdf` 后重建。独立 `pdf(wi)` 是另一次 query，用于 NEE 与一致性测试。

LayerStack random-walk `evaluate()` 是 stochastic reference。`evaluation_samples > 1` 由 generic kernel 重复调用同一 state contract并只平均 `f`；PDF/event 必须一致，任一非有限或 invalid evaluation 令 query 失败。该字段是 route 级通用预算，不触发 LayerStack 分支。确定性 source 使用 1。禁止 outlier clamp、按亮度删样本或 group-p95 过滤。

## 5. Typed Online Training Batches

### 5.1 Schema

删除 `TrainingBatch@1` 的全字段矩形合同，使用不兼容的 tagged batch：

```text
TrainingConditioning
  source_index, source_snapshot_ids
  wo
  uv, uv_dx, uv_dy, mip_level（source adaptation 需要时）
  native_features（bootstrap 需要时）
  provenance + lease

EvaluatorBatch@2
  conditioning
  wi
  target_f

MethodSamplerBatch@2
  conditioning
  sample_u
```

核心不再要求 `target`、`solid_angle_weight`、`reference_pdf`、`sample_count`、`rng_seed`、`query_role`。确有数学意义的 method-specific tensor 可由 route contract 声明，但不能用零值占位。

`MethodDescriptor.training_batch_requirements` 改为按 route kind 声明的 schema。NVIDIA objective 必须接收一个 evaluator batch 与一个 method-sampler batch；传错 tag、缺字段或多出旧 target estimator 字段均 fail closed。

### 5.2 Route config

`TrainingRoute@3` 用显式 `kind` 替代 numeric `query_role` 与 `target_estimator`：

- `reference-evaluator`：生成 half/difference `wo/wi`，执行 source `evaluate`；
- `method-sampler`：生成独立 conditioning 与 `sample_u`，不执行 source scattering query。

方向 proposal、空间 filtering、directional mollification、reference evaluation sample count 都是版本化 route/query recipe。正式 NVIDIA 两条 65k route 保持独立 seed stream。

### 5.3 Native feature 边界

LayerStack、MaterialX 等 native parameter layout 天然不同。它们可以保留 source-specific feature adapter，但该 adapter：

- 位于 method/source-adaptation 层，不位于 reference/data producer；
- 只生成 encoder 输入、surface footprint 与 materialization pyramid；
- 不拥有 `evaluate/sample/pdf`，不导入旧 provider；
- 由 `(method_key, family_id, source_contract_version)` registry 选择，generic producer 不分支；
- method 没有注册 adapter 时明确拒绝训练，不把缺少的 native feature 填零。

现有 `native_features.py` 因而迁出 `ncls.data`；LayerStack feature 编码保留为 NVIDIA source adaptation，不再构成 LayerStack 专用 batch source。

## 6. NVIDIA 方法语义

### 6.1 Evaluator

作者论文的 decoder target 是 BRDF value。Python/Slang core 统一改为：

```text
f_hat = exp(raw - 3)
L_eval = mean(abs(log1p(f_hat) - log1p(target_f)))
runtime evaluate().f = f_hat
```

删除 `response_cos` 命名、`response()/forward()` 二义性、`ResponseToBareF` 以及 runtime 除 cosine。package parity 字段改为 `expected_f`。旧 checkpoint/method descriptor 不兼容，方法 version 与 implementation identity 更新。

### 6.2 Learned sampler

sampler route 只提供 conditioning 与 `sample_u`。method 当前 learned proposal 执行 sample 与 PDF：

```text
wi ~ q_theta(wi | latent, wo)
f_hat = stopgrad(evaluate_f(detach(latent), wo, wi))
g(wi) = luminance(f_hat) * abs(dot(n_s, wi))
L_sampler = -mean(stopgrad(g / q_theta) * log(q_theta))
```

source reference `sample/pdf` 不参与该 loss。latent/evaluator detach、GGX9 head、两条独立 batch、Adam/cosine schedule、300k/100k lifecycle 与 65k batch size保持论文 correspondence。

## 7. Config、Checkpoint 与 CLI

- `TrainingConfig@3` 删除 `batch_source.kind=offline|live`，改为 `source`、`online_query` 与 typed routes；加载旧 version 直接失败。
- `TrainingCheckpoint@3` 把 `data_source_identity/batch_source_state` 改为 `source_snapshot_ids/reference_program_identity/query_stream_identity/query_stream_state`。resume identity 包含 source locator resolution、reference descriptor、query recipe、method source adapter implementation 与各 route RNG state。
- `learn train/evaluate` 只构造 `OnlineTrainingProducer`；CLI 不再有 `_batch_source` family/options 分支。
- `learn export` 通过 checkpoint 中的 generic source locator 重载 snapshot 并校验 snapshot id；删除 LayerStack/MaterialX export 分支。
- 删除 `data validate/plan/collect/validate-corpus/audit-dense` 命令。需要 reference parity 的工具改调 dispatcher，不新增 batch recording 命令。

## 8. 删除与迁移清单

### 8.1 删除

- `src/ncls/data/` 中 offline corpus、collector、store、dataset、statistics、selection、profiles、priors 与全部 provider/live batch source；删除 HDF5 schemas。
- `shaders/ncls/data/reference_*.cs.slang` 与 MDL 专用 `mdl_query.slang`。
- `configs/corpus/`、旧 offline 配置与 corpus selection。
- 只验证旧 shard/corpus/provider/live producer 的 tests 与 CLI 入口。
- `environment.yml`、`pyproject.toml` 中仅由旧数据路径使用的 `h5py` 依赖。
- `src/ncls/paths.py` 的 `REFERENCE_RESPONSE_ROOT`。

### 8.2 迁移/重写

- Falcor device/shared-buffer helpers 迁到 runtime/query 所有权，不再以 data 命名。
- surface/query context 迁到 reference query contract。
- native feature pyramid 与 LayerStack/MaterialX encoder inputs 迁到 learning source adaptation。
- MDL asset loading、SDK compilation与 typed resources 迁到 source/reference 层；falcor2 oracle 仍只做隔离 parity。
- pbrt、MDL parity、viewer asset preparation tools 改用 source loader + reference program/dispatcher。
- LayerStack 与 MaterialX learning configs 重写为 `TrainingConfig@3` online source locator；不保留 v2 reader。
- `AGENTS.md`、`.trellis/spec/`、architecture/data/research 文档移除 HDF5/corpus 当前路线，明确 online query 与 `f` target。

磁盘 `data/reference-responses/` 不由代码删除。实施开始前再次只读审计并向用户报告；用户删除后只验证目录无 `.h5/.hdf5`。

## 9. 验证设计

### 9.1 静态与 unit

- reference registry 对五个正式 family 唯一映射并检查完整 capabilities；缺入口、重复 family 或 descriptor identity 漂移均构造失败。
- CLI/producer/dispatcher 静态扫描不含 family id、旧 provider、HDF5、offline/replay/recorded 分支。
- typed batch 分别拒绝 dummy target、错 tag、错 device/shape 与旧字段。
- config/checkpoint v3 roundtrip、resume RNG identity 与旧 version fail-closed。
- NVIDIA evaluator test 验证 target/output 均为 `f`；sampler objective test 用手工 `f` 与 cosine 验证 target density。

### 9.2 GPU

- generic dispatcher 对 LayerStack、MaterialX、OpenPBR、MERL、MDL 执行 `evaluate/sample/pdf`，覆盖 finite、event、reflection/transmission 与 reverse PDF。
- project-owned deterministic continuous sampler按 `weight=f·|cos|/pdf` 验证；native source 分开验证 sample tuple identity 与 independent evaluate↔pdf，沿用 shared scattering spec 对极窄掠射方向的边界。
- dispatcher 与 direct canonical backend/source oracle 使用 `references/acceptance.json` 中冻结的 deterministic/Monte Carlo profile；不根据本任务结果放宽。
- CUDA/Falcor shared tensor、双 slot lease、显式同步与无 host response readback通过 GPU lifecycle test。
- LayerStack stochastic evaluate 用独立 replica 与 `references/acceptance.json` Monte Carlo coverage gate；不把单点 MC 波动当 deterministic parity。

### 9.3 Training integration

- LayerStack 与 MaterialX 各跑 2-step smoke，覆盖 bootstrap→materialization→finetune、checkpoint/resume、两条独立 route 与 zero-placeholder absence。
- formal NVIDIA config只做静态 correspondence 与代表性 focused step preflight；本任务不启动 300k formal training，也不以 observed quality/time/memory 决定完成。
- package query parity验证 learned runtime `evaluate().f`，并编译共享 Slang module closure；不要求本任务执行完整 viewer 场景研究。

## 10. Rollback 与禁止方案

按以下提交/验证边界推进：reference dispatcher → typed online batches → NVIDIA `f` semantics → CLI/config/checkpoint → legacy deletion/docs。每个边界必须先通过对应 focused tests再删除旧实现，便于用普通 commit revert 回滚；不使用兼容层双写或双读。

明确拒绝：

- 保留 HDF5 作为“临时 cache”或调试 fallback；
- 用 source `sample/pdf` 监督 NVIDIA learned sampler；
- 让 generic producer 识别 family 并调用不同 evaluator；
- 把 source reference 先写成 `ScatteringPackage` 再查询；
- 继续训练 `f·cos` 后在 runtime 除以 grazing cosine；
- 对随机游走 target 做亮度 clamp、孤点删除或事后阈值扩张。
