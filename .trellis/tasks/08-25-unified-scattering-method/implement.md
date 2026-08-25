# 统一散射方法与路径追踪验证实施计划

## 0. 执行原则

- 本目录是父任务，负责完整需求、方法冻结、子任务依赖和最终集成验收；父任务本身不直接作为实现 target。
- 六个带数字前缀的子任务严格按依赖顺序执行。每个子任务必须独立 planning、检查、提交和归档；后一个任务不能把前一个未通过的 gate 当成 TODO 带过。
- 现在只冻结各子任务的职责、依赖和验收边界；每个子任务启动前，必须根据前序真实产物完整更新并审阅自己的 `prd.md`、`design.md`、`implement.md` 和 context manifests。父任务本身不 `start`，直到六个子任务归档后才执行最终验收。
- 本任务树已获得用户的连续执行授权。上述子任务细化只要保持在父任务冻结边界内，完成 planning gate 后立即 `start`，质量 gate 通过后自动创建 scoped local commits 并归档，不设置人工停点；所有 commit 排除不相关 dirty files且不 push。范围扩大、未经授权的不可恢复删除、外部权限或真实 blocker 仍须暂停。
- NVIDIA paper-scale / deployment-matched baseline 与 `core-frame-neural-v1` 的 matched 比较未通过前，不进入 bundle/viewer 部署收口；sample/PDF 数学 gate 未通过前，不进入 method PT。
- 所有 Python/pytest 命令使用 `conda run -n neural-shading`；Falcor Python 经 `scripts/run_falcor_python.ps1`；viewer 只经 `scripts/build_viewer.ps1`。
- 旧实现先迁移仍有价值的数学原语，再删除身份和调用路径；全程不增加 fallback。

## 1. 子任务图

```text
01-reusable-scattering-math
   ↓ math/sample-PDF gate
02-mollification-data-adequacy
   ↓ frozen corpus/schema decision
03-neural-baseline-and-candidate
   ↓ quality + sampler + selection gate
04-generic-method-bundle-runtime
   ↓ bundle load/parity gate
05-viewer-method-deferred-pt
   ↓ capture/replay gate
06-legacy-method-reset
   ↓ reachability/full-regression gate
父任务最终跨层验收与归档
```

### 01. `01-reusable-scattering-math`

职责：建立唯一的 Falcor-free Slang 数学层，迁移并验证 cosine、tilted cosine、LTC、non-centered anisotropic GGX NDF、GGX/VNDF、frame、finite-mixture 与 `sample/pdf` 原语；保留 reflection null event 的诚实语义。

完成项：

- [ ] 从 `reference/sampling.slang`、`interfaces.slang`、`legacy_ltc_k2` 和 `lobe_residual` 识别公式唯一所有者，建立不依赖 LayerStack IR/Falcor 的公共模块。
- [ ] LTC 提供 normalized density、sample、参数有限性检查和 response basis；sample 与 PDF 共用同一 transform 定义。
- [ ] VNDF 区分“visible-normal 分布”与“reflection direction + null event”，测试不把 null mass 伪装成方向 PDF 归一化误差。
- [ ] 按 NVIDIA Appendix A 实现 9 参数 tilted-diffuse / non-centered anisotropic GGX proposal；它与现有 VNDF 组件分开命名、分开测度测试。
- [ ] finite mixture helper 完成 component selection、区间 remap 和完整 mixture PDF。
- [ ] 现有 LayerStack reference 改为包含公共原语，随机游走/接口数值与采集行为不改变。
- [ ] 新建显式 analytic backend/control 的 Slang 核心，但不接 viewer fallback。

Gate：

- [ ] PDF 固定 quadrature 归一化；LTC/cosine/tilted-cosine sample histogram 与 PDF 一致；reflection proposal 的连续积分 + null bin 概率为 1。
- [ ] `sample.pdf == pdf(sample.wi)`；grazing、极端各向异性和参数边界无 NaN/Inf。
- [ ] VNDF sample/normal-PDF 对照通过；reflection null event 频率有显式测试。
- [ ] reference GPU 回归和锁定 Slang 2024.1.34 编译通过。

建议验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -q
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu -q -k "sampling or scattering or reference"
```

Rollback：公共原语无法给出单一测度或 exact inverse 时，不让该分布进入 sampler family；保留明确失败并回到设计，不复制近似公式。

### 02. `02-mollification-data-adequacy`

职责：在 neural 训练开始前独立判定当前 peak-aware v5 corpus 能否忠实构造 NVIDIA directional mollification；不足时完成最小、版本化的新数据合同和 corpus，而不是让方法任务临时改写数据语义。

完成项：

- [ ] 在查询结果产生前冻结 diffuse、窄导体峰、grazing、四个既有尾部 state，以及 `wo`、cone distribution/radius、reference sample count、误差指标和阈值。
- [ ] 对同一 anchor 比较“现有 corpus 邻域/权重重建”与“新鲜 cone-averaged reference queries”，保存 matched 证据。
- [ ] adequacy 通过时冻结确定性重建算法并复用当前 v5 corpus。
- [ ] adequacy 未通过时，版本化 cone radius、anchor/group、jitter distribution、sample count、curriculum level 与 manifest 语义，并生成可供 `03` 直接训练的新 corpus identity。
- [ ] 不因 learned frames、sampler KL 或解析 sample/PDF 本身重采；不删除或覆盖合法 v5 corpus。

Gate：

- [ ] protocol/阈值先于结果冻结，运行可重复。
- [ ] 结论严格为“复用现有 v5”或“使用已验证的新 corpus identity”，没有待 `03` 处理的数据 TODO。
- [ ] reader、measure、manifest、reference provenance 和 repository policy 一致。

建议验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -q -k "corpus or dataset or measure"
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu -q -k "reference or dataset"
conda run -n neural-shading python -m ncls <frozen-adequacy-command>
```

Rollback：现有数据不足但新 schema/corpus 尚未完整生成和验证时，`03` 保持阻塞；不以邻近点启发式近似冒充忠实 baseline 数据。

### 03. `03-neural-baseline-and-candidate`

职责：在 `02` 冻结的数据入口上先实现 NVIDIA 结构 baseline，再实现 exact-core positive-residual 候选；完成 sampler 2×2 matched 对照、SlangPy 训练、offline cook、质量与方差评测，产出最终选择的冻结 checkpoint/compiled materials。

完成项：

- [ ] 实现 `nvidia-frame-two-lobe-v1`：two learned frames + direct positive BRDF MLP + 9 参数 tilted-diffuse/non-centered-GGX sampler；逐项登记论文忠实项与本项目适配。
- [ ] 实现 paper-scale diagnostic（论文 `64×64×64` evaluator、`32×32×32→9` sampler）和相同结构的 `≤2k MAC` deployment-matched baseline；两者使用同一 Slang backend ABI，前者登记为 `runtime_class=diagnostic`，不冒充当前 realtime，并在 `04/05` 中进入同一 MethodBundle/viewer 路径。
- [ ] 实现 `core-frame-neural-v1`：相同 learned-frame MLP 主体 + exact top core + positive residual；不使用 prediction clamp。
- [ ] sampler 分别实现 `nvidia-diffuse-ggx9` 与 `ltc-k2`；两者都加固定 full-support cosine safety component，并通过同一 sample/PDF 合同。
- [ ] `prepare` 使用共同的 bounded trunk 与 evaluator/sampler projections；最大 deployment payload 为 27×FP16、state ≤64 B，各组合报告实际有效 bytes。
- [ ] `evaluate` 使用 exact top core + 必选 `17→32→32→3` residual MLP；无 prediction clamp、无 lobe-only 配置、无高频 Fourier 输入。
- [ ] sampler 相对当前 evaluator 分布做 KL，loss 对 latent/shared evaluator encoding stop-gradient；reference-response cross-entropy 只作 oracle。runtime state 保存 proposal 参数，sample/pdf 不重新运行随机 neural warp。
- [ ] SlangPy 对同一源执行 forward/backward；Torch 只持 loss、optimizer、finite-difference/parity oracle。
- [ ] 只消费 `02` 冻结的 corpus identity、reader 和 mollification 构造；不在训练任务中重新选择或静默解释数据合同。
- [ ] 实现 source program → reference response query → latent direct fit → CompiledMaterial 的 offline cook；源编辑触发新 identity 与重新 cook。
- [ ] 训练 target-visible P1 回归并做 held-out-state direct-fit workflow 测试；报告全部质量、尾部、能量和成本诊断。
- [ ] evaluator `{NVIDIA direct, exact-core residual}` × sampler `{NVIDIA GGX9, LTC-K2}` 形成 matched 2×2；sampler 报告 evaluator-KL、reference oracle、归一化/null、histogram 和相对 cosine 的方差/效率。

Gate：

- [ ] `C_prepare ≤ 10,000`、`C_eval ≤ 2,000`、state `≤64 B`、compiled material `≤512 B`、evaluate weights `≤32 KB`，按实际部署 bytes/MAC 机械检查。
- [ ] Q1：median ≤ `0.045`、p95 ≤ `0.10`、15 个单层 state 各 ≤ `0.013`、4 个旧尾部 state 各 ≤ `0.15`。
- [ ] bootstrap CI、leave-one-state-out、signed energy、`E_core/E_ref` 与最差 state 清单齐全。
- [ ] 同 evaluator 的 deterministic integration 与 sampler MC 估计在预设 CI 内一致。
- [ ] learned sampler 在冻结场景集上相对 cosine 的方差结论有 matched bootstrap 支撑；不显著时继续方法迭代，不虚报收益。
- [ ] 最终部署选择相对 deployment-matched NVIDIA baseline 在 evaluator quality 与 sampler variance 上非劣，并至少在质量、时间或内存一项形成 paired-bootstrap 支撑的 Pareto 改善；否则部署 baseline 本身。
- [ ] SlangPy 与 Falcor 双编译、half-packed state parity 和有限差分梯度通过。

建议验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_deployment_budget.py tests/unit/test_pipeline_contract.py tests/unit/test_quality_evaluation.py -q
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu -q -k "core_neural or ltc or slangpy"
conda run -n neural-shading python -m ncls learn train --config <frozen-config>
conda run -n neural-shading python -m ncls learn evaluate --config <frozen-config> --suite configs/evaluation/quality-v1.json
```

Rollback：Q1 未过时只允许在冻结预算内修改 frame/warp、loss、latent 或网络特征，并重新形成 matched 证据；自研变体不优于 baseline 时选择 baseline，不启用旧 lobe-only 或超预算 Film fallback。

### 04. `04-generic-method-bundle-runtime`

职责：建立 backend-agnostic MethodBundle Slang module/type specialization、通用 exporter/loader、layout 反射、成本/capability gate 与 bundle parity。

完成项：

- [ ] manifest/schema 描述 shader module、concrete backend type、contract version、resource layout、state stride、entry capability 和 hashes；不靠 `backend_id` 推断实现细节。
- [ ] exporter 从 Slang reflection/冻结 layout 写 params、CompiledMaterial 和 parity probe；删除手写 Film 权重偏移模式。
- [ ] viewer loader 先做 schema/hash/platform/contract/cost/capability 校验，再创建 generic specialization；不写 method ID 分支。
- [ ] generic prepare/evaluate/sample/pdf adapter 通过 `INclsScatteringBackend`；pass 不直接调用 backend 自由函数。
- [ ] analytic control bundle 与 neural bundle 都走相同 loader；两者身份、runtime class 和 capability 明确，互不 fallback。
- [ ] loader 与 viewer 接受 `diagnostic` 和 `realtime` 两类合法 bundle；二者执行相同语义接口，但 UI/capture 显式显示 runtime class，只有后者通过 realtime 成本门。
- [ ] 不兼容 bundle 报具体错误，篡改 shader/weight/layout/hash 后必须拒绝。

Gate：

- [ ] export → validate → load → parity 的 headless 链路通过。
- [ ] realtime cost/capability 机械门与 Python manifest 判定一致。
- [ ] loader 与 shader 中不存在 `film_m1`、新 method ID 或旧 control ID 的硬编码分支。
- [ ] Slang module/type specialization 的内容 hash 与 capture/replay identity 稳定。

建议验证：

```powershell
conda run -n neural-shading python -m pytest tests/unit -q -k "bundle or scattering or deployment"
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu -q -k "bundle or parity or backend"
```

Rollback：动态 module/type 在锁定 Falcor/Slang 下不可行时，回到合同层设计；不得退回按 backend 字符串调用自由函数的 registry。

### 05. `05-viewer-method-deferred-pt`

职责：让右侧 MethodBundle 在 deferred 与 PT 两条 renderer path 中使用同一 generic specialization，并提供可重复 capture/replay 证据。

完成项：

- [ ] `Reference PT | Method Deferred`：右侧 G-buffer prepare 后对灯光方向调用通用 `evaluate`。
- [ ] `Reference PT | Method PT`：ray hit inline `prepare`，通过同一 state 调 `sample/pdf/evaluate`，不加入 source-family dispatch。
- [ ] method PT 拥有独立 accumulation/reset 生命周期；左右共享 scene/camera/material/light/exposure/tone mapping。
- [ ] UI、CLI 和 capture manifest 记录两侧 integrator、bundle identity、spp、bounce limit、random seed、raw-authoritative 标志。
- [ ] deferred 的方向/余弦 adapter 与离线 evaluator 探针一致；PT 的 path weight 恰好计算一次 `f·cos/pdf`。
- [ ] headless capture/replay 对两个模式可重复；bundle 切换与场景/物理变化遵守既有 invalidation 规则。

Gate：

- [ ] method deferred pixel probe 与同方向 backend evaluate 一致。
- [ ] method PT 在 constant/environment probe 上与同 evaluator 的确定性积分 CI 一致。
- [ ] source-reference PT vs method PT capture 覆盖单层、4 个旧尾部、多 lobe/各向异性和 grazing 场景。
- [ ] capture/replay 逐字节一致；hash 篡改非零退出。
- [ ] `scripts/build_viewer.ps1` 完成后 `external/Falcor` 仍在锁定提交且工作树干净。

建议验证：

```powershell
.\scripts\build_viewer.ps1 -Configuration Release
.\scripts\benchmark_viewer.ps1 <frozen-preset-args>
git -C external\Falcor status --short
```

Rollback：method PT 与 deferred 不能共用 backend specialization 时停止 viewer 集成并修正通用 adapter；不复制第二套 method shader。

### 06. `06-legacy-method-reset`

职责：替代路径全部验收后，删除旧方法身份、错误模型、配置、旁路、测试和失效数据入口，更新稳定文档/spec，并完成最终可达性审计。

完成项：

- [ ] 删除 Film M1 pipeline/model/exporter/backend/viewer 硬编码、配置与专属测试。
- [ ] 删除未完成 `lobe_residual` 方法注册、配置、backend identity 与 TODO；公共原语已由 `01` 接管。
- [ ] 删除旧 `legacy_ltc_k2` identity；只保留新命名 analytic control 和公共数学模块。
- [ ] 删除/改写过时研究路线、稳定文档、spec 和命令入口；`method-constraints.md` 与根目标统一为 mandatory direct evaluator + matched sampler。
- [ ] 删除 tracked 旧 artifact references、schema 字符串、fallback 和 dead code。
- [ ] 给用户执行未版本化 v3/v4 HDF5、smoke 与 P1 artifact 删除清单；保留合法 v5 corpus、assets、references、external。
- [ ] 全仓扫描旧 ID、错误术语、固定 entry point 与旧数据 schema，逐项解释唯一允许保留的历史复现文本。

Gate：

- [ ] 生产入口只保留最终选中的 neural realtime method 与 analytic control；paper-scale/未胜出的 matched 变体只留实验注册与 provenance，不进入默认 viewer 方法列表；无静默 fallback。
- [ ] 全部 unit/GPU/viewer/capture 回归通过。
- [ ] root repo policy、reference registry、data manifest 和 MethodBundle provenance 一致。
- [ ] 清理后从 source program offline cook 到两个 viewer 模式可全新重建。

建议验证：

```powershell
rg -n "film_m1|film-m1|legacy_ltc_k2|legacy-ltc-k2|lobe_residual|lobe-residual" src shaders apps tests configs docs .trellis/spec
conda run -n neural-shading python -m pytest tests/unit -q
.\scripts\run_falcor_python.ps1 -m pytest tests/gpu -q
.\scripts\build_viewer.ps1 -Configuration Release
git -C external\Falcor status --short
```

## 2. 父任务最终验收

- [ ] 按 `prd.md` 逐条核对完整数学合同、method quality、sampler correctness、bundle genericity、viewer 两模式和清理归零。
- [ ] 重新读取 core/data/learning/viewer 四层 Quality Check，执行全量跨层检查。
- [ ] 从干净 runtime assets 重新导出 bundle，完成 headless deferred/PT capture 和 replay。
- [ ] 核对运行产物只进入 `artifacts/`，root Git 无 data/build/report/cache。
- [ ] 使用 `trellis-update-spec` 固化公共数学、method backend、bundle specialization 和 viewer 生命周期规则。
- [ ] 形成按逻辑单元分组的提交计划，用户确认后提交；不包含用户无关文件 `SmileySans-Oblique.otf`。

## 3. 用户可预先删除的未版本化数据

现在即可删除：

- `data/reference-responses/` 根目录中的 v3/v4 `layer-stack-e1-*`、`layer-stack-e2-*` 与四个 `*-pilot-v3.h5`；
- `data/reference-responses/smoke/`；
- 若不再复现 P1：`artifacts/{runs,exports,captures,audits,comparisons,oracles,configs,diagnostics}/`。

暂时保留：

- `data/reference-responses/layer-stack-p1-v1/`；
- `artifacts/corpus/layer-stack-p1-v1.json`；
- `assets/`、`references/`、`external/`。

删除旧 artifacts 会失去本机 checkpoint/capture，之后只能重新训练/导出；v5 corpus 与 source/reference provenance 不受影响。
