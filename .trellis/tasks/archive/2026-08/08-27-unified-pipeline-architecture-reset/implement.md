# 统一 Pipeline 架构重整实施计划

## 完成定义

本任务是一次原子迁移，不以“新接口已添加但旧路径仍可达”为完成。只有以下全部成立才可收尾：

- offline/live 数据、训练、checkpoint、部署、viewer 都各自只有一个公共实现；
- LayerStack/OpenPBR/MERL/MaterialX reference 与 NVIDIA 都通过同一 scattering contract、package loader、PT/deferred renderer；pbrt coated probe 保持为 pipeline 外部的 LayerStack 两界面交叉验证工具，不注册新 source/reference runtime。
- viewer 两个 slot 对称、固定 50/50，失败/resize 不改变 aspect；
- 旧方法、旧 MethodBundle/compiled-set、独立 sampler/mollification pipeline 已递归删除；
- specs/docs、CLI、配置、测试和构建输入全部迁移；
- 完整开发机质量门禁通过，或按开发机状态把不能运行的命令写入根 `TESTING.md`，不做虚假验证声明。

不拆成独立 child task：数据、runtime 与 viewer 的接口切换必须在同一任务内完成，避免产生可被误用的中间双轨。实施中可按以下 rollback point 分段提交，但不在中间阶段宣称任务完成。

## 0. 实施前环境与最小可行性 probe

- [x] 执行 `trellis-before-dev`，重新读取 project/core/data/learning/viewer 对应 spec。
- [x] 按 `.trellis/spec/project/dev-environment.md` 判定本机为完整、仅 GPU 或静态状态；第一次验证回复记录证据。
- [x] 记录根仓库与 `external/Falcor` 初始 dirty state，只保护用户已有改动，不回滚无关文件。
- [x] 在本任务 `scratch/` 中建立 Falcor/CUDA interop probe：Shared buffer → compute write → `to_torch()` → CUDA loss/backward；断言无 `to_numpy()`。
- [x] 建立 package shader probe：通用 host pass 从临时 package 的绝对路径加载一个未列入 viewer CMake 的 test module，并解析稳定 scattering ABI。
- [x] 若任一 probe 失败，先分类为构建能力、同步、module closure 或 ABI 问题；不得降级为 CPU readback或恢复预编方法枚举。

Rollback point A：两个基础能力 probe 均通过，尚未切换产品入口。

## 1. 冻结公共合同与目录所有权

- [x] 更新 `.trellis/spec/project/`、`core/`、`data/`、`learning/`、`viewer/`，移除 lobe/Film/legacy/MethodBundle v1 陈旧主线，写入本任务批准的合同。
- [x] 更新 `docs/architecture.md` 的目标目录与跨层数据流；同步 `docs/contracts/` 中 scattering、checkpoint、package 和 viewer 合同。
- [x] 在 core 定义 source-family、reference-program、reference-execution、training-batch、method-definition、scattering-package、binding/capability 的版本身份与所有权。
- [x] 冻结 `SourceParameterView@1`/`SourceEditPatch@1`/`SourceEditResult` 的 typed tree、稳定 path/element ID、list/variant operation、binding provenance、可编辑性、冲突检查、invalidation 与 canonical hash 合同；明确它不是通用材质 IR。
- [x] 冻结 `SourceAdaptationContract`：方法按 source contract/schema version 声明参数域，并对 edit 返回 `unchanged/runtime-patch/recompile/unsupported`。
- [x] 冻结方向、response measure、PDF、sample event、source identity、runtime/material/package identity 的 canonical hash 规则。
- [x] 为 reference 建立迁移/排除矩阵：LayerStack/OpenPBR/MERL/MaterialX 四个 pipeline/viewer-ready ground-truth reference 全部进入 package/binding；pbrt 明确排除于 source/method/runtime/viewer discovery，只保留现有 external/tool/manifest/artifact 所有权边界。
- [x] 加入 test-scoped contract fixture；确认它不进入产品 registry/CLI/config。

主要风险文件：`.trellis/spec/**`、`docs/architecture.md`、`docs/contracts/**`、`src/ncls/core/**`、`shaders/ncls/contracts/**`。

## 2. 统一 reference query 与 batch source

- [x] 把 LayerStack source-family 语义从 Falcor evaluator 类中拆出为 source definition/reference program；Falcor 类只保留 execution/transport。
- [x] 把 OpenPBR、MERL、MaterialX provider 迁到同一 `SourceFamilyDefinition`/`ReferenceProgramDefinition` 与 query stream，保留各自原生资源、编辑、插值/颜色空间和 parity oracle。
- [x] 为 LayerStack 实现层级/list/variant editor schema 与 coat `insert/remove/move`；为 OpenPBR 实现原生命名 typed parameter/binding schema；为 MaterialX 实现 constant editable/connected read-only；为 MERL 返回无伪造连续参数的编辑面。
- [x] 把 scene/corpus 的 source state 读写收敛为 family-owned canonical snapshot；parameter view 每次由 snapshot 生成，不存第二份 truth。
- [x] 保护 pbrt 对照边界：不修改 `external/pbrt-v4`，不将 `tools/reference/pbrt_probe`/`pbrt_compare.py` 抽成新 runner。若 LayerStack 正式 API/路径迁移破坏其 import 或调用，只修正最小调用点与必需 manifest hash；否则不改它。
- [x] 定义唯一 `ReferenceQueryStream` 与 `TrainingBatch` schema；把方向采样、mollification/target estimator、seed/sharding 表达为 recipe 组件。
- [x] 让现有 offline collector 接同一 query stream 与唯一 shard sink；保留正式 corpus identity/role 隔离。
- [x] 实现 `OfflineBatchSource`，读取后生成统一 device batch。
- [x] 实现 `LiveReferenceBatchSource`：Falcor Shared buffer、`from_torch()`/`to_torch()`、明确同步与 lease；target 不落 HDF5、不经过 NumPy/CPU。
- [x] 让同一训练配置只通过 `batch_source` 选择 offline/live，其他 runner/objective/evaluation 配置不分叉。
- [x] 删除 mollification 专用 collector、budget/lock schema、store/reader/CLI 与 v2…v8 配置链；迁移后保留的 recipe 使用非历史相对命名和单一 schema。

目标文件范围：`src/ncls/data/**`、`src/ncls/learning/data.py`、`configs/corpus/**`、`src/ncls/cli.py`、`tests/unit/test_*data*`、新增 GPU contract test。

Rollback point B：offline/live schema parity 与真实 CUDA one-step smoke 通过；旧 collector 尚未成为默认入口时可整体回退。

## 3. 统一方法注册、训练与 checkpoint

- [x] 新建唯一 product method discovery/registry；产品 registry 只发现 NVIDIA `MethodDefinition`。
- [x] 把 NVIDIA model、objective、sampler head、runtime/material compiler 迁入一个 definition/recipe。
- [x] 为 NVIDIA 声明真实的 source adaptation domain；对未编译 source state 明确返回离线 `recompile` 或 `unsupported`，不伪造 runtime-editable latent。
- [x] 扩展 test-scoped contract fixture，覆盖一个 `runtime-patch` 和一个 `recompile` adapter，证明新 method 适配 source 参数不需改通用 runner/package/viewer。
- [x] 收敛为一个 `TrainingRunner`，支持 recipe phase、parameter group、多个 loss/head；删除 concrete-model `isinstance` 分支。
- [x] 合并 evaluator 与 sampler checkpoint lifecycle；sampler/head 不能再由独立 runner/CLI 产生不相干身份。
- [x] 建立 `TrainingCheckpoint@2` 公共 envelope、tensor schema 校验、atomic writer/reader、hash 与 recovery。
- [x] 统一 `ncls learn train/evaluate/export` 命令；方法、data source 和 phase 来自 config/descriptor，不增加 method-specific 子命令。
- [x] 保持当前 NVIDIA diagnostic 身份和“非论文忠实复现”说明可追溯；旧 checkpoint 不设兼容 reader，必要证据由 Git/artifacts 历史保留。

目标文件范围：`src/ncls/learning/methods/**`（或设计确定的单一方法目录）、`pipelines/**`、`models/**`、`training/**`、`evaluation/**`、`src/ncls/cli.py`、`configs/learning/**`。

## 4. 替换 MethodBundle/compiled-set 为统一 package

- [x] 实现 `ScatteringPackage@1` manifest/schema、safe URI、typed blob descriptor、三个独立 identity 与 tamper rejection。
- [x] 实现唯一 package writer：消费 method/reference runtime payload 与 material payload，公共代码拥有目录、文件、hash、provenance 和 parity 布局。
- [x] 实现唯一 Python loader；LayerStack/OpenPBR/MERL/MaterialX reference package 与 NVIDIA package 走同一入口。
- [x] package method module 只依赖稳定 host ABI；打包完整的 method-private module closure。
- [x] 从 immutable `TrainingCheckpoint@2` 经一个 deployment compiler 生成 NVIDIA runtime + 一个或多个 material asset；source reference 不经过训练 checkpoint，但生成相同 package schema。
- [x] 删除 `MethodBundleManifest`、`export_compiled_set_bundle`、unified/NVIDIA exporter 与 method-specific Slang session/hash 分支。

目标文件范围：`src/ncls/bundle/**`（迁移后按新语义重命名）、`src/ncls/learning/*artifacts.py`、`src/ncls/learning/slang/**`、schemas、CLI、tests。

Rollback point C：Python package roundtrip、NVIDIA/reference/test fixture 三者 parity 与 identity 测试通过；尚未切 viewer loader。

## 5. 统一 reference 与 neural scattering runtime

- [x] 保留/收敛 `INclsScatteringBackend` 与 state 接口，补足 package host ABI、sample event、capability 与私有 payload binding。
- [x] 把 LayerStack random-walk、OpenPBR、MERL 与 MaterialX reference 分别包装为同一 backend/state contract；source-family 私有 dispatch 不进入 renderer。
- [x] 把 NVIDIA runtime module 迁为 package module，移除 viewer 源码树预编依赖。
- [x] 对四个 viewer-ready reference 与 NVIDIA 分别按其权威 oracle 验证 `evaluate/sample/pdf` 方向、measure、有限性、sample→pdf 和 estimator 权重；MaterialX 保留既有 upstream image parity。
- [x] 明确 LayerStack 多界面无 marginal PDF 时的 capability/sample event；通用 integrator 禁止伪 MIS。
- [x] 删除 lobe residual、unified candidate 等旧 neural backend，以及只服务废弃方法的 legacy LTC/analytic control 身份与接线；仅把仍被 reference/NVIDIA/公共数学真实使用的函数迁入中性模块。

目标文件范围：`shaders/ncls/contracts/**`、`shaders/ncls/reference/**`、`shaders/ncls/backends/**`、Python scattering descriptors/tests。

## 6. 重构 viewer 为两个对称 slot

- [x] 将 `NclsViewer.cpp` 按职责拆成 package loader、slot controller、renderer、capture/replay 和 UI 语义单元，避免继续扩张单体文件。
- [x] C++ 通用 loader 从四个 reference 与 NVIDIA package 的真实 module 路径创建 `ScatteringBinding`；删除 backend ID/LayerStackIR/method shader tree/按 reference family 的 renderer 分支。
- [x] 从现有 `ReferencePathTracer.cs.slang` 提取唯一 `ScenePathIntegrator.cs.slang`，所有 source-family scattering 通过 binding specialization 调用。
- [x] 建立唯一 deferred renderer（prepare/evaluate 可保持两个语义 pass），任一 slot/binding 都能实例化。
- [x] 建立 `ComparisonSlot[2]`：独立 selection、PT/deferred mode、capability、status、resource、accumulation、timing；UI 与 host 不硬编码左右角色。
- [x] 建立一个 schema-driven source editor：只渲染 `SourceParameterView` 并提交 `SourceEditPatch`，删除 `renderOpenPbrUi`/`renderMaterialXUi`/LayerStack 手写 UI 及 77/24-float offset 改写。
- [x] 建立 source edit transaction：新 canonical snapshot 成功后独立 rebind 两个 slot，按 adaptation result 进入 `ready/compiling/unsupported/error`，只有新 material asset 完整验证后才原子换入。
- [x] 固定 50/50：相同 panel width/height、奇数宽度固定 divider、camera aspect 与 availability 解耦、composite 1:1 sampling。
- [x] 加载/编译/能力失败在原 slot 显示错误，不 resize 另一 slot；修复 bundle 重扫导致的旧活动状态丢失根因。
- [x] capture/replay/benchmark 迁为 `slots[2]` schema；删除 split、reference/approximation 命名和旧 schema reader。
- [x] 更新 `scripts/benchmark_viewer.ps1` 与 viewer presets，只使用新 package/slot/mode 合同。

目标文件范围：`apps/viewer/**`、`patches/falcor-viewer-overlay.patch`（仅在构建列表语义需要时更新）、`configs/viewer-*.json`、`scripts/benchmark_viewer.ps1`、viewer tests。

Rollback point D：新 viewer 已能以同一 PT source 完成 reference/reference 与 reference/NVIDIA，旧 viewer 入口随后立即删除。

## 7. 递归删除与入口切换

- [x] 删除 Film、analytic residual、per-state teacher、lobe residual、unified candidate 等废弃 neural method 的 product 源码/配置/shader/测试/CLI/文档入口。
- [x] 删除只服务上述方法的 legacy LTC/analytic control 产品身份；若有被 NVIDIA/reference 真实使用的数学组件，确认已迁入中性模块。
- [x] 确认 LayerStack/OpenPBR/MERL/MaterialX reference registry package、原生资源语义和 parity 证据均被迁移而未删除；pbrt 仍是 pipeline 外部 independent crosscheck，上游、tool、manifest 与输出边界没有被误删或误注册。
- [x] 删除 sampler runner/config、method-specific exporter/session/parity/audit CLI 中已被公共实现覆盖的部分。
- [x] 删除 MethodBundle v1/compiled-set、right-only approximation、reference-only transport、split 及旧 capture/replay schema。
- [x] 递归删除旧 checkpoint/MethodBundle/CLI/capture/replay 的 reader、writer、alias、converter、schema/version 探测、fallback 与兼容测试；历史只由 Git/仓库外 artifacts 追溯。
- [x] 全仓库搜索旧 identity、schema、命令、路径和描述；分类每个命中为历史归档允许项或必须删除项。
- [x] 更新 active Trellis 旧任务状态为 superseded，并指向本任务；不改写其历史研究内容。
- [x] `docs/`、`.trellis/spec/`、README/CLI help/配置示例只描述新架构与 NVIDIA 当前身份。
- [x] 确认根仓库没有新增 artifacts/data/build/cache，`external/Falcor` 恢复干净。

## 8. 验证矩阵

实际执行前先依据开发机状态调整；所有 Python 命令只用 `neural-shading` 环境。

### 静态与单元测试

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_training_batch.py tests/unit/test_method_definition.py tests/unit/test_training_checkpoint.py tests/unit/test_scattering_package.py
conda run -n neural-shading python -m pytest tests/unit/test_source_parameter_editor.py tests/unit/test_viewer_slots.py tests/unit/test_viewer_studio.py
conda run -n neural-shading python -m pytest tests/unit
git diff --check
```

### Falcor/CUDA 与 runtime parity

```powershell
& scripts/run_falcor_python.ps1 -m pytest tests/gpu/test_live_reference_batch.py
& scripts/run_falcor_python.ps1 -m pytest tests/gpu/test_scattering_package_parity.py
```

验收证据必须包含：`TrainingBatch` 全 tensor 在 CUDA、无 host readback/HDF5 调用、offline/live schema parity、reference/NVIDIA packed parity、test fixture 不触发公共代码修改。

reference 证据必须逐项列出 LayerStack、OpenPBR、MERL、MaterialX 的 source identity、package/binding、适用 parity 与 viewer PT/deferred 状态。pbrt 只验证所有权边界：锁定 upstream 保持 clean，tool/manifest 未被误删，新 pipeline discovery 不列出它。如果本任务修改了 compare tool 调用点，再使用其原有命令运行 smoke；不把 numerical parity 提升为本架构任务的新门槛。

### Viewer 构建与 headless 组合

```powershell
& scripts/build_viewer.ps1 -Configuration Release
& scripts/benchmark_viewer.ps1 -PackageRoot artifacts/exports/unified-pipeline-smoke -Preset configs/viewer-benchmark-v1.json -OutputDirectory artifacts/benchmarks/unified-pipeline-smoke
git -C external/Falcor status --porcelain
```

headless matrix 至少包含：

- reference/PT | reference/PT；
- reference/PT | NVIDIA/PT；
- NVIDIA/PT | reference/PT；
- reference/PT | NVIDIA/deferred；
- NVIDIA/deferred | reference/PT；
- package missing、hash/ABI/module error、PT capability missing；
- 偶数/奇数宽度与窗口 resize。
- LayerStack 值编辑/增删重排、OpenPBR constant、MaterialX constant/connected read-only、MERL 无连续编辑；覆盖 reference 重绑定、NVIDIA `unsupported/recompile` 和 fixture `runtime-patch` 原子切换。

每项检查两个 slot extent 相同、camera aspect 相同、失败不改变另一侧、composite 无非等比采样、capture/replay roundtrip 完整。

上述 `reference` 组合至少分别用 LayerStack、OpenPBR、MERL、MaterialX 各执行一次；reference/reference 另取两个不同 source family 的组合，证明 slot/renderer 不按 family 分支。

### 全量门禁与清理搜索

```powershell
conda run -n neural-shading python -m pytest
rg -n "film-evaluator|analytic-residual|per-state-teacher|lobe-residual|legacy_ltc_k2|unified-neural|ncls.method-bundle|compiled_set|train-sampler|export-unified|export-nvidia|gSplit|mSplit" src shaders apps configs scripts tests docs .trellis/spec
git status --short
git -C external/Falcor status --short
```

`rg` 命中不能机械要求为零：归档迁移说明或明确的禁止项测试可以保留；每个命中必须在最终清理矩阵中解释。产品注册、CLI、构建、源码与稳定主线文档中不得可达。

## 最终收尾检查

- [x] PRD 每条验收项都有测试、capture 或 reachability 证据。
- [x] 没有通过缩减功能、复用 reference sampler 或 CPU readback替代用户要求。
- [x] NVIDIA 论文忠实性仍按真实训练 lifecycle/预算标注，不因 pipeline 跑通而改称忠实复现。
- [x] 四个 ground-truth reference 已完整迁移；pbrt independent crosscheck 仍在现有 pipeline 外部边界，没有被迁入、误删或冒名成 LayerStack runtime。
- [x] 方法/数据/训练/package/viewer 五个入口各只有一个稳定实现。
- [x] viewer UI、capture、日志清楚显示两个 slot 的 program/runtime/material/mode/status。
- [x] viewer UI 通过 source-family schema 保留所有原生可编辑参数；方法适配状态明确，不将 stale neural 结果当作同一 source state 的有效比较。
- [x] 所有旧路径删除完成，未留下 alias、reader、converter 或 fallback。
- [x] 运行 `trellis-check` 做最终 spec、lint/type、tests、跨层数据流与复用检查。
- [x] 运行 `trellis-finish-work` 前向用户报告改动、验证状态、无法运行的命令和任何剩余风险。
