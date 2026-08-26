# 统一 Pipeline 架构重整

## 目标与用户价值

一次性重整 neural material 的 source/reference、数据采集、训练、checkpoint、部署和 viewer 架构，形成一条不可分叉的公共 pipeline。新增 neural method 只实现受约束的表达、训练 recipe 与 runtime payload；source family 只实现原生 source/reference program；二者都不复制 runner、文件 I/O、viewer pass 或跨层接线。当前已接入训练/viewer pipeline 的 LayerStack、OpenPBR、MERL 与 MaterialX reference 全部迁入新架构；只作为外部交叉验证 oracle 的 pbrt coated probe 不被提升为 pipeline participant。

本任务同时清除先前错误方法与重复基础设施，并把 viewer 改成两个完全对称、固定 50/50 的 comparison slot。reference material 与 neural method 对 viewer 是同一 scattering program 的等价替换；每侧都能独立运行同一个 PT renderer 或同一个 deferred renderer。

## 背景与已确认问题

当前 NVIDIA 方法虽未忠实复现论文训练 lifecycle，但已证明现有端到端链路可以运行。问题在于这条链路不是稳定的公共架构：

- `src/ncls/learning/pipelines/__init__.py:3` 仍注册 lobe、P1、unified 与 NVIDIA 多组方法；旧 Film、analytic residual、per-state teacher、lobe residual 和 legacy LTC 仍有可达入口。
- `src/ncls/learning/runner.py:102`、`src/ncls/learning/training/sampler_runner.py`、`src/ncls/learning/slang/session.py`、两个 method-specific exporter 和 CLI 按具体方法分支；新方法必须多处接线。
- `src/ncls/data/contract.py:302` 已有 reference provider 语义基础，但 LayerStack corpus 与 mollification 又形成独立 collector/schema/reader/CLI；当前 Falcor evaluator 在 `src/ncls/data/reference.py:135` 使用 `to_numpy()`，不是 GPU-resident online training。
- `src/ncls/bundle/compiled_set.py:11`、`:96` 把公共部署路径固定到 `MaterialProgram`/LayerStack；`:137` 又把具体 material state 纳入 `method_id`。
- `apps/viewer/MethodBundle.cpp:90` 固定要求 LayerStackIR；`:133` 虽然 bundle 内复制 shader，实际仍要求同一 shader 预编进 viewer 源码树。
- `apps/viewer/NclsViewer.h:192` 只有 reference PT；右侧只有 prepare/approximation deferred。`ReferencePathTracer.cs.slang:300`、`:523`、`:587` 自行分派 reference evaluate/sample/pdf，并拥有独立 scene transport loop。
- `apps/viewer/NclsViewer.cpp:506`、`:681` 把 viewport/camera aspect 与 method availability 耦合；方法重扫在 `:831` 替换列表后才于 `:915` 读取活动状态，可能跳过 resize。`Composite.cs.slang:38` 又把固定半宽纹理映射到可拖动 split，形成第二个拉伸根因。

完整证据与删除清单见 `research/current-state-audit.md`。

## 已冻结决策

1. 本次交付真实的 GPU-resident online training 最小链路，不只声明未来接口。target 从 Falcor shared GPU buffer 直接进入 CUDA tensor/loss，不经过 `to_numpy()`、HDF5 或 CPU replay。
2. LayerStack 是当前 source family；Falcor 是 execution backend。当前 live source 组合 LayerStack random-walk reference program 与 Falcor GPU executor，但公共类型与 batch schema 不混合这两个层次。
3. 采用三层产物边界：公共 `TrainingCheckpoint` → `MethodRuntime + CompiledMaterialAsset` → 统一 viewer deployment package。viewer 不读取裸 `.pt/.pth`。
4. 当前 `MethodBundle v1` 被替换。所有 reference/neural program 使用相同的版本化 package/manifest/blob 格式；method runtime、material asset 与 package 分别具有独立 identity。
5. reference 与 neural method 都实现同一 renderer-facing `prepare/evaluate/sample/pdf` 合同，并从同一个 source material identity 派生。私有 `CompiledMaterial`/`ScatteringState` 布局不进入公共 ABI。
6. viewer 两个 slot 完全对称：每侧均可选择 reference 或任意 neural method，每侧均可按 capability 独立选择 PT/deferred。默认 preset 只是初始选择，不定义左右角色。
7. 删除可拖动 split，固定为两个严格 50/50、相同 render extent 和相同 camera projection 的 viewport；任一侧失败也不改变布局。
8. 产品 method registry 只保留当前 NVIDIA 方法。第二种表达结构只用 test-scoped contract fixture 证明扩展性，不作为产品方法、配置、CLI 或 viewer 默认项。
9. reference 不属于 neural method 清理对象。LayerStack random-walk、OpenPBR 1.1.1、MERL 与 MaterialX/Poly Haven 四个 pipeline/viewer-ready ground-truth reference 全部保留并迁入统一 scattering/package/viewer 合同。pbrt coated crosscheck 只是 `ncls.layer-stack@1` 两界面 coated slice 的外部独立 oracle，不是 LayerStack 的另一个产品 runtime；它保持在锁定 `external/pbrt-v4/`、`tools/reference/`、`references/pbrt-coated-crosscheck-v1/` 与 `artifacts/` 的现有所有权边界中，不进入新 method/source discovery、training batch、deployment package、viewer 或新造的统一 validation runner。
10. “公共层不看到 family 私有字段”不等于“source 参数不对外开放”。source family 必须通过统一、版本化、自描述的 editor contract 暴露其原生可编辑表面；viewer 只渲染 schema 和提交 edit patch，不理解 LayerStack/OpenPBR/MaterialX 语义。该 schema 不得成为所有 source family 必须归约的通用材质表示。
11. source 编辑不受当前 neural method 能力阻塞。编辑成功后 reference 立即改用新 canonical state；neural slot 若需重编译且已启动可用 compiler 则进入 `compiling`，若不支持或当前只能离线重编译则进入带明确原因的 `unsupported`。旧 neural 图像、difference 和比较统计立即失效；只有对同一新 source state 的 asset 完整验证后才原子换入。
12. 旧 checkpoint、MethodBundle/compiled-set、CLI、capture/replay 与方法 identity 不保留任何兼容层。旧 reader、alias、converter、schema/version 自动探测、fallback 和隐藏双轨必须在调用方迁移后递归删除；历史证据只由 Git 与仓库外 artifacts 追溯。

## 需求

### R1. Source/reference 与执行后端正交

- source family 保留原生 source state、资源、参数编辑、canonical identity 与权威 reference；公共层不要求归约为 LayerStack、closure 或某个 backend state。
- reference program 定义 GT 数学/随机语义，execution backend 只负责任务提交、buffer、同步和 tensor transport。
- 新 source family 不能新增 collector、trainer、checkpoint writer、viewer integrator 或 renderer pass。
- 当前四个 viewer-ready reference 分别实现自己的私有 source payload 与 reference scattering program，但通过同一 `ReferenceProgramDefinition`、query/batch 合同、deployment writer/loader 和 viewer binding 进入公共 pipeline。
- pbrt coated crosscheck 与 LayerStack random-walk 的两界面 slice 共享物理语义，但 LayerStack random-walk 不是 pbrt 代码的 Falcor port；二者是对同一 coated 构造的独立实现。pbrt 只保留为 pipeline 外部对照工具，不实现 `SourceFamilyDefinition`、`ReferenceProgramDefinition`、`ScatteringPackage`、viewer binding 或新 pipeline runner。若 LayerStack 正式 API 路径因本任务变化，只对 pbrt compare tool 做保持可用所必需的最小调用点更新，不借机将它迁入架构。

### R2. 单一数据采集与 TrainingBatch

- offline collector 与 live training 使用同一个 reference query stream、方向/measure/seed/sharding 语义和 typed `TrainingBatch`。
- persisted corpus reader 与 live reference executor 是可替换 batch producer；offline collection 是同一 query stream 加 shard sink。
- 方向扰动、mollification、target estimator 与 curriculum 只能作为通用 recipe 组件，不拥有独立 manifest/reader/CLI pipeline。
- live producer 必须返回 CUDA tensor，并通过明确同步/lifetime 合同直接进入同一个 training runner。

### R3. 单一方法接口、训练与 checkpoint

- neural method 只提供一个 `MethodDefinition`：descriptor、trainable/expression、training recipe、tensor state schema、runtime payload compiler 与 material payload compiler。
- evaluator、matched sampler 与可选 head 属于同一 trainable/recipe lifecycle；普通训练和 sampler 训练不得使用两个 runner。
- registry、CLI、optimizer loop、checkpoint selection、evaluation、checkpoint writer/reader 不按方法名或 concrete model 类型分支。
- 所有方法使用同一版本化 checkpoint envelope。方法只导出/恢复受 schema 约束的 tensor mapping，不自行读写文件。

### R4. 单一部署格式与 loader

- 公共 deployment writer/loader 拥有目录、manifest、typed blobs、safe URI、hash、ABI/capability 校验、provenance 和 parity；NVIDIA 与四个 viewer-ready reference 都使用该入口。
- `program_runtime_id` 不含具体材质；`material_asset_id` 绑定 source/compiler/payload；`package_id` 组合 runtime 与 assets。
- package 内 method-specific shader/module 是 viewer 实际加载源；新增方法不修改或重编 viewer host/CMake。
- 公共 manifest/loader 不解析 LayerStack、NVIDIA tensor 名或方法私有 state 字段。

### R5. Reference/neural 等价替换合同

- reference material binding 与 compiled neural material binding 都由公共 loader 产生同一 host ABI，并实现同一 `prepare/evaluate/sample/pdf` 语义。
- 两个 binding 必须可追溯到同一个 source material identity；不能比较两个无关 preview material。
- PT capability 必须由被选 program 自己提供匹配的 `sample/pdf`；不能借用 reference 或另一方法的 sampler。
- capability/sample event 必须如实表达 marginal PDF/MIS 限制，不能伪造 PDF 来满足形式接口。

### R6. 单一 PT、单一 deferred、对称 slot

- 仓库只保留一个 scene PT path loop。reference 与 neural method 只是同一 PT source 的不同 scattering specialization。
- 仓库只保留一个 deferred renderer lifecycle，可对任一 slot/binding 实例化。
- 两个 slot 使用相同 selection、mode、loading、status/error、resource、timing、capture/replay schema。
- 每个 slot 的 mode 根据 binding capability 启用；能力不足或编译/加载失败时在原 slot 显示明确状态。

### R7. 固定尺寸与显示正确性

- `panel_width = floor(output_width / 2)`，两侧使用完全相同的 `(panel_width, output_height)`；奇数总宽度剩余像素作为固定 divider/background。
- camera aspect 只由 panel extent 决定，与 selection、mode、capability、package 状态无关。
- composite 只做 1:1 对应与共同显示变换，不做非等比 UV remap。
- 删除 split UI/state/shader/replay/capture 字段；失败侧不得让成功侧铺满全窗。

### R8. Neural/approximation 清理、reference 迁移与稳定文档

- 删除 Film、analytic residual、per-state teacher、lobe residual、unified candidate 等废弃 neural method 的产品注册、模型、配置、CLI、shader、测试和稳定文档入口。
- legacy LTC/analytic control 等旧 approximation/backend 若只服务废弃方法则删除；若其中某个数学组件仍被 NVIDIA 或 reference 真实使用，只迁移该通用组件并移除旧方法身份、目录和接线。
- 删除独立 sampler runner/config/CLI、mollification 平行基础设施、method-specific exporter/session、MethodBundle v1/compiled set 和左右专用 viewer 路径。
- LayerStack random-walk、OpenPBR、MERL、MaterialX 四个 ground-truth reference 的原生语义、资源与验证证据必须保留，并全部迁入新 source/reference/scattering/package/viewer 路径；不得因删除旧 viewer reference dispatch 而删掉 reference 本身。
- pbrt coated crosscheck 的锁定上游、长期 compare tool、身份/适用范围 manifest 和历史观察保持在既有 `external/`、`tools/reference/`、`references/` 和 Git/artifacts 边界；它不是新架构迁移对象，也不是 neural 方法清理对象。仅在 LayerStack 公共 API 迁移导致 compare tool 无法调用时修正最小边界。
- 复用仍有真实调用方的公共数学/hash/packing/reference 组件；不能用“以后可能需要”保留旧产品身份。
- `.trellis/spec/`、`docs/`、CLI help、配置和 tests 只描述新合同；旧活动任务标为 superseded，历史归档原字段不改写。
- 旧 checkpoint/MethodBundle/CLI/capture/replay 的调用方在同一任务中切换到新合同，随后删除旧 schema、writer/reader、alias、converter、格式探测、fallback 和兼容测试；不能只从 UI 或默认配置隐藏。

### R9. Source 原生参数暴露与 method 编辑适配

- 每个 `SourceFamilyDefinition` 对当前 immutable source snapshot 返回 `SourceParameterView@1`：它使用稳定 parameter path/element ID 描述 group、list、variant、scalar、bool、enum、vector/color、resource 与 read-only 节点，并附带值类型、单位、范围、步长、显示提示、binding provenance、可编辑性与不可编辑原因。
- 公共 `SourceEditPatch@1` 支持 typed `set/insert/remove/move/replace-variant` 操作，以便 LayerStack 可增删、重排 coat，而不只能编辑扁平 float。family 自己验证 patch，产生新 canonical source snapshot/state ID、changed paths、invalidation 和 diagnostics；viewer 不得直接改写 backend packet 或固定 float offset。
- LayerStack 暴露原生 interface/media 列表与参数；OpenPBR 暴露原生命名参数及 Constant/Texture/Graph/Geometry binding 状态；MaterialX 暴露可编辑 constant input，对 connected input 显式 read-only，不把图连接静默替换为 constant；MERL 可以没有连续可编辑节点，选择另一测量表属于 source asset 切换。
- reference binding 必须能正确绑定每个合法新 snapshot。`MethodDefinition` 按 source contract/schema version 声明 `SourceAdaptationContract`，对 edit 返回 `unchanged | runtime-patch | recompile | unsupported`；method 的 material compiler 消费完整 immutable source snapshot，不依赖 viewer 为它解释私有参数。
- 当前 NVIDIA 方法不必为了满足公共 UI 而伪造运行时可编辑 latent。它可以对未编译 source state 诚实声明 `unsupported`/需要离线 `recompile`；以后的 neural method 可在同一 contract 下实现 runtime patch 或增量/完整重编译，不改 viewer UI、loader、renderer 或 capture schema。
- 数据与训练使用 source snapshot/state identity 和 method 声明的 source adapter/features，不把 editor schema 强制展平为全方法共享的 neural 输入向量。
- source edit 成功后不因 neural method 的适配能力回滚或禁用。reference 重绑定新 state；neural slot 按 adaptation result 进入 `ready/compiling/unsupported/error`。旧 neural output 不得作为新 state 的有效比较，新 asset 必须在 source identity、schema、hash 和 parity 验证后原子换入。

## 验收标准

| ID | 类型与来源 | 可观察验收 |
| --- | --- | --- |
| A1 | 需求交付：用户要求新方法无复杂接线 | NVIDIA 与一个 test-scoped 不同布局 fixture 通过同一 registry/runner/checkpoint/package 路径；加入 fixture 不修改公共 runner、CLI、writer、viewer C++/CMake/UI。 |
| A2 | 需求交付：用户确认真实 online | 同一训练命令只切换 batch source 即可跑 offline/live；live target 为 CUDA tensor，执行路径没有 `to_numpy()`、HDF5 或 CPU replay，完成有限 loss/gradient/optimizer one-step。 |
| A3 | 需求交付：单一训练 | 产品代码只有一个 training orchestration；sampler/head 使用同一 recipe/checkpoint lifecycle；通用代码不存在 NVIDIA/concrete model 分支。 |
| A4 | 需求交付 + 语义正确性：三层产物决策 | checkpoint 只由公共 reader/writer处理；runtime/material/package identity 独立；NVIDIA 与 LayerStack/OpenPBR/MERL/MaterialX reference 都由同一 package writer/loader roundtrip 并拒绝 tamper/ABI/hash 错误。 |
| A5 | 理论/语义正确性：公共 scattering contract | 四个 viewer-ready reference 与 NVIDIA 分别通过其适用的 `evaluate/sample/pdf` 方向、measure、有限性和 estimator correctness；PT 不使用另一 binding 的 sampler。容差在正式结果前由 dtype/oracle 冻结。 |
| A6 | 需求交付：完整双 PT 与独立 deferred | 同一 PT shader source 能在任一 slot 实例化四个 reference 或 NVIDIA，并至少完成 reference/reference、reference/NVIDIA、NVIDIA/reference；公共 deferred 完成 PT/deferred 与 deferred/PT；capture 明确记录每个 slot 的 program/material/mode/integrator。 |
| A7 | 需求交付：等价替换 | 从同一 source identity 生成 reference/neural binding；替换 binding 不修改 scene、camera、light、integrator、RenderGraph 或 display path。 |
| A8 | 需求交付：长宽比修复 | 偶数/奇数宽度、resize、任一 package missing/hash/ABI/shader/capability failure 下，两侧 extent/camera aspect 保持合同，composite 无非等比缩放，另一侧不铺满。 |
| A9 | 需求交付：彻底清理且不误删 reference | 废弃 neural/approximation 方法在产品 registry、CLI、源码、shader、CMake、配置、测试夹具与稳定文档均不可达；LayerStack/OpenPBR/MERL/MaterialX 通过新 renderer 合同；pbrt 仍位于锁定 `external/` + `tools/reference/` + `references/` 对照边界，不被删除也不出现在新 source/method/runtime/viewer discovery 中；旧 checkpoint/MethodBundle/CLI/capture/replay 的 reader、alias、converter、schema 探测和 fallback 不存在；全仓库每个残余旧 identity 命中都有历史归档或禁止项测试解释。 |
| A10 | 工程正确性：项目 Trellis/环境合同 | 目标 unit/GPU/viewer headless matrix、全量 pytest、viewer Release build、external/Falcor clean 与 `trellis-check` 通过；无法执行的项目按开发机状态写入 `TESTING.md`，不宣称已验证。 |
| A11 | 需求交付：source 参数通用暴露与 neural 适配 | 同一通用 viewer editor 不按 family/method 分支地完成 LayerStack 数值+列表结构编辑、OpenPBR 原生命名参数编辑、MaterialX constant 编辑/connected read-only 与 MERL 空编辑面；每次合法编辑生成新 source state ID，reference 重绑定，neural slot 严格按 adaptation result 处理；`compiling/unsupported` 时旧 output/difference/statistics 失效，同 source state 的新 asset 验证后原子换入；新增 test fixture 的 runtime patch/recompile 支持不修改 viewer C++/UI。 |

所有 quality/time/memory 数值属于 observed result 或既有部署成本合同，不用 NVIDIA 当前质量、历史 run 数值或论文结果作为本架构任务的成功门槛。

## 范围内

- 上述跨层公共合同、真实 online one-step、NVIDIA 迁移、LayerStack/OpenPBR/MERL/MaterialX 四个 viewer-ready reference 的 renderer 迁移、pbrt 外部对照边界保护、统一 package、source parameter editor/adaptation contract、reference/neural 等价 viewer、PT/deferred 对称 slot、固定尺寸、旧格式/兼容层与重复基础设施的递归清理。
- 为验证扩展点所需的 test-scoped fixture、contract/parity/failure/resize/capture 测试。
- 同步更新稳定 specs/docs/CLI/config/build scripts 和旧活动任务状态。

## 范围外

- NVIDIA 论文规模训练、最终数值忠实性结论或自动扩大训练预算；本任务保留其当前 diagnostic 事实并建立真实 online lifecycle。
- 新增第五种 source family、UE 集成、多灯 scaling 研究、PT 方差研究或额外 neural candidate。
- 将 pbrt coated probe 改造为新架构的 source/reference runtime、通用 validation runner、deployment package 或 viewer binding；它保持现有专用外部 oracle 边界。
- 保留或新增旧 checkpoint、MethodBundle、capture/replay、CLI 或方法 identity 的 alias、converter、legacy reader、格式探测或 fallback。这些兼容层的删除明确属于范围内。
- 修改锁定的 `external/Falcor` 上游源码；使用现有 CUDA/DLPack 与绝对 shader path 能力。

## 规划状态

用户拥有的目标、范围、reference 角色/迁移、兼容清理边界、产物、source editor/adaptation 与 viewer UX 决策已全部收敛，无阻塞问题。技术设计见 `design.md`，原子实施与验证顺序见 `implement.md`。本文档已完成最终 PRD convergence pass；需要用户对最新完整规划摘要显式批准后才可进入实现。
