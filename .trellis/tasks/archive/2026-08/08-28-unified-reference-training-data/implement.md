# 统一材质 Reference 训练数据回路实施计划

## 0. 实施前用户交接（不启动 task）

- [ ] 对 `D:\01_Workspace\NeuralShading\data\reference-responses` 做只读复核，报告 resolved path、顶层项、`.h5/.hdf5` 数量与总字节数。
- [ ] 明确告诉用户该目录是唯一待人工删除目标；不提供代码迁移、不由 agent 执行删除。
- [ ] 用户确认删除完成后，只读验证目录不存在旧 HDF5，再运行 `task.py start unified-reference-training-data`。
- [ ] 若删除目标或数量与 planning 审计显著变化，暂停并重新向用户确认，不扩大删除范围。

Planning 审计基线（2026-08-28，只读）：目标根为 `data/reference-responses/`，共 260 个 `.h5`，456,787,257 bytes；顶层包含 10 个 LayerStack 目录与 2 个 audit HDF5。实施前必须重新统计，不能直接把此观察值当删除命令输入。

## 1. 冻结 focused oracle 与公共合同

- [ ] 读取 `.trellis/spec/project/index.md`、core/data/learning 对应 checklist 与 method constraints；再次记录开发机状态。
- [ ] 更新/新增 focused tests，先冻结 `evaluate().f` response measure、sample tuple、independent PDF、typed binding 与 NVIDIA sampler density的数学 oracle；容差只来自 `references/acceptance.json`、浮点实现分析和现有 shared scattering contract。
- [ ] 给 `ReferenceProgramDescriptor`/registry 增加 source-contract 唯一映射和完整 capability fail-closed test。
- [ ] 记录 rollback point A：测试先表达新合同，但实现尚未迁移；不得运行 formal training。

## 2. Generic ReferenceQueryDispatcher

- [ ] 在 reference/runtime 层实现 typed payload binder与 generic Falcor query dispatcher；迁移 `data.falcor` 中仍通用的 device/shared-buffer helper。
- [ ] 新增 family-agnostic Slang query kernel，完整构造 `NclsScatteringContext` 并通过 `NclsPackage*` specialization 调 `prepare/evaluate/sample/pdf`。
- [ ] 标准化 LayerStack、MaterialX、OpenPBR、MERL、MDL runtime/material descriptors；补齐 texture/sampler/source module 的真实 binding usage。
- [ ] 把 MDL canonical runtime 从专用 `mdl_query.slang` 迁到 `mdl.slang + NclsMdlGenerated` typed module；保持锁定 SDK target code、argument block、RO/texture语义。
- [ ] 实现 CUDA↔Falcor 显式同步、shared output tensor、双 slot lease、state_dict/resume identity 与严格 close lifecycle。
- [ ] 实现 generic stochastic-evaluate averaging；只平均 `f`，不 clamp，不忽略 invalid/non-finite。
- [ ] 运行 registry/unit、五 family GPU evaluate/sample/pdf、MDL native parity、LayerStack Monte Carlo 与无 host response readback tests。
- [ ] 记录 rollback point B：dispatcher已能独立替代 provider query，但训练仍未切换。

## 3. Source loader 与 method source adaptation

- [ ] 给 source family registry 增加 generic locator→snapshot 入口；把 LayerStack path、MaterialX catalog、MDL artifact、OpenPBR 与 MERL asset resolution 放回各自 source family 所有权。
- [ ] 新增 method source-adaptation interface/registry；迁移 `NativeFeaturePyramid`、LayerStack native feature encoding 与 MaterialX spatial feature pyramid。
- [ ] 保留 NVIDIA 当前 LayerStack/MaterialX adaptation，未支持 family明确 fail closed；adapter不得导入 dispatcher之外的 reference math。
- [ ] 用 unit tests证明 CLI/producer 不含 family 分支，feature adapter 不提供 `evaluate/sample/pdf`。
- [ ] 记录 rollback point C：source acquisition、reference execution 与 method conditioning 已解耦。

## 4. Typed online batches 与 TrainingRunner

- [ ] 用 `EvaluatorBatch@2`、`MethodSamplerBatch@2` 和 common conditioning 取代 `TrainingBatch@1`；移除全局 required tensor list。
- [ ] 实现唯一 `OnlineTrainingProducer`：evaluator route 调 dispatcher `evaluate().f`，sampler route只生成独立 conditioning + `sample_u`。
- [ ] `TrainingRoute@3` 使用显式 kind；删除 `query_role/target_estimator`，并把 direction/filtering/mollification/evaluation_samples 配置归入 versioned query recipe。
- [ ] TrainingRunner 只调 typed producer并管理 lease；materialization通过 method source adapter取得 feature pyramid。
- [ ] checkpoint/resume 保存各 route RNG、reference program、source snapshots、query recipe与 adapter implementation identity。
- [ ] 运行 batch validation、runner lifecycle、route independence、same-device/no-readback 与 resume determinism tests。
- [ ] 记录 rollback point D：正式训练入口已不依赖旧 `ncls.data` producer。

## 5. NVIDIA evaluator/sampler 语义纠正

- [ ] Python model把 decoder 输出命名并定义为 `evaluate_f`；删除 `response_cos` 与 `forward()/cos` adapter。
- [ ] Slang FP32/FP16 core与 package backend直接返回 `f`；删除 `nclsNvidiaNeuralResponseToBareF` 路径，更新符号、layout identity与 package parity字段。
- [ ] evaluator log-L1直接比较 `f_hat` 与 `target_f`。
- [ ] sampler forward-KL 用 detached `f_hat` 显式乘 shading-normal absolute cosine；保留 learned sample/PDF、latent detach与 score-function梯度路径。
- [ ] bump MethodDescriptor、TrainingConfig、TrainingCheckpoint 与 recipe/config identity；旧 checkpoint/config直接拒绝，不写 converter。
- [ ] 运行 Python↔Slang evaluator/sampler parity、analytic loss、gradient ownership、FP16 package query与2-step LayerStack/MaterialX smoke。
- [ ] 记录 rollback point E：新 checkpoint首次产生后，禁止回退到旧 `f·cos` 语义继续训练。

## 6. CLI、工具与旧路径删除

- [ ] `learn train/evaluate/export` 改用 generic source loader + online producer；删除 `_batch_source` 和 family-specific export分支。
- [ ] 删除 `data validate/plan/collect/validate-corpus/audit-dense` 命令、offline/corpus configs、HDF5 schemas与相关 tests。
- [ ] 删除/迁空旧 `src/ncls/data`：collector/corpus/dataset/store/provider/live source/statistics/selection/profile/prior；通用 helper必须先迁到明确的新 owner。
- [ ] 删除 `shaders/ncls/data/reference_*.cs.slang` 与 `mdl_query.slang`。
- [ ] pbrt/MDL parity和 viewer asset preparation tools迁到 source loader + dispatcher；仍需的 correctness tests重写为 canonical query，不简单删掉覆盖。
- [ ] 移除 `h5py` project dependency、`REFERENCE_RESPONSE_ROOT` 与全部 offline/HDF5 imports；用 `rg` 静态门确认没有 compatibility residue。
- [ ] 记录 rollback point F：legacy code删除完成；磁盘数据清理由用户在 Phase 0 完成，不在此阶段执行删除。

## 7. 稳定文档与 Trellis spec

- [ ] 更新 `AGENTS.md` 根目标与 repository policy 中的 `data/`说明：正式训练完全 online，根仓库没有 reference-response HDF5 产品。
- [ ] 更新 `docs/architecture.md`、`docs/realtime_material_compilation.md`、`docs/data.md`、研究框架/候选/日志中当前架构描述；历史实验行保留原 identity并明确其为被替代的 archived evidence，不把旧 HDF5重持久化。
- [ ] 用 `trellis-update-spec` 重写 data/learning contracts：generic query dispatcher、typed batches、source sample/pdf职责、`f` measure与 fail-closed规则。
- [ ] 检查所有中文文档术语统一为 reference、direct fit、backend-specific `ScatteringState`；不再把 LayerStackIR写成所有 source的 GT。

## 8. 质量门与收尾

- [ ] unit：source/reference registry、typed resources、typed batches、config/checkpoint、runner、NVIDIA objective与CLI静态边界。
- [ ] GPU：五 family generic evaluate/sample/pdf、LayerStack stochastic gate、MDL native crosscheck、CUDA/Falcor lease、Python/Slang与package query parity。
- [ ] integration：LayerStack和MaterialX 2-step online smoke、checkpoint/resume/export；验证 sampler batch无 target且 evaluator target是 `f`。
- [ ] 运行与本次改动相关的完整 pytest集合；检查 `external/Falcor`、其它锁定 upstream clean，根 worktree只包含本任务与用户原有改动。
- [ ] 将单次验证输出写入 `artifacts/`，不写回 PRD；不运行300k formal，不新增 observed quality hard gate。
- [ ] 执行 Trellis check，更新 task related files/summary，完成后进入 finish-work；不自动创建 recorded-batch 后续任务。

### 8.1 计划中的验证命令

具体测试文件可随实现重命名，但每组语义覆盖不得减少。所有 Python 命令使用项目唯一 Conda 环境；导入 Falcor 的测试通过项目 wrapper：

```powershell
conda run -n neural-shading python -m pytest tests\unit -q

.\scripts\run_falcor_python.ps1 -m pytest `
  tests\gpu\test_reference_query_dispatcher.py `
  tests\gpu\test_reference_backend_contracts.py `
  tests\gpu\test_mdl_native_crosscheck.py `
  tests\gpu\test_scattering_package_parity.py -q

.\scripts\run_falcor_python.ps1 -m pytest `
  tests\integration\reference `
  tests\integration\learning -q

conda run -n neural-shading python -m ncls learn train `
  configs\learning\nvidia-rta2024-layer-stack-smoke.json `
  artifacts\runs\unified-reference-training-data\layer-stack-smoke.pt

conda run -n neural-shading python -m ncls learn train `
  configs\learning\nvidia-rta2024-materialx-smoke.json `
  artifacts\runs\unified-reference-training-data\materialx-smoke.pt
```

静态 residue 与 upstream 边界检查：

```powershell
rg -n "OfflineBatchSource|LiveReferenceBatchSource|MaterialXLiveReferenceBatchSource|MdlLiveReferenceBatchSource|h5py|HDF5|reference-responses|target_estimator|response_cos" `
  src shaders configs tests tools docs .trellis\spec AGENTS.md pyproject.toml environment.yml

git -C external\Falcor status --short
git status --short
```

`rg` 可命中明确标注的历史实验行；任何当前代码、schema、配置、用户入口或稳定架构说明中的命中都必须逐项处理。长于 smoke 的运行不属于本任务质量门。

## 9. 失败分类

- dispatcher与 canonical backend/独立 oracle不一致：implementation defect，停在相应 rollback point修复。
- typed resource无法表达某 source runtime：protocol/design defect，回到 planning；不得加 family-specific provider fallback。
- 65k focused preflight显存/吞吐不可行：resource/throughput defect，报告证据并回到 planning；不得缩小 formal recipe仍沿用 formal identity。
- 忠实在线训练的 quality较低：正常 empirical outcome，不阻塞本架构任务，也不触发自动扩样本/训练预算。
