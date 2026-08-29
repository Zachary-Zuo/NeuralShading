# 统一 pipeline 可执行合同

## 1. Scope / Trigger

修改 source family、reference execution、online producer、method、checkpoint、package 或 viewer 时适用。新增方法或 source family 不得修改其他层的通用实现，也不得增加磁盘 batch、family-specific producer 或 CLI 分支。

## 2. Signatures

```text
SourceFamilyDefinition.load_snapshot(locator) -> SourceSnapshot
SourceFamilyDefinition.describe_parameters(snapshot) -> SourceParameterView@1
SourceFamilyDefinition.apply_edit(snapshot, SourceEditPatch@1) -> SourceEditResult
ReferenceProgramDefinition.compile_runtime() -> RuntimePayload
ReferenceProgramDefinition.compile_material(snapshot) -> MaterialPayload
ReferenceProgramDefinition.preflight_provider(platform_id, project_root) -> tuple[ReferenceProgramProviderStatus, ...]
create_reference_backend(...) -> ReferenceBackendCapability
ReferenceBackendCapability.doctor(programs=None) -> ReferenceBackendReport
ReferenceBackendCapability.open(definition, snapshots, query_capacity, device, slot_count=2) -> ReferenceBackendSession
ReferenceBackendSession.evaluate(query, wi, seeds, evaluation_samples=1) -> ReferenceEvaluateResult
ReferenceBackendSession.sample(query, seeds) -> ReferenceSampleResult
ReferenceBackendSession.pdf(query, wi, seeds) -> ReferencePdfResult
OnlineTrainingProducer.next_batch(TrainingRouteRequest) -> EvaluatorBatch@2 | MethodSamplerBatch@2
MethodDefinition.compile_runtime(checkpoint) -> RuntimePayload
MethodDefinition.compile_material(snapshot, checkpoint) -> MaterialPayload
write_scattering_package(...) -> ScatteringPackageManifest@1
ScatteringPackage.open(path).create_binding() -> ScatteringBinding
```

CLI：

```text
ncls learn train <config-v3.json> <checkpoint.pt> [--resume <checkpoint.pt>]
ncls learn evaluate <config-v3.json> <checkpoint.pt> [--batches N]
ncls learn export <checkpoint.pt> <package-dir> [--material-index N]
ncls reference doctor [--json]
ncls reference probe
```

## 3. Contracts

- `SourceSnapshot` 是 source 唯一真相。每个正式 source contract 在 registry 中恰有一个 canonical reference，且完整声明 `PREPARE|EVALUATE|SAMPLE|PDF`。
- `references/reference-backend-toolchains.json` 是 Windows/Linux build 与 program provider requirement 的唯一 manifest；`asset_policy` 必须是 `external-only-no-source-assets`。source locator 及其资源仍由用户复制或既有资产流程管理。
- `ReferenceBackendDescriptor` 同时记录跨平台语义 identity 和平台 build identity；session 的 `reference_program_identity` 必须包含完整 backend identity，checkpoint 因而拒绝在未知 build 上静默续训。
- `RuntimePayload` / `MaterialPayload` 通过 `kind/dtype/shape/stride/alignment/format/color_space/usage` typed descriptor 绑定；capability/session 不判断 family 名称。
- `evaluate()` 返回线性 RGB `f`，不含几何余弦。renderer response adapter 与 sampler density 需要 `f·|cosθi|` 时在消费点显式计算。
- evaluator route 调 source `prepare/evaluate`，生成 `EvaluatorBatch(conditioning, wi, target_f)`；method-sampler route只生成 `MethodSamplerBatch(conditioning, sample_u)`。source `sample/pdf` 用于 PT 与正确性验证，不是 NVIDIA sampler teacher。
- live target 保持在同一 CUDA device；CUDA→Falcor、Falcor→CUDA 顺序显式同步，consumer持有 lease 时输出 slot 不得复用。
- reference 的 `valid=false` 是可表达的局部 domain rejection。producer 在 GPU 上压实有效行并继续补采，记录 `candidate_count/rejected_count/rejection_rounds`；不得 clamp、把无效行当零 GT，或因任一行无效而杀死整个 batch。每轮 result lease 在复制已选行后必须释放，达到 `maximum_rejection_rounds` 则失败。
- `TrainingConfig@3` 只含 source locator、online query recipe 与 typed routes；`TrainingCheckpoint@3` 保存 source snapshots、reference/query identities、各 route RNG 与 lifecycle。旧 config/checkpoint 无 reader或 converter。
- 正式训练不读写 HDF5、shard、corpus 或 recorded batch。磁盘只保存 source 资产、checkpoint、package 和 `artifacts/` 中的验证结果。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| source/reference family、version、program identity 不一致 | 构造期拒绝 |
| reference 缺 `prepare/evaluate/sample/pdf` | descriptor/registry 构造失败 |
| unknown OS/arch、manifest program/provider 不完整 | capability 构造或 doctor 失败关闭 |
| Falcor build/module/runtime、program provider 缺失 | doctor 返回 generic `missing/invalid`；`open()`拒绝 |
| typed binding 缺字段、usage 重复或 payload 尺寸错误 | session/package writer拒绝 |
| reference row invalid | 在线拒绝采样并补足 batch；超过轮次上限时报错 |
| batch tag、shape、device、finite/nonnegative 错 | typed batch构造失败 |
| config出现 offline/recorded/HDF5或旧 route字段 | `TrainingConfig@3` 拒绝 |
| checkpoint hash/schema/tensor/query identity不符 | 拒绝恢复 |
| package URI越界、文件/hash/ABI错 | 拒绝 binding |
| 新 material 编译或验证失败 | viewer保留旧 binding并显示 error |

## 5. Good / Base / Bad Cases

- Good：新增 source 只实现 family loader、canonical reference program和 typed payload，并把 build requirement登记进根 manifest；统一 session 测试其四个 operation。
- Base：同一个 material-local normal 使少量通用方向 invalid；producer拒绝这些行并用同一确定性 RNG stream补足。
- Bad：upper module 直接 import Falcor、分支 `DeviceType.D3D12/Vulkan`，为 LayerStack 增加专用 collector，或把 `f·cos` 持久化后再除 cosine训练 evaluator。

## 6. Tests Required

- unit：registry唯一性、typed batch字段、invalid-row压实与lease释放、config/checkpoint v3、resume确定性、package tamper。
- GPU：五个正式 source 通过同一 backend/session执行 `evaluate/sample/pdf`；MDL native crosscheck；NVIDIA Python/Slang evaluator与sampler梯度；package shader parity。
- integration/smoke：LayerStack、带纹理MaterialX与固定无空间纹理MDL各完成 bootstrap→materialization→finetune，并产出可恢复 `TrainingCheckpoint@3`。
- 静态：upper modules中无 `sys.platform`、`DeviceType.D3D12/Vulkan`、平台 build 路径、source-specific producer、旧 dispatcher 或旧采集 shader。

## 7. Wrong vs Correct

```python
# 错：把旧矩形 batch 强加给 sampler route
TrainingBatch(target=torch.zeros(...), wi=torch.zeros(...), sample_u=u)

# 对：route只携带有语义的字段
MethodSamplerBatch(conditioning, sample_u=u)
```

```python
# 错：局部法线造成一个 invalid row就终止整批
torch._assert_async(result.valid.all())

# 对：复制valid行、释放lease、继续从同一route RNG stream补采
selected = torch.nonzero(result.valid.all(dim=1)).flatten()
```

```python
# 错：上层选择平台并直接创建 Falcor device
device = falcor.Device(type=falcor.DeviceType.D3D12)
session = ReferenceBackendSession(...)

# 对：平台和底层实例只由 capability 解析
backend = create_reference_backend()
session = backend.open(definition, snapshots, query_capacity=capacity, device="cuda:0")
```
