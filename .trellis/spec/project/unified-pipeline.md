# 统一 pipeline 可执行合同

## 1. Scope / Trigger

修改 source family、reference execution、batch producer、method、checkpoint、package 或 viewer 时适用。新增方法或 source family 不得修改其他层的通用实现。

## 2. Signatures

```text
SourceFamilyDefinition.describe_parameters(snapshot) -> SourceParameterView@1
SourceFamilyDefinition.apply_edit(snapshot, SourceEditPatch@1) -> SourceEditResult
ReferenceBackend.prepare(context, compiledMaterial) -> ReferenceState
ReferenceState.evaluate(wiWorld, sampleGenerator) -> NclsScatteringEval
ReferenceState.sample(sampleGenerator) -> NclsScatteringSample
ReferenceState.pdf(wiWorld) -> NclsScatteringPdf
BatchSource.next_batch(size) -> TrainingBatch@1
MethodDefinition.compile_runtime(checkpoint) -> RuntimePayload
MethodDefinition.compile_material(snapshot, checkpoint) -> MaterialPayload
write_scattering_package(...) -> ScatteringPackageManifest@1
ScatteringPackage.open(path).create_binding() -> ScatteringBinding
```

## 3. Contracts

`SourceSnapshot` 是 source 唯一真相；每个 runtime reference 必须完整提供自己的 `prepare/evaluate/sample/pdf`，descriptor 缺任一 path-tracing capability 时 fail closed。统一接口不改变 source 原生参数、图结构、资源、closure 或 proposal；renderer 不识别 source family。batch 全 tensor 同 device；live target 禁止 host readback；checkpoint tensor schema 来自 method descriptor；package 三个 identity 独立；viewer 恰有两个对称 slot，panel 宽度为 `floor(W/2)`。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| stale edit base snapshot | 拒绝 patch |
| runtime reference 缺 prepare/evaluate/sample/pdf | descriptor 构造失败 |
| sample/pdf 与 evaluate 不匹配 | GPU 数值正确性失败，不得 fallback 或 clamp |
| batch 缺 tensor、跨 device 或 shape 错 | 构造失败 |
| checkpoint hash/schema/tensor 不符 | 拒绝恢复 |
| package URI 越界、文件/hash/ABI 错 | 拒绝 binding |
| slot mode 缺 capability | 该 slot 为 unsupported，另一侧不变 |
| 新 material 编译或验证失败 | 保留旧 binding，显示 error |

## 5. Good / Base / Bad Cases

- Good：test fixture 使用不同 tensor layout 和 source adapter，不改 runner/writer/viewer。
- Base：同一 runtime 换 material，只改变 material/package identity。
- Bad：增加 method-specific CLI/exporter/session，或用 CPU copy 伪装 live batch。

## 6. Tests Required

unit 覆盖 scattering descriptor、response measure、batch、method fixture、checkpoint、package tamper、source editor、slot extent/studio；Falcor 覆盖各 source 的 sample→pdf/weight 恒等式、CUDA live batch 与 package absolute-path compile；Release viewer build 后确认 Falcor clean。

## 7. Wrong vs Correct

```python
# 错
if isinstance(model, ConcreteModel): ...
# 对
loss, metrics = method.training_objective(model, batch, phase)
```
