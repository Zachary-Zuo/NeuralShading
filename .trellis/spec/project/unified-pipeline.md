# 统一 pipeline 可执行合同

## 1. Scope / Trigger

修改source family、reference execution、online producer、method、checkpoint、package或viewer时适用。新增方法/source不得修改其他层通用实现，不得增加磁盘batch、family-specific producer或CLI分支。

## 2. Signatures

```text
SourceFamilyDefinition.load_snapshot(locator) -> SourceSnapshot
ReferenceProgramDefinition.compile_runtime/material(...) -> typed reference payload
compile_reference_execution_plan(...) -> ReferenceExecutionPlan@1
ReferenceBackendCapability.open(plan, ...) -> ReferenceBackendSession
OnlineTrainingProducer.next_batch(...) -> AssetTileBatch@1 | EvaluatorBatch@3 | MethodSamplerBatch@3
MethodDefinition.compile_program/asset/instance(...) -> deployment payloads
TrainingRunner.run(...) -> TrainingCheckpoint@4
write_scattering_package(...) -> ScatteringPackageManifest@2
ScatteringPackage.open(path).create_binding() -> ScatteringBinding(program, asset, instance)
InstancePayload(parameters, blobs, blob_descriptors, editor, compiler)
Viewer.programGpuRuntime(program_id) -> shared passes/weights/compiler
Viewer.applyMaterialEditor(slot, method, candidate_parameter_view) -> atomic slot update
```

## 3. Contracts

- `SourceSnapshot`是source唯一真相；每个source contract恰有一个完整`prepare/evaluate/sample/pdf` reference。
- `ReferenceExecutionPlan@1`拥有grouping与global/local index映射；backend只执行plan，不识别family。platform/Falcor只由capability拥有。
- `NativeAssetCollection@1`拥有multi-asset/mip/tile+halo输入；training batch只在GPU在线产生，不存在HDF5、corpus或recorded batch。
- `TrainingConfig@4`是任意phase graph；`MethodDescriptor@2`严格拥有parameter/component contracts；generic conformance覆盖execution、gradient/update和artifacts。
- live target保持同CUDA device；invalid reference行GPU压实补采；lease与prefetch不越过资源生命周期和checkpoint边界。
- `TrainingCheckpoint@4`保存method/components、phase-local optimization、plan/asset/query/source identity与coverage，严格恢复。
- `ScatteringPackage@2`独立计算program、asset、instance与package identity；viewer先验证全部section与typed resources，再原子绑定slot。
- editable instance的`editor`与`compiler`必须同时存在或同时为空。editor递归验证native path/type/value/operation、normalization default、raw word地址与derived writes；raw和compiled usages必须指向instance section中的mutable structured buffers。host不能只验证当前value而让损坏的default或unsupported operation延迟到UI阶段失败。
- viewer以`programId`缓存shader、shared weights和material compiler；asset/instance是slot-local binding。typed edit与package替换都先构造candidate buffers、上传raw state并编译compiled state，全部成功后才替换active slot。MDL package compatibility比较包含export/default/transitive resources的`sourceSnapshotId`，其他family比较各自native asset identity。
- quality-first neural shader允许把大矩阵从完全展开改为有静态`MaxIters`的runtime loop，并由通用package renderer分块调度以适应平台watchdog；这种调度不得减层、减宽、跳权重或按source family新增renderer分支。具体tile尺寸是平台诊断值，不是方法身份或质量门。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| source/reference/plan identity不一致 | 构造拒绝 |
| unknown platform/provider或typed payload错误 | capability/session拒绝 |
| config phase/component/parameter ownership不完整 | config/conformance拒绝 |
| loss、gradient、update或checkpoint identity失败 | training/resume拒绝 |
| package URI/hash/section identity/instance binding错误 | loader/viewer拒绝 |
| editable view的default类型、operation、word offset或mutable usage非法 | Python writer/loader与C++ viewer均在创建GPU binding前拒绝 |
| typed edit compiler或candidate resource创建失败 | 丢弃candidate，保留旧slot buffers、editor value与accumulation identity |
| MDL package用root `.mdl` hash替代canonical snapshot比较 | package不兼容；修正为`sourceSnapshotId`，不能放宽为任意同module source |
| 新asset验证失败 | viewer保留旧binding |

## 5. Good / Base / Bad Cases

- Good：多个native source状态先编入grouped plan，再由同一producer产生typed online batches；method compiler独立输出program/asset/instance，viewer只解释package schema与usage。
- Good：两个package共享`programId`但拥有不同asset/instance；viewer复用一次program runtime，每个slot独立绑定asset grids与compiled material。
- Base：单source、无纹理、单phase方法仍使用同一plan/collection/phase/package路径，不新增简化入口。
- Bad：runner按source family选择producer；exporter识别method名称补artifact；viewer用旧package/preset reader掩盖不完整迁移。
- Bad：UI先覆盖active raw buffer再运行compiler；compile失败会留下raw/compiled不一致状态，且无法原子回滚。

## 6. Tests Required

- unit：registry、plan、asset collection、typed batches、phase resume、component conformance、checkpoint v4、package v2 tamper；
- unit：editable instance roundtrip；unsupported operation、错误normalization default、越界word与非mutable usage逐项fail closed；viewer静态断言program cache只以`programId`为键且compiler先于slot move；
- GPU：五source同一backend plan/session，NVIDIA training core与package parity；
- integration：LayerStack、MaterialX、固定MDL真实online两phase训练并reload/evaluate/export；
- viewer：Release build、program cache/asset/instance双slot和Falcor clean；
- static：upper modules无platform path、旧schema reader、固定lifecycle或source-specific producer。

## 7. Wrong vs Correct

```python
# 错：runner硬编码bootstrap/materialization/finetune
if step == materialization_step: ...

# 对：phase定义transition，method执行它
definition.apply_phase_transition(model, phase.transition, native_assets)
```

```cpp
// 错：先修改active binding，compiler失败后raw/compiled已经分裂。
upload(active.buffers);
compile(active.buffers);

// 对：在candidate中完成上传与编译，成功后一次性替换。
auto candidate = active.buffers;
upload(candidate);
compile(candidate);
active.buffers = std::move(candidate);
```
