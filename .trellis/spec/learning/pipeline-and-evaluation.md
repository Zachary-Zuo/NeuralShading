# 方法、训练、评测与部署编译合同

## 1. Scope / Trigger

新增或修改`MethodDefinition`、method descriptor、训练objective、artifact compiler、`ScatteringPackage`、评测或viewer部署时适用。公共runner、bundle writer和viewer不得通过concrete method名称补齐缺失语义。

## 2. Signatures

```text
MethodDefinition.descriptor -> MethodDescriptor@2
method.training_objective(model, batches, phase) -> (loss, component_outputs, metrics)
method.compile_program(checkpoint) -> ProgramPayload
method.compile_asset(checkpoint, source_snapshot) -> AssetPayload
method.compile_instance(program, asset, material_index) -> InstancePayload
write_scattering_package(program, asset, instance) -> ScatteringPackage@2
```

## 3. Contracts

- descriptor完整登记parameter groups、required components、active phases、typed batch dependencies、Python outputs、runtime artifacts和Slang entry points；generic conformance按登记内容检查，不能识别NVIDIA或其他具体方法。
- artifact compiler输出program/asset/instance三段独立identity。program持有module closure、defines、runtime blobs/samplers与能力；asset持有source identity、compiled material、纹理与asset samplers；instance只选择兼容program/asset及material index。
- typed blob必须声明`usage/dtype/shape/stride/path/sha256`，viewer按usage绑定，不拼接假定的weights/material buffer。当前packed FP16权重使用`packed-float16x2-uint32@1`且stride为4；未知dtype fail closed。
- sampler不是伪纹理resource：program/asset分别显式列出sampler descriptor，usage全package唯一并进入对应identity。module source只属于module closure，不能伪装成runtime typed blob。
- writer、Python loader、C++ viewer对同一schema做严格字段、hash、shape、index与artifact inventory验证；source reference不包装成package。
- NVIDIA formal recipe保留独立evaluator/sampler online batch、encoder→hierarchical latent materialization→finetune、matched sampler与packed-FP16 runtime；smoke只缩小step/batch/tile，不删除component或改写方法。
- Metal `metal-fused-neural-material@1`在evaluator研究切片中只声明`prepare/evaluate`能力，但descriptor同时冻结full profile layout、328个tensor state、18个required evaluator components及11-component proposal reservation。matched `sample/pdf`与Package@2/Slang artifacts完成前，compiler入口必须fail closed，不能把evaluator checkpoint冒充完整部署包。
- 评测复用checkpoint冻结的source、reference plan、asset collection、query stream和phase recipe。质量比较使用matched数据与bootstrap CI；成本/质量先report-only，除非已有明确需求或数学正确性门槛。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| descriptor漏required component/output/artifact/entry | conformance或写包前拒绝 |
| program/asset source、ABI或sampler usage不兼容 | instance/package构造拒绝 |
| typed blob未知dtype、shape非正、size与stride不整除 | Python/C++ loader拒绝 |
| resource或sampler usage重复、descriptor与文件hash不符 | package加载拒绝 |
| compiled material index越界 | Python/C++ loader拒绝 |
| module source出现在typed blobs | writer拒绝，要求放回module closure |
| Metal evaluator-slice缺sampler/runtime artifacts却请求package编译 | method compiler拒绝，并由后续sampler/runtime任务补齐同一identity的能力合同 |

## 5. Good / Base / Bad Cases

- Good：一个NVIDIA program被多个metal asset复用；每个asset拥有自己的hierarchical latent texture和显式sampler，instance只选择compiled material index。
- Good：Metal evaluator checkpoint能严格resume/evaluate，asset cook能生成独立flat INT8 high/low grids与FP16 adapter，但在matched sampler和Slang entry尚未完成时不产生Package@2。
- Base：无纹理方法仍产生空resource/sampler集合，但使用同一program/asset/instance schema和严格loader。
- Bad：writer把缺失sampler静默补成linear-wrap；viewer硬编码`gNclsRuntimeWeights`并把所有blob拼接；或为了smoke跳过sampler/decoder训练。

## 6. Tests Required

- unit：descriptor正负conformance、三段identity、typed blob/sampler/inventory严格校验、v1拒绝与material index bounds。
- GPU：Python与Slang evaluator/sampler梯度、package parity、显式texture/sampler绑定。
- integration：online phase训练→checkpoint→program/asset/instance→package加载闭环。
- integration：Metal evaluator slice先验证checkpoint identity与Python oracle；sampler/runtime任务完成后再把同一冻结layout接入三段artifact闭环。
- Release viewer：从自身绝对module closure编译，按usage绑定全部blob/resource/sampler并通过parity probe。

## 7. Wrong vs Correct

```python
# 错：writer知道某方法缺什么并静默造默认artifact。
if method.name == "nvidia" and not samplers:
    samplers = [linear_wrap_sampler]

# 对：方法编译器完整产生descriptor；generic writer只验证并序列化。
artifacts = method.compile_asset(checkpoint, source_snapshot)
validate_artifact_conformance(method.descriptor, artifacts)
write_scattering_package(program, artifacts, instance)
```

```cpp
// 错：viewer假定两个固定buffer。
root["gNclsRuntimeWeights"] = concatenate(programBlobs);

// 对：schema usage就是host ABI binding name。
for (const auto& blob : program.blobs) root[blob.usage] = createBuffer(blob);
for (const auto& blob : asset.blobs) root[blob.usage] = createBuffer(blob);
```
