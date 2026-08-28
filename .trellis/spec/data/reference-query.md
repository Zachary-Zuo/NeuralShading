# Reference online query 合同

## 1. Scope / Trigger

修改 `ReferenceQueryDispatcher`、typed payload binder、Falcor/CUDA共享资源或 evaluator在线补采时适用。

## 2. Signatures

```python
ReferenceQueryDispatcher(definition, snapshots, query_capacity, device, slot_count=2)
evaluate(query, wi, seeds, evaluation_samples=1) -> ReferenceEvaluateResult
sample(query, seeds) -> ReferenceSampleResult
pdf(query, wi, seeds) -> ReferencePdfResult
end_iteration() -> None
close() -> None
```

## 3. Contracts

- family差异只经 `ReferenceProgramDefinition` 和 typed payload注入；query kernel只调用 concrete state的 `prepare/evaluate/sample/pdf`。
- CUDA输入用 `Buffer.from_torch()`，dispatch前 `wait_for_cuda()`；输出shared buffer在`wait_for_falcor()`后映射为同device tensor。
- 至少两个slot；lease未释放不得复用或`end_iteration/close`。
- `evaluate`只平均多次stochastic `f`；PDF/event必须一致，非有限返回invalid，不做亮度clamp或异常值删除。
- material-local shading normal可能使世界半球方向在局部domain无效。online producer压实valid行并补采；这属于proposal rejection，不是reference噪点修复。
- 训练response不做host/NumPy readback，也不写磁盘。CPU readback只存在于显式parity/诊断工具。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| query超过capacity或tensor device/shape错 | dispatch前拒绝 |
| typed usage重复、kind/dtype不支持 | binder拒绝 |
| active lease时结束frame/close | 抛错 |
| evaluator少量invalid | producer补采并记录拒绝统计 |
| 连续达到最大拒绝轮次仍填不满 | 训练明确失败 |

## 5. Good / Base / Bad Cases

- Good：MaterialX normal map改变局部frame，producer保留有效行并用新方向补齐。
- Base：LayerStack deterministic query一次填满；batch仍走相同压实逻辑。
- Bad：把invalid行写成黑色target，或读取`result.f`到CPU后筛选。

## 6. Tests Required

- unit：固定valid mask验证压实顺序、候选/拒绝计数与每轮lease释放。
- GPU：五family同一dispatcher的evaluate/sample/pdf与finite/event检查。
- smoke：LayerStack和MaterialX连续两步online训练，MaterialX覆盖normal-map rejection。

## 7. Wrong vs Correct

```python
# 错
if family_id == "materialx.document@1.39.4":
    return MaterialXLiveProducer(...)

# 对
reference = get_reference_program_for_source(family_id, version)
return ReferenceQueryDispatcher(reference, snapshots, ...)
```
