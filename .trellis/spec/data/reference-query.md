# ReferenceExecutionPlan 与 online query 合同

## 1. Scope / Trigger

修改reference definition、plan、backend/session、typed payload binder、Falcor/CUDA共享资源或online补采时适用。

## 2. Signatures

```python
compile_reference_execution_plan(entries, query_recipe=...) -> ReferenceExecutionPlan@1
create_reference_backend(...) -> ReferenceBackendCapability
backend.open(plan, query_capacity, device, slot_count=2) -> ReferenceBackendSession
evaluate(query, wi, seeds, evaluation_samples=1) -> ReferenceEvaluateResult
sample(query, seeds) -> ReferenceSampleResult
pdf(query, wi, seeds) -> ReferencePdfResult
```

## 3. Contracts

- plan具有有序且唯一的execution groups、稠密global source index、group-local material index、argument/RO offset和query recipe identity；work item必须group-homogeneous。
- 多个execution group属于一个公共session：Falcor frame只由session开始/结束一次，各group只提交自己的dispatch。close先原子预检全部group lease；任一group仍有lease时不得结束部分group或部分frame。构造中途失败必须逆序释放已创建group。
- family差异只经`ReferenceProgramDefinition`和typed payload注入；backend只路由group/global index，不判断family。
- capability独占platform、Falcor import/device/build layout。upper code不得直接构造session/device或保留旧open签名。
- CUDA输入经`Buffer.from_torch()`；dispatch前`wait_for_cuda()`，输出在`wait_for_falcor()`后映射为同device tensor。至少两个slot，lease未释放不得复用、`end_iteration`或close。
- `evaluate`平均stochastic `f`，PDF/event必须一致；非有限返回invalid，不做clamp。公共`f`以输入frame的transport measure定义。
- producer在GPU压实material-local invalid行并补采；不写零target、不host readback筛选。
- typed texture shape为spatial-first，binder在创建资源前验证rank、extent、dtype和payload大小。
- 每个group的material blob table/layout在session生命周期内不可漂移。typed resource与sampler descriptor必须按usage一一覆盖program payload；sampler usage不得重复，filter/address语义必须按descriptor创建，不能由backend补默认值。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| plan group/index不稠密、snapshot重复或query recipe空 | plan构造拒绝 |
| unknown group或source不属于group | dispatch拒绝 |
| doctor存在missing/invalid provider | open在导入Falcor前拒绝 |
| active lease时frame结束/close | 抛错 |
| 一个group仍有active lease而session关闭 | 所有group保持打开，frame不结束；释放lease后才能整体close |
| material blob table/layout在dispatch间变化 | dispatch拒绝，不用新布局覆盖旧buffer |
| resource缺typed descriptor、sampler usage重复或filter/address未知 | session构造拒绝 |
| evaluator连续达到拒绝上限 | training明确失败 |

## 5. Good / Base / Bad Cases

- Good：一个plan同时包含LayerStack与两个共享MDL target-code graph的argument state；session只开始一次frame，MDL state共享代码与纹理但使用各自argument/RO offset。
- Base：单program、单source plan仍走相同group表、lease和frame ownership，不存在特例session。
- Bad：每个group各自调用`beginFrame/endFrame`；close先关闭无lease group，最后才发现另一group有lease；或MDL sampler缺descriptor时偷偷创建linear-wrap默认值。

## 6. Tests Required

- unit：plan global/local mapping、重复snapshot拒绝、invalid压实、typed texture extent；
- GPU：五family经同一plan/session执行evaluate/sample/pdf；MDL native response交叉验证；
- smoke：LayerStack、MaterialX和固定MDL的online phase training。

## 7. Wrong vs Correct

```python
# 错
ReferenceBackendSession(definition=definition, snapshots=snapshots, device=device)

# 对
plan = compile_single_program_plan(definition, snapshots, query_recipe=recipe)
backend.open(plan, query_capacity=capacity, device="cuda:0")
```

```python
# 错：group拥有Falcor frame，关闭失败会留下半关闭session。
for group in groups:
    group.end_frame()

# 对：session先验证全部lease，再只结束一次共享frame。
session.assert_no_active_leases()
session.end_shared_frame()
```
