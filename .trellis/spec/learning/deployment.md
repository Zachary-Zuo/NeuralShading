# 冻结 checkpoint 的当前 runtime 部署

## 1. 适用范围

修改 Slang/compiler 后部署既有权重时适用。训练 resume 和严格 evaluation 要求完全相同 method implementation；部署需要分别追溯原训练与新 runtime，不能为了 shader 修复要求重训，也不能放宽正式研究结论。

## 2. 签名

```text
load_deployment_snapshot(path) -> EvaluationSnapshot
snapshot.require_ready("diagnostic-evaluator")
load_evaluation_snapshot(path) -> 严格原 method identity 检查
ncls.deployment-readiness@1
```

## 3. 合同

- 仅接受 `TrainingCheckpoint@1`，验证 checksum sidecar、内部 plan hash、method/public key、cursor、finite tensors 和 data identity。
- `ResolvedTrainingPlan.from_manifest` 只读已冻结、内部一致的 manifest；`from_dict` 和训练/评测仍额外验证当前 implementation，不能把 deployment reader 接到 resume。
- 用当前 plugin 和冻结 model context 建 template，checkpoint tensor 名称集合、shape、dtype 必须完全一致，再调用 codec.restore。tensor copy 的广播不能代替结构验证。
- 部署要求 phase complete、所有 required component group 的 finite/nonzero gradient/parameter update coverage；不完整 snapshot 可返回带理由的 readiness，但 exporter 必须 require_ready。
- 保留原 checkpoint SHA-256 与 `training_method`，另记 `runtime_implementation_sha256`、`exact_method_identity`、`runtime_validation=gpu-parity-required`。`checkpoint_compatibility=exact` 不等于训练与部署代码 hash 相同。
- 当前 deployment reader 只给 diagnostic-evaluator readiness；正式质量/泛化结论仍走严格 evaluation。四入口 runtime capability 来自实际 shader 实现及 GPU witness，与研究阶段标签独立。
- 新 package/module closure hash 包含 sampler，实现改变必须生成新 program/package identity；不修改旧包或原权重。完整 prepare cost 包含 proposal adapter，不能继续报 evaluator-only 成本。

## 4. 错误矩阵

| 条件 | 结果 |
|---|---|
| checksum/plan/data identity 不一致或未知 schema | reader 拒绝 |
| tensor 缺失/额外、shape/dtype 不同 | restore 前拒绝 |
| 未 complete、缺 required coverage | readiness=false，exporter 拒绝 |
| 只有 runtime implementation 改变，其他均严格匹配 | 可部署，保留两个 identity，要求 GPU parity |
| GPU evaluator/sample/pdf/weight 任一不符 | 不交付为通过验证的包，不改权重或容差隐藏误差 |

## 5. 正常、基础与错误案例

- 正常：同一 step2048 hybrid 权重增加匹配 sampler，新 program identity，原 checkpoint bytes 完全不变。
- 基础：训练和 runtime implementation 完全一致，exact_method_identity=true。
- 错误：修改 checkpoint method hash 冒充当前训练结果；或仅 shape 相似就允许未训练模型部署。

## 6. 必要测试

`test_training_checkpoint_new.py` 检查原 identity 保留、严格 eval drift 拒绝、部署 tensor mismatch；`test_training_plan.py` 保留当前配置的冻结 hash 回归。Metal GPU 检查 quantized prepare/evaluate、sample/PDF/reverse/weight、边界和独立归一化 oracle。

## 7. 错误与正确

```text
错误：shader 变了 -> 覆写 checkpoint 中的训练 hash / 关闭 resume identity 验证
正确：冻结 checkpoint 原样保留 -> 当前 compiler 读独立 deployment snapshot -> 新 package + GPU witness
```
