# 执行计划

## 顺序

1. 新增layout/profile生成器和`MethodDescriptor@2` required-component合同；
2. 实现Metal adapter消费已冻结`NativeAssetCollection`，验证slot/role/mip/split identity；
3. 实现shared texture codec、semantic heads、encoder-only materialization与QAT grid storage；
4. 实现typed compiler、optimized-state teacher和compiler functional distillation；
5. 实现deterministic two-mip prepare、learned frames、angular bank、analytic/positive hybrid evaluator；
6. 将loss接入canonical `codec-warmup/joint-appearance` phases，不实现hidden sub-lifecycle；
7. 实现component execution/gradient/update/export-state conformance；
8. 运行fixture、小cohortfull-shape smoke和full-cohort activation preflight；
9. 产出evaluator-slice v4 checkpoint、Python oracle、layout/packing/proposal spec和report-only初始指标。

## 重点文件

- `src/ncls/learning/methods/metal_fused.py`
- `src/ncls/learning/models/metal_fused*.py`
- canonical asset collection adapter modules
- `src/ncls/learning/abi/metal_fused_layout_v1.json`
- `configs/learning/metal-fused-*.json`
- 新增unit/GPU model tests；不修改NVIDIA model数学identity。

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_method_definition.py tests/unit/test_training_batch.py tests/unit/test_training_runner_lifecycle.py tests/unit/test_training_checkpoint.py
conda run -n neural-shading python -m pytest tests/unit -k "metal_fused or source_adapter"
scripts/run_falcor_python.ps1 -Command "python -m pytest tests/gpu -k metal_fused"
scripts/run_falcor_python.ps1 -Command "python -m ncls.cli learn train <frozen-smoke-config> <artifact-checkpoint>"
scripts/run_falcor_python.ps1 -Command "python -m ncls.cli learn evaluate <frozen-smoke-config> <artifact-checkpoint> --batches 2"
git diff --check
```

正式long run使用`tqdm`真实work units；预计吞吐异常时暂停profile，不自动缩batch/模型或扩大预算。

## 回滚点

- layout/adapter；
- codec；
- compiler；
- hybrid evaluator；
- phase/conformance。

实现缺陷回滚到最近模块提交；observed quality低直接登记，不触发自动v2循环。profile若无法静态部署，返回parent更换identity后再调整。
