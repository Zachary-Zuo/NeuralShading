# 执行计划

## 顺序

1. 冻结 proposal component enum、state reservation、random tuple 与 hemisphere convention；
2. 实现 Python component sample/PDF oracle、mixture normalization 和 property tests；
3. 在冻结 evaluator 上实现 proposal dataset/query 与训练 objective；
4. 在canonical`proposal-fit`phase训练full proposal head并建立analytic-only、reference proposal controls；
5. 实现 Slang sample/PDF，执行随机 state/query parity 与统计归一化测试；
6. 实现proposal component execution/gradient/update与Python/Slang artifact coverage；
7. 冻结full method capability/layout并交给runtime child；
8. 记录短run density/support/weight-tail/cost诊断，不执行formal scene variance；
9. 不在本child删除full components或自动排队compact task。

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "sampler or sample_pdf or proposal"
scripts/run_falcor_python.ps1 -Command "python -m pytest tests/gpu -k 'metal_fused and (sampler or pdf)'"
scripts/run_falcor_python.ps1 -Command "python -m ncls.cli learn train <metal-proposal-smoke-config> <proposal-checkpoint>"
git -C external/Falcor status --short
git diff --check
```

训练和评测只使用冻结 config；长任务用 `tqdm` 按真实 batch/scene/spp 更新。结果写入 `artifacts/`。

## 回滚点

- component oracle/layout；
- proposal training；
- Slang parity；
- component/artifact conformance。

任一阶段失败时full method保持未完成，不导出Metal package；不得以analytic-only fallback或evaluator-only artifact声明sampler已完成。
