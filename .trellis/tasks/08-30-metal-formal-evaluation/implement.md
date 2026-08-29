# 执行计划

## 顺序

1. 扩展generic evaluation manifest/runner与raw metric schema；
2. 生成并审计六类split，执行泄漏测试；
3. 实现local/energy/peak/reciprocity/semantic/sweep metrics；
4. 实现source-state bootstrap与matched comparison；
5. 接入bundle/checkpoint静态bytes/MAC/read账本；
6. 实现GPU-only query、coherent/divergent working-set和viewer workload计时；
7. 建立BC4/5/7 conventional control和optimized-source control；
8. smoke后冻结formal manifest，执行一次formal矩阵；
9. 解释failure类别，生成中文report并更新experiment log。

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "evaluation or bootstrap or split"
scripts/run_falcor_python.ps1 -Command "python -m pytest tests/gpu -k 'metal_fused or cost'"
conda run -n neural-shading python -m ncls learn evaluate <frozen-config> <checkpoint> --batches <frozen-count>
scripts/benchmark_viewer.ps1 -Config <frozen-viewer-config>
git diff --check
```

长评测使用`tqdm`按真实state/query/workload更新；结果统一写`artifacts/`，不把run日志写回PRD。

## 回滚点

- manifest/split；
- metrics/bootstrap；
- cost controls；
- formal run/report。

formal开始后只有implementation/protocol defect允许修改并以新identity重跑；observed outcome不改变冻结合同。
