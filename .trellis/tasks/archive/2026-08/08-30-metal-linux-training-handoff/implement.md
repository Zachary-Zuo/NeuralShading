# 执行计划

## 顺序

1. [x] 生成并审计full-cohort preflight与stratified component activation manifest；
2. [x] 实现generic component execution/gradient/update/artifact coverage reporter；
3. [x] 由registry生成覆盖全部required components/groups与关键source语义的最小stratified训练子集，冻结Windows full-profile smoke的四phase step/batch/cadence和loss下降统计方法；
4. [x] 执行Windows online run、phase checkpoint/resume与数值/gradient检查；
5. [x] profile训练热路径、memory、sync与failure分类，修复implementation defect但不扩大预算追门；
6. [x] 导出diagnostic package并执行Python/QAT/Slang/viewer、bundle/edit和sample/PDF验证；
7. [x] 从同一semantic config生成Linux smoke/long profiles并执行config-diff/static platform检查；
8. [x] 编写Linux assets/deploy/preflight/train/resume/monitor/stop/recovery handoff；
9. [x] 实现long-run review manifest/summary生成器，不挂接formal/ablation命令；
10. [x] 完成父任务需求映射、Windows回归、Linux shell syntax和external clean检查。

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "metal or training or checkpoint or handoff"
scripts/run_falcor_python.ps1 -Command "python -m pytest tests/gpu -k 'metal_fused or package'"
scripts/run_falcor_python.ps1 -Command "python -m ncls.cli learn train <metal-full-smoke-config> <windows-smoke-checkpoint>"
scripts/run_falcor_python.ps1 -Command "python -m ncls.cli learn export <windows-smoke-checkpoint> <diagnostic-package>"
scripts/build_viewer.ps1 -Configuration Release
bash -n scripts/deploy_reference_linux.sh scripts/run_falcor_python.sh
git -C external/Falcor status --short
git diff --check
```

Linux交付命令写入`TESTING.md`和稳定deployment文档；实际Linux运行结果进入目标机`artifacts/`，不预写为Windows已验证。

## 回滚点

- activation/coverage manifest；
- optimization smoke；
- profiling fixes；
- package/viewer；
- Linux config/handoff；
- review summary。

正常低quality/throughput作为诊断结果进入handoff；只有implementation/protocol/resource defect允许当前child修复或回planning。
