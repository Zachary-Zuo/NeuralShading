# 执行计划

## 顺序

1. 更新project/core/data/learning/viewer specs和稳定contracts，冻结v4/v2 schemas及旧symbol denylist；
2. 实现`ReferenceExecutionPlan`与backend/session group routing，递归迁移五个reference和query tests；
3. 实现`NativeAssetCollection`、三类typed batches与collection adapters，删除`NativeFeaturePyramid`；
4. 实现`MethodDescriptor@2` component/phase contracts和generic conformance fixtures；
5. 实现`TrainingConfig@4`、phase runner、parameter groups、precision、prefetch和`TrainingCheckpoint@4`；
6. 迁移NVIDIA method、四个configs、CLI train/evaluate/export和checkpoint tests；
7. 实现`ScatteringPackage@2` Python schema/writer/loader及program/asset/instance typed payload；
8. 迁移NVIDIA exporter、Slang package、C++ viewer loader/cache/bindings和双slot tests；
9. 删除v3/v1 readers/schemas/aliases/converters/fallbacks及过时docs/tests；
10. 执行静态denylist、unit、GPU/reference、NVIDIA online training/package/viewer回归和external clean检查。

## 重点文件

- `.trellis/spec/`、`docs/contracts/`、`docs/architecture.md`、`docs/learning.md`
- `src/ncls/references/`、`src/ncls/learning/`、`src/ncls/bundle/`、`src/ncls/cli.py`
- `configs/learning/`
- `apps/viewer/ScatteringPackage.*`、`MaterialProgram.*`、`NclsViewer.*`及package shaders
- 对应unit/GPU/integration/viewer tests

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit
scripts/run_falcor_python.ps1 -Command "python -m pytest tests/gpu"
scripts/build_viewer.ps1 -Configuration Release
rg -n "TrainingConfig@3|TrainingCheckpoint@3|ScatteringPackage@1|NativeFeaturePyramid|ncls.training-config.*3|ncls.training-checkpoint.*3|ncls.scattering-package.*1" src tests configs apps docs .trellis/spec
git -C external/Falcor status --short
git diff --check
```

denylist命令预期只允许明确的历史说明fixture；正式代码/config/spec为零，test对旧format执行拒绝而不是加载。

## 回滚点

- reference plan/session；
- asset collection/batches；
- method/config/checkpoint；
- NVIDIA migration；
- package v2；
- viewer migration；
- legacy deletion。

回滚使用git提交/模块边界，不通过恢复旧公共API或兼容reader实现。
