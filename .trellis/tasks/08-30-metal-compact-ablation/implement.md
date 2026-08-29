# 执行计划

## 顺序

1. 从full design生成完整component/capacity/precision轴清单与interaction map；
2. 冻结ablation manifest、预算、seed、selection和evaluation identities；
3. 扩展generic method/profile config，确保每个变体静态预检与唯一identity；
4. 执行结构消融及预注册交互项，收集matched raw rows与bootstrap CI；
5. 执行capacity/precision/read sweeps和真实Slang microbench；
6. sampler child完成后补执行proposal-specific结构与容量sweeps；
7. 选择有限个observed non-dominated组合做distillation/QAT/refinement；
8. 为每个候选生成package，执行Python/Slang/viewer parity与生命周期测试；
9. 生成中文Pareto/组件贡献报告并更新experiment log；
10. 回到parent与用户对齐产品阈值、聚合口径和默认profile，不自动切换。

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit -k "ablation or method_profile or cost"
scripts/run_falcor_python.ps1 -Command "python -m pytest tests/gpu -k 'metal_fused or package or cost'"
scripts/build_viewer.ps1 -Configuration Release
scripts/benchmark_viewer.ps1 -Config <frozen-metal-ablation-config>
git -C external/Falcor status --short
git diff --check
```

所有长训练/评测使用冻结config并以`tqdm`显示真实进度；checkpoint、package、raw rows与报告写入`artifacts/`。

## 回滚点

- ablation manifest/identity；
- structure variants；
- capacity/precision variants；
- distillation/QAT；
- package/viewer candidates。

full baseline始终只读保留。protocol defect以新run identity修正；正常低质量或低效率结果不得通过改seed/预算覆盖。
