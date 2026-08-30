# Metal full evaluator 切片验证证据

## 验证边界

- 环境：完整 Windows；GPU 为 RTX 4090；使用唯一 `neural-shading` Conda 环境和锁定的 Falcor Release Python/D3D12 backend。
- 本任务验证的是 `metal-fused-neural-material@1` 的完整 evaluator 切片：codec、typed compiler、`prepare/evaluate`、hybrid head、proposal state reservation 与三种asset cook。matched `sample/pdf` 和 Package@2/Slang 属于后续任务，本切片在这些入口 fail closed。
- smoke 只缩短为8步、asset cohort为1项、patch为8；`metal_fused_full_v1` 的网络宽度、9 slots、32 typed tokens、全部18个required components与12个parameter groups均未缩小或关闭。

## 冻结 identity

- training config SHA-256：`abd117c4552be208145237e1330b1ff066389978812493e7eefd1e4de1c4a130`
- method descriptor SHA-256：`da9cb7acfb55c2c518c79f6949863a00bccbfba9c3a25c5a2e0d0f85745ddf8c`
- implementation identity：`4ccf288d4c7cfd205ed41cdd7217b21da4454c0d158c9c151ba6cfd3e1026cf4`
- final checkpoint SHA-256：`69f14b4aa300e544ba2547b1f8015339e26ae8d4972abafccd86b5155a1ebf7e`
- full-cohort preflight identity：`ab28319c67494b3d8b6f80031c7e0f4475d0dd504211a5c158fb8ee79257fda1`

对应ignored运行产物位于：

- `artifacts/08-30-metal-fused-full-method/evaluator-slice-verified.pt`
- `artifacts/08-30-metal-fused-full-method/evaluator-slice-verified-resumed.pt`
- `artifacts/08-30-metal-fused-full-method/evaluator-slice-verified.metrics.jsonl`
- `artifacts/08-30-metal-fused-full-method/full-cohort-verified-preflight.json`

## 已执行验证

```powershell
conda run -n neural-shading python -m compileall -q src tools/learning tests/unit/test_metal_fused_method.py tests/unit/test_metal_fused_preflight.py tests/gpu/test_metal_fused_model.py tests/gpu/test_metal_asset_cook.py
conda run -n neural-shading python -m pytest tests/unit -q
conda run -n neural-shading python -m pytest tests/gpu/test_metal_fused_model.py tests/gpu/test_metal_asset_cook.py -q
conda run -n neural-shading python tools/learning/preflight_metal_fused.py --output artifacts/08-30-metal-fused-full-method/full-cohort-verified-preflight.json
& scripts/run_falcor_python.ps1 -m ncls.cli learn train configs/learning/metal-fused-full-windows-smoke.json artifacts/08-30-metal-fused-full-method/evaluator-slice-verified.pt
& scripts/run_falcor_python.ps1 -m ncls.cli learn train configs/learning/metal-fused-full-windows-smoke.json artifacts/08-30-metal-fused-full-method/evaluator-slice-verified-resumed.pt --resume artifacts/08-30-metal-fused-full-method/evaluator-slice-verified.pt
& scripts/run_falcor_python.ps1 -m ncls.cli learn evaluate configs/learning/metal-fused-full-windows-smoke.json artifacts/08-30-metal-fused-full-method/evaluator-slice-verified.pt --batches 2
git diff --check
```

结果：

- Python bytecode compile通过；unit为`166 passed`；GPU full-model/asset-cook为`4 passed`。
- full cohort闭包覆盖692 exports、178 graphs、64 schema table entries、52 texture sets、36 recipes、全部6种参数类型和6个responsibility groups；greedy activation set使用3个exports激活全部18个required components。
- 真实online run经4步FP32 codec warmup与4步BF16 joint appearance完成；输出`TrainingCheckpoint@4`，final state为step 8 / complete。
- 12个required parameter groups均在step 7之前观测到finite gradient、nonzero gradient和实际optimizer update；checkpoint用当前descriptor严格load成功。
- completed checkpoint的resume成功并产生新SHA-256产物；两批真实online evaluation完成，`mean_loss=1.18959466`。
- 数值测试覆盖有限且非负的`f`、invalid access、typed missing/discrete语义、typed edit与bundle replacement分离、相邻mip、BF16精度敏感插值及三种新资产cook identity。

## report-only观察

这些数值只描述本次correctness smoke，不是质量门槛，也不用于自动改结构或扩大预算：

- codec phase loss：`0.232922 → 0.072625`；joint phase首步/末步loss：`2.14952 → 0.669627`。
- step 8 validation total：`0.563242`；其中semantic `0.056215`、normal angular `0.007441`、structured state `0.054716`、linear energy `0.104062`、peak support `0.567033`、reciprocity `0.004034`。
- 8步训练主体耗时`6.512 s`，step 8累计peak CUDA allocation为`393,659,904 bytes`；这些是batch=1/6、patch=8的Windows smoke观察值，不外推Linux long run吞吐。
- 静态runtime合同为maximum texture reads `106`、PreparedState `2816 bytes`；真实shader时间由runtime部署任务测量。
