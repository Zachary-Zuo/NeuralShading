# 执行计划

## 顺序

1. 生成并校验opaque Metal registry/schema；把现有scratch审计变成稳定生成逻辑，原始资产仍在`assets/`；
2. 使用child 1的`ReferenceExecutionPlan`生成Metal execution groups与aggregation tests；
3. 实现MDL compiled-material offsets和同group argument-block aggregation；
4. 实现canonical session-pool group routing，先验证同group，再验证跨group；
5. 生成52-asset `NativeAssetCollection`与role/schema/mip/tile tests；
6. 扩展generic pipeline选择group-homogeneous、asset/tile-coherent batch；
7. 实现deterministic typed-state recipe、pool identity、v4 checkpoint resume；
8. 实现footprint integration与GPU数值oracle；
9. 补齐current/optimized source成本计数和全量静态账本；
10. 运行全回归并冻结registry/plan/asset/query IDs供full method使用。

## 执行结果

1–10均已完成。稳定inspection/generator替代task scratch逻辑；registry、typed-state recipe、lazy grouped session、content-addressed file resource、52-set collection、完整footprint积分与report-only ledger均进入canonical路径。验证命令、identity和observed诊断见`research/verification-evidence.md`。

## 重点文件

- `src/ncls/source_materials/mdl_metal.py`
- `references/mdl-vmaterials2-v1/metal-opaque-v1.json`及schema
- child 1冻结的reference plan/asset collection modules
- `src/ncls/references/query.py`
- `src/ncls/references/programs/mdl.py`
- `src/ncls/learning/producer.py`与source adapter registry
- `shaders/ncls/reference_query/reference_query.cs.slang`
- `shaders/ncls/reference_backends/mdl*.slang*`

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_mdl_source.py tests/unit/test_mdl_vmaterials_catalog.py tests/unit/test_online_training_producer.py tests/unit/test_training_checkpoint.py -q
scripts/run_falcor_python.ps1 -m pytest tests/gpu/test_reference_query_dispatcher.py tests/gpu/test_reference_backend_contracts.py tests/gpu/test_mdl_native_crosscheck.py -q
conda run -n neural-shading python -m pytest tests/integration/reference
git diff --check
```

正式新增tests必须覆盖registry regeneration、group key、offset packing、跨group routing、state split/resume、footprint zero/finite/continuity和lease failure matrix。

## 回滚点

- registry独立提交；
- group metadata/MDL offset独立提交；
- session-pool routing独立提交；
- asset collection独立提交；
- state/footprint recipe独立提交。

任一实现失败时回滚本child模块提交并修正canonical合同；不得恢复旧single-session/pyramid接口、Metal专用producer或response缓存。
