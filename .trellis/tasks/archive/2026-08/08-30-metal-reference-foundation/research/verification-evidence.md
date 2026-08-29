# 验证证据

## 环境

- 状态：完整 Windows；GPU 为 NVIDIA GeForce RTX 4090；`neural-shading` Conda 环境与 Falcor Windows Release Python 均存在。
- 训练与reference数据仍只走GPU-resident online query；本任务没有写入response batch/corpus。

## 冻结身份与静态审计

- tracked registry：`fa6642e60d469231839756d749283b3d7d93e7163284c4094837770379dec8cc`。
- registry严格回读：837 authored、692 opaque、145 cutout-rejected、178 opaque graphs、52 texture sets、64 authored schemas；source closure为267个文件。
- report-only observed ledger：`artifacts/08-30-metal-reference-foundation/observed-ledger.json`，ledger identity为`b8633c72de53a7803647a222aa6e75bd1a32e37846938eadf8b12c7cf1939911`。
- 静态账本记录323,338,273 bytes authored compressed textures、5,185,342,464 bytes unique decoded textures/SDK tables；这些值不是hard gate。

## Registry / asset / state

- `scripts/build_mdl_metal_registry.ps1`默认source-closure/identity检查通过；`-Refresh`公共入口会调用锁定SDK的127 module discovery与837 exact export class inspections，不依赖task scratch实现。
- unit验证unknown/missing/cutout fail closed、六组责任完整、最多9 slots、16-bit与provider BSDF provenance。
- integration真实读取一个`Rgba_16` tile和一个SDK BSDF-data tile；都经同一52-asset `NativeAssetCollection@1`、memmap、mip、tile+halo和bounded lease路径，未建立Metal专用producer。
- generic `expand_source_states()`用公共MDL editor生成train/validation Sobol states；两split pool identity不同，除共享authored default外的采样snapshot IDs不重叠。producer resume在恢复任何cursor前校验`typed_state_pool_identity`。

## Reference / footprint

- 同一`Aging_Copper` graph的4个typed states被编入单一execution group；argument offsets非重复且16-byte aligned；GPU `evaluate/sample/pdf`均finite且sample/PDF一致。
- 纹理fixture验证：零UV derivative时64个footprint samples退化为中心response；非零footprint改变完整线性`f`；`footprint_samples`与`evaluation_samples`分别进入shader/host合同。
- invalid footprint写NaN并置`valid=false`，producer只在GPU压实有效行，不产生零target。
- 3-export真实probe产生3个groups，plan identity为`2fcdef72130204b045ba0bad1ab7dfc77324c741a1835410fbe4ac4c61f93b9d`；resident容量2后的group数量依次为1/2/2，证明lazy LRU没有展开全量group资源。
- authoritative与`prepare-hoisted-pdf-reuse@1`在3个真实groups上的最大绝对差均为0；首次group materialization与warm query分别记账，未把shader编译时间冒充query cost。

## 最终命令

```powershell
conda run -n neural-shading python -m compileall -q src/ncls tools/reference tests
conda run -n neural-shading python -m pytest tests/unit -q
# 158 passed
conda run -n neural-shading python -m pytest tests/integration/reference -q
# 5 passed
scripts/run_falcor_python.ps1 -m pytest tests/gpu/test_reference_query_dispatcher.py tests/gpu/test_reference_backend_contracts.py tests/gpu/test_mdl_native_crosscheck.py -q
# 14 passed
scripts/build_mdl_metal_registry.ps1
git diff --check
```

`jsonschema`未加入项目环境，因此没有临时安装新依赖；tracked schema由既有schema枚举unit test、严格`MdlMetalRegistry` loader与generator回读共同验证。
