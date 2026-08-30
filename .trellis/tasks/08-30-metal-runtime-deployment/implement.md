# 执行计划

## 顺序

- [x] 从layout JSON生成Python/Slang ABI与packing tests；
- [x] 实现BF16/QAT Python inference oracle和program/asset/instance packers；
- [x] 实现Metal Slang decoder/access/compiler/evaluator/sample/pdf与GPU kernel tests；
- [x] 为`ScatteringPackage@2`登记Metal INT8/FP16 typed resources和三section validation；
- [x] 扩展viewer generic resource factory与Metal三层bindings，不改变canonical viewer类型；
- [x] 实现required editable-material compute entry、schema UI、atomic state update；
- [x] 接入deferred和PT full capability parity；
- [x] 执行package parity、viewer capture、Release build和Falcor clean；
- [x] 记录`B_shared/B_asset/B_instance`、prepare/eval/sample/pdf静态计数与viewer diagnostic timing。

## 重点文件

- `shaders/ncls/backends/metal_fused/`
- `src/ncls/bundle/typed_texture.py`、`writer.py`、`loader.py`
- `src/ncls/learning/methods/metal_fused.py` program/asset/instance compilers
- `apps/viewer/ScatteringPackage.*`
- `apps/viewer/NclsViewer.*`及package shaders
- package/viewer unit与GPU parity tests

## 验证

```powershell
conda run -n neural-shading python -m pytest tests/unit/test_scattering_package.py tests/unit/test_viewer_slots.py tests/unit/test_viewer_studio.py
scripts/run_falcor_python.ps1 -m pytest tests/gpu/test_metal_runtime_package.py -q
scripts/build_viewer.ps1 -Configuration Release
external/Falcor/build/windows-vs2022/bin/Release/NclsViewer.exe --headless --request .trellis/tasks/08-30-metal-runtime-deployment/scratch/metal-runtime-capture-request.json
git -C external/Falcor status --short
git diff --check
```

## 回滚点

- generated ABI/packer；
- Slang core；
- package resources；
- runtime cache/asset swap；
- typed editor。

candidate program/asset/instance创建失败时只丢弃candidate，不修改active slot。禁止恢复Package@1或evaluator-only Metal package路径。
