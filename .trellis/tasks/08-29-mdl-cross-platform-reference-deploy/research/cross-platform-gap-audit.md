# 统一跨平台 Reference Backend：现状审计

## 1. 已存在的正确公共语义层

- `src/ncls/core/scattering/program.py`定义`ReferenceProgramDefinition`、`RuntimePayload`与`MaterialPayload`，已经是材质无关的program插件合同。
- `src/ncls/references/programs/registry.py`按source family/version唯一发现canonical program。
- `src/ncls/references/query.py`使用同一个query shader、typed binder、CUDA/Falcor shared buffers与lease执行`evaluate/sample/pdf`。
- `src/ncls/learning/producer.py`从source registry取得program并在线请求target；正式训练不保存batch。

结论：不应新建“统一材质/closure接口”。本任务应把现有dispatcher下面的平台执行能力抽出，并让上层通过backend capability创建session。

## 2. 五个canonical program的真实差异

| Program | Source family | runtime/material输入 | 额外code/toolchain依赖 |
|---|---|---|---|
| `ncls.layer-stack-random-walk@1` | `ncls.layer-stack@1` | packed layer/slab记录 | 无 |
| `ncls.merl-brdf@1` | `merl.measured-brdf@1` | 测量BRDF float table | 无；表属于assets |
| `ncls.openpbr@1` | `openpbr.material@1.1.1` | resolved inputs、Adobe LUT | OpenPBR/openpbr-bsdf |
| `ncls.materialx-polyhaven@1` | `materialx.document@1.39.4` | 原生document解析结果与纹理 | MaterialX |
| `ncls.mdl-vmaterials2@1` | `mdl.program@1` | generated HLSL、argument/RO/texture | MDL SDK、stb、bridge |

`references/registry.json`仍含部分旧source family文字，代码descriptor才是当前正式合同；实施时需收敛。

## 3. 当前平台泄漏

- `src/ncls/references/query.py`直接调用`import_falcor()`与`create_falcor_device()`。
- `src/ncls/references/falcor.py`持有`sys.platform`、D3D12/Vulkan选择，但只是函数集合，没有版本化descriptor/doctor/build identity。
- 多个GPU tests直接创建`falcor.DeviceType.D3D12`，因而不能复用为Linux公共backend验收。
- `run_falcor_python.ps1/.sh`与`build_falcor_python_linux.sh`分别拼平台build/runtime路径，尚无共同manifest。

## 4. MDL内部provider缺口

- `src/ncls/references/mdl.py`、`MdlFamilyDefinition`和`MdlReferenceProgram`原先直接依赖Windows SDK目录、DLL/EXE和concrete bridge。
- `tools/reference/mdl_sdk_bridge/main.cpp`原先直接包含`Windows.h`、调用`LoadLibraryW`并写死Windows plugin名。
- Windows fetch/build脚本各自重复SDK build、archive与layout常量。
- 官方同一锁定release提供Linux x86-64 archive：
  - `MDL-SDK-2025.0.0-387700.1252-linux-x86-64.tgz`
  - size `239371782`
  - SHA-256 `943a035bb08a4dce282a0f925ea2a0bd45a0bdcea3a4988c9e30c12ed316c5f4`
- Linux动态加载应使用`dlopen/dlsym`并链接`${CMAKE_DL_LIBS}`；实际archive内部layout与ABI仍须目标机验证。

## 5. Registry与validation边界

- `references/registry.json`还登记`ncls.pbrt-coated-crosscheck@1`，其role是`independent-validation`，没有Falcor runtime。
- falcor2只用于隔离MDL parity。
- 用户已决定两者不进入公共backend。公共discovery以代码中五个`ReferenceProgramDefinition`为准，stable registry应显式区分canonical ground-truth和validation-only。

## 6. Deployment与资产边界

- `scripts/fetch_reference_sources.ps1`负责OpenPBR/MaterialX等代码依赖；`fetch_mdl_sdk.ps1`与`fetch_stb.ps1`负责toolchain。
- `scripts/fetch_source_materials.py`、`fetch_mdl_assets.ps1`等负责`assets/`，不应被deployment调用。
- MERL、Poly Haven、OpenPBR example和vMaterials有各自资产/license manifest；用户会自行迁移这些内容。
- 用户已决定deployment允许联网获取锁定code/toolchain，但整个deployment不下载或修改任何材质资产，成功也不依赖assets存在。

## 7. Online training缺口

- NVIDIA method和source adapter registry已有LayerStack与MaterialX，没有`mdl.program@1`。
- MDL generated module当前只允许单snapshot，适合诚实的fixed-snapshot smoke，不适合在本任务宣称parameter-aware能力。
- `effect-pigment-metallic`已登记且无2D spatial texture/BSDF-data依赖，可用typed常量构造1×1 adapter；RO data仍由reference program真实处理。

## 8. 设计结论

1. 公共抽象应是platform/backend capability与query session，不是MDL compiler capability。
2. `ReferenceProgramDefinition`继续是五族插件边界；MDL compiler下沉为内部provider。
3. runtime与deployment共享一个root build manifest；source assets明确不在manifest中。
4. 不保留旧Falcor薄函数或Windows-only MDL常量的compatibility alias；一次性迁移项目调用方。
5. deployment compile成功与真实资产query成功分开报告；后者在用户复制assets后进入Windows/Linux质量gate。
