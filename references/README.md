# Reference package 固定入口

`references/` 是项目所有源材质 reference 的固定登记位置。任何 reference，无论实现由本项目维护、来自锁定上游，还是依赖下载的测量/纹理资产，都必须先在这里拥有唯一 package 记录，才能进入正式采集、viewer 和评测。

固定逻辑布局为：

```text
references/
  registry.json
  <reference-package-id>/
    README.md                 语义、适用范围、原生编辑能力和验证状态
    reference.json            后续稳定 manifest；实现前不得伪造字段
    adapters/                 仅放本项目维护的轻量 adapter，可选

external/<upstream>/          锁定的第三方 reference 源码克隆
assets/source-materials/
  <family-id>/<version>/      原始纹理、测量表和其他族专属大资源
data/reference-responses/
  <dataset-id>.h5             reference 查询后生成的唯一正式数据产品
```

职责边界：

- `references/` 保存 reference 身份和可审阅的项目 adapter，是唯一发现入口；
- `external/` 保存固定提交的第三方实现，不复制进根 Git 历史；
- `assets/source-materials/` 保存原始源材质资产，不与采集结果混放；
- `assets/viewer/` 保存 viewer 场景、HDRI 等原始运行资产；
- `data/reference-responses/` 只保存 `.h5/.hdf5` 派生监督；其他数组、cache 和旧实验输出进入 `artifacts/`；
- package 必须记录所有实现文件、上游提交、原始资源清单、许可证、哈希、查询语义和适用范围。

manifest 的 `path_root` 只有三种稳定映射：`project` 指项目根，`external` 指 `external/`，`source-materials` 指 `assets/source-materials/`。代码统一通过 `ncls.references.resolve_reference_path()` 解析，并拒绝绝对路径或越界的 `..`；调用方不得自行拼接另一套根目录。

当前随机游走实现仍位于历史正式路径 `shaders/ncls/reference/`，pbrt probe 仍位于 `tools/reference/`。它们已由本目录的 package 记录统一登记；在不改变数值语义并完成路径迁移测试前，不为了目录整洁移动 shader。

新增 reference 不得只把源码或资产随意放进 `tools/`、`assets/`、`data/` 或 `external/` 后由调用方猜测。调用方从 `references/registry.json` 和 package 入口解析身份，再按 manifest 定位实现与资产。

当前 active package 包括：LayerStack 随机游走、pbrt coated 独立交叉验证、OpenPBR 1.1.1、MERL 100 个测量 BRDF、8 个 MaterialX/Poly Haven 高分辨率纹理材质，以及原生 MDL/vMaterials 2.4.0。它们分别保留层模型、纯数学参数、测量表、原生图/纹理或 MDL program/资源，不共享内部 GT 表示。

MDL 正式路径由项目 bridge 直接调用锁定 MDL SDK，再由当前 Falcor 8 执行 SDK 生成的 HLSL；falcor2 只在隔离进程中作为数值验证 oracle，不能成为 provider、collector、live batch 或产品 CLI 的 fallback。MDL V1 当前固定 `ExplicitLod(0)`，已有逐方向 numerical parity；viewer 直接验证并加载同一 compiled artifact，可切换六种 vMaterials。独立 renderer image oracle 尚未建立，因此 `image_parity` 仍保持 pending。
