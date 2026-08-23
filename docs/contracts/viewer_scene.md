# Viewer scene 合同

## 它是什么

`ncls.viewer-scene@1` 是 NclsViewer 的可编辑场景状态 sidecar。Falcor 的 `.glb`、`.gltf` 等文件继续保存几何、instance 与 material slot；viewer scene 负责补上这些 slot 分别绑定哪一种源材质 reference，以及相机、HDRI 和物理灯光当前是什么状态。

它不是新的统一材质格式，也不把 MERL、OpenPBR 或 MaterialX 改写成 LayerStack。每个 slot 的 `source.family_id` 决定如何解释对应状态。

## 为什么需要

只保存 Falcor scene 无法恢复 viewer 中的源材质绑定；只保存一个 `--material` 路径也无法表达多 slot。原 capture v3 虽记录了每个 slot 的来源摘要，但没有保存 UI 修改后的参数，因此多 slot capture 不能完整 replay。

viewer scene 把“场景引用”和“族专属材质状态”分开：大资源仍位于 `data/` 或外部资产目录，JSON 保存可移植 URI、内容 identity 和必要的内嵌参数。加载时逐项验证 URI、identity 与参数，再创建相应的族专属 GPU 资源。

## 顶层字段

- `format_name = "ncls.viewer-scene"`、`format_version = 1`：格式 identity；
- `reference_integrator = "ncls.scene-path-tracer@1"`：有限深度 reference 语义；
- `geometry`、`environment`：相对 viewer scene 文件解析的 URI 与 SHA-256；
- `camera`、`lighting`、`reference`、`display`：共同的物理输入、积分上限和非物理显示状态；
- `active_material_id`：保存时 UI 正在编辑的 Falcor material slot；
- `material_bindings[]`：每个 Falcor material ID 恰好一项，不允许重复或遗漏。

方向光的 `sun_direction_to_light` 是从着色点指向光源的单位化方向向量。矩形灯由 `rectangle_center`、`rectangle_axis_u` 和 `rectangle_axis_v` 定义；两个 axis 都是 half-axis，`normalize(cross(U,V))` 是单面发光法线。

## 各材质族怎样保存

### LayerStack

`family_id = "ncls.layer-stack@1"`。`material_program` 内嵌完整 `ncls.material-program@1`，保存界面、均匀 slab、原生可编辑参数和显示名。加载后重新规范化为 `LayerStackIR`，但 scene 文件不持久化 backend packet。

### OpenPBR

`family_id = "openpbr.surface@1.1.1"`。`color_space` 与 `parameters` 保存当前 resolved native inputs，参数使用 OpenPBR 名称而不是 77-float runtime offset。`geometry_normal/tangent` 的 geometry binding 在命中点由局部 shading frame 提供；scene 中保存的 basis 值用于精确恢复当前 resolved state。

`source_uri` 与 `source_asset_sha256` 保留原始 OpenPBR source asset provenance。即使原文件暂时不可用，完整具名参数仍足以恢复当前 constant/resolved viewer subset；存在原文件时必须先验证它的 hash。

### MERL

`family_id = "merl.measured-brdf@1"`。测量表本身就是材质状态，因此只保存 `.binary` 的 `source_uri`、`source_asset_sha256` 和整个状态的 `state_sha256`。viewer 不提供不存在的粗糙度、金属度等伪参数；切换材质就是选择另一份测量表。

### MaterialX

`family_id = "materialx.textured-surface@1"`。`source_uri` 指向原生 `.mtlx`，`source_asset_sha256` 是文档与正式 surface-response subset 所引用纹理的组合 identity。`parameters` 只保存当前 UI 可编辑且没有被纹理连接驱动的 constant input override；纹理驱动的 base color、metalness 或 roughness 不允许被 scene override 静默替换。

MaterialX 文档、图结构和纹理始终是 GT 的一部分，加载时必须存在并通过 identity 验证。当前 displacement 图仍保留在源文档中，但不进入 surface-response reference query。

## Identity 与 replay

每个 `source` 同时记录 source asset identity 和 `state_sha256`。前者回答“原始测量表、文档或资源是什么”，后者回答“加上 UI 参数编辑后实际求值的状态是什么”。二者不能混用。

加载器要求：

1. 几何与 HDRI hash 匹配；
2. material binding 数量与 Falcor material slot 数量一致，ID 无重复且全部在范围内；
3. MERL/MaterialX 原生资源存在并通过 hash；
4. 各族参数有限、维度合法，纹理连接不被非法 override；
5. 重建后的 `state_sha256` 与文件记录一致。

capture v3 的 `viewer_scene` 字段引用同目录下的 `*-scene.json`。`--replay capture.json` 优先加载该 sidecar，再应用 capture 的 method/comparison 显示状态；也可以用 `--viewer-scene FILE` 直接打开 authoring state。
