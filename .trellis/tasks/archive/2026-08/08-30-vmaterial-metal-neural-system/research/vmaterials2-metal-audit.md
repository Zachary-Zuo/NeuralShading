# vMaterials 2 Metal 全目录审计

## 1. 它是什么

本文审计 NVIDIA vMaterials 2.4.0 的完整 `Materials/vMaterials_2/Metal/` 目录，为后续共同冻结 Metal neural material 的覆盖范围提供证据。审计先回答 source 中究竟有什么，再讨论模型；当前不把既有两个 neural cohort family 当作整个 Metal 系的代表，也不在本文中冻结最终 cohort。

审计对象保持 MDL 原生身份：module、exact export、typed 参数、纹理资源、compiled scattering 与额外输出能力共同属于 source。没有把它们转换为 LayerStack、OpenPBR 或固定 SVBRDF channels。

## 2. 证据与边界

审计使用：

- source pack：NVIDIA vMaterials 2.4.0；
- module root：`assets/source-materials/mdl-vmaterials2/2.4.0/Materials/`；
- MDL SDK：`2025.0.0-387700.1252`；
- bridge：`build/mdl-sdk-bridge/Release/ncls_mdl_sdk_bridge.exe`；
- 本次 bridge SHA-256：`d9203f9f7abf23481968c7c4e719afe39d40009acfc318802d5b59dd9a778385`；
- 运行环境：完整 Windows，RTX 4090、`neural-shading` Conda 环境、锁定 Falcor Windows 构建均可用。

两层审计均成功：

1. 对 127 个 `.mdl` module 调用 MDL SDK module discovery，权威枚举 exact exports；
2. 对全部 837 个 exports 做 metadata-only class compilation，读取 typed 参数、compiled/sub-expression identity、资源 metadata 和 capability audit。

两层均为 0 failure。metadata-only compilation 不解码、复制或训练 4K texture，也不是 837 个材质的 full runtime/GPU parity。当前只有已登记的 Copper Patinated 与 Aluminum Scratched 具有既有正式 reference 证据；其余条目需要在纳入 cohort 后完成 decoded artifact、统一 backend 和代表性 GPU query 验证。

任务 scratch 中保留可再生的中间数据：

- `scratch/metal-module-audit/summary.json`：127 个 module、837 个 exact exports；
- `scratch/metal-export-inspection/summary.json`：逐 export inspection；
- `scratch/metal-audit-analysis.json`：逐 module 聚合与词法特征。

这些文件被 task-local `.gitignore` 忽略；本文保存稳定结论，不把 8 MB inspection JSON 提交进根仓库。

## 3. 总体盘点

| 项目 | 审计值 | 含义 |
|---|---:|---|
| MDL module | 127 | Metal 根目录 124 个，`Mesh/` 子目录 3 个 |
| authored export | 837 | preset/entry，不等于 837 套独立网络 |
| parameter schema | 64 | module 间存在大量同构模板，也存在特殊 typed 接口 |
| compiled graph identity | 193 | preset 的常量、分支和子表达式会形成不同 compiled identity |
| texture set identity | 57 | 多个金属/finish module 共享同一组纹理资源 |
| 唯一 source texture path | 140 | 118 JPG、22 PNG，压缩文件合计 326,096,250 bytes（约 311 MiB） |
| opaque surface export | 692 | capability audit 未发现 cutout、emission、volume 或 displacement |
| cutout surface export | 145 | `surface.scattering` 之外还使用 `geometry.cutout_opacity` |
| discovery/inspection failure | 0 | 只证明 SDK discovery 与 metadata compilation 成功 |

全部 837 个 exports 都有 `surface_bsdf_evaluate=true`，且本次未发现 emission、volume 或 displacement。差异只剩两种 capability identity：692 个 opaque 与 145 个 cutout。每个 compiled export 当前都使用一个 DF handle；argument block 为 48–144 bytes，RO data 为 0–2048 bytes。generated HLSL 大约 29–149 KB，只作为 source program 复杂度线索，不是 neural runtime 成本或质量指标。

## 4. 材质内容分组

### 4.1 标准金属 × 表面 finish 矩阵

主体是 13 种金属身份与 7 类 finish 的规则矩阵，共 91 个 module、579 个 exports：

- 金属身份：Aluminum、Brass、Bronze、Chromium、Copper、Gold、Iron、Nickel、Platinum、Silver、Titanium、Tungsten、Zinc；
- finish：Base、Brushed、Foil、Hammered、Knurling、Scratched、Sheet。

| finish | module | export | 去重后的相关纹理路径 | cutout export | 主要压力 |
|---|---:|---:|---:|---:|---|
| Base | 13 | 63 | 5 | 0 | 金属颜色/IOR、粗糙度、smudge |
| Brushed | 13 | 75 | 5 | 0 | 各向异性方向、brush normal/roughness |
| Foil | 13 | 86 | 5 | 0 | crumpled normal、双 roughness、AO/smudge |
| Hammered | 13 | 43 | 6 | 0 | pit/hammer normal、directional factor、污染层 |
| Knurling | 13 | 30 | 7 | 0 | 规则高度/normal、curvature、grunge |
| Scratched | 13 | 86 | 8 | 0 | 多尺度 scratch masks、normal、粗糙度与污渍相关性 |
| Sheet | 13 | 196 | 6 | 110 | sheet streak/roughness；部分 module 另含十种 punched topology |

除 Brass/Bronze 的若干扩展版本外，同一 finish 下的大部分 module 共享 parameter schema 和 texture set，只改变金属身份、默认参数或少量 compiled 常量。例如 11 个非 Brass/Bronze 的 `*_Scratched` module 共用 18 参数 schema 和同一组 3 张 scratch texture；11 个 `*_Sheet` module 的 punched 分支共用 25 参数 schema。

这构成最清楚的组合结构：

```text
metal identity / optical state
× finish texture bundle
× authored continuous/discrete parameters
→ native MDL scattering
```

它非常适合检查一个 shared method 能否把“金属光学身份”和“空间 finish 资产”分开编码，但审计本身不证明这种分解在函数空间里完全可分。

### 4.2 老化、涂层、抛光与复杂表面

这一组有 14 个 module、113 个 exports：

- `Aging_Copper`
- `Aluminum_Anodized`
- `Blued_Steel_Cold`
- `Brass_Antique`、`Brass_Polished`
- `Bronze_Antique`、`Bronze_Polished`
- `Copper_Antique_Brushed`
- `Copper_Antique_Brushed_Patinated`
- `Iron_Pitted_Steel`
- `Steel_Galvanized`
- `Steel_Painted`
- `Steel_Painted_Cracked`
- `Zinc_Galvanizing`

它们覆盖 oxidation/patina、paint/metal mix、crack/damage、wear、rust、haze、dirt、normal mixing 与多层 roughness。代表性 typed 自由度包括：

- `Aluminum_Anodized`：22 个参数，控制 coating、anodization roughness/bump、颜色、abrasion、wear、scratch、smudge 和 dirt；它是 127 个 module 中唯一不使用 GGX Smith、改用 Beckmann/simple-glossy 组合的 source；
- `Copper_Antique_Brushed_Patinated`：11 个参数，控制 metal roughness、patina/metal blend、blend softness、corrosion offset、bump 和 smudge；
- `Steel_Painted_Cracked`：19 个参数，控制 paint color/roughness、crack bump/darkness、rust damage、wash、dirt 与多组 normal strength；
- `Iron_Pitted_Steel`：9 个参数、9 张纹理，包含 pit 资源选择、rust、heat treatment 与 roughness。

这组不是简单换 base-color。它们把金属与 dielectric/diffuse contamination、mask 和多个 normal source 组合起来，是 neural evaluator 是否保留原生层/混合语义的主要压力集。

### 4.3 工业特殊材质

这一组有 9 个 module、51 个 exports：

- `Mercury`
- `Metal_Cast`
- `PCB_Copper`、`PCB_Goldfinger`
- `Solder_Paste`
- `Stainless_Steel`
- `Stainless_Steel_Brushed`
- `Stainless_Steel_Milled`
- `Steel_Carbon`

它们补充 cast/solder、PCB mask、stainless damage/smudge、milled/brushed anisotropy 和 carbon-steel contamination。`Mercury` 在当前 source 中仍是 surface metal，而不是 volume/liquid transport capability。

### 4.4 结构纹理、板材与 cutout

这一组有 13 个 module、94 个 exports：

- 6 个 `Diamond_Plate_*` 图案；
- 3 个 `Mesh/Metal_Mesh_Weave_*`；
- `Punched_Circular_Plate`；
- `Bronze_Sheet_Punched`；
- `Stainless_Steel_Punched`；
- `Stainless_Steel_Brushed_Punched`。

普通 diamond/mesh 条目仍然只输出 opaque surface scattering；它们用 normal/height/AO/mask 模拟结构外观，不等于真实几何或 silhouette。真正 punched 条目和 11 个标准 `*_Sheet` module 中的 punched presets 会输出非平凡 `geometry.cutout_opacity`。

145 个 cutout exports 的来源恰好是：

- 11 个标准 sheet module × 10 个 punched preset = 110；
- `Bronze_Sheet_Punched` = 10；
- `Punched_Circular_Plate` = 4；
- `Stainless_Steel_Punched` = 10；
- `Stainless_Steel_Brushed_Punched` = 11。

当前公共 `evaluate()` 只返回线性 RGB `f`，没有 opacity/coverage 输出与 renderer composition 合同。因此这 145 个条目必须 fail closed；把它们当作 opaque、只压 normal 或忽略洞口会改变 source GT。

## 5. 参数与纹理审计

### 5.1 Typed 参数

- 每个 export 有 9–31 个 editable 参数；
- 全目录出现 163 个不同参数名；用户确认的 692 个 opaque exports 中出现 154 个，另外 9 个名字全部是 cutout 专用参数；
- 类型包括 `float`、`bool`、`float2`、`int`、`enum` 和 `color`；
- 常见公共轴是 `texture_scale/translate/rotate`、roughness、bump strength、round-corner 开关/半径和 UV space；
- family-specific 轴包括 patina blend、corrosion、brush direction、scratch/dirt correlation、paint/crack/rust、cutout shape/grid 等。

opaque 参数的完整名称、类型与运行职责分析见 `metal-typed-parameters.md`。cutout-only 的 9 个名字不进入 Metal-v1 catalog 或 neural 参数合同。

837 个 authored exports 没有任何 editable `texture_2d` 参数。source texture 是 module 内部固定资源，而不是用户可直接替换的 authored argument。113/127 个 module 的全部 presets 只使用一个 texture-set identity；发生两个 texture set 的 14 个 module 都与 opaque/punched 分支切换相关。

因此：

- authored preset 的大多数差异是同一 texture set 上的参数/default 变化；
- 从 Aluminum Scratched 切换到 Copper Scratched 是换 source module/metal identity，不是换三张 scratch 文件；
- 从 Scratched 切换到 Brushed/Foil/Patina 才对应不同空间纹理 bundle 和图结构；
- “允许任意用户图像替换 MDL 内部纹理”不是 vMaterials Metal 当前原生编辑自由度，若要支持，需要额外定义 reauthor/import contract，不能伪装成已有 typed edit。

### 5.2 Texture 资产

140 个唯一 source texture 的尺寸分布为：

| 尺寸 | 数量 |
|---|---:|
| 4096×4096 | 94 |
| 2048×2048 | 29 |
| 1024×1024 | 9 |
| 其他 | 8（含 1 张 8192² 与若干 lookup/gradient） |

pixel type 为 132 个 `Rgb`、5 个 `Sint8`、2 个 `Rgba_16`、1 个 `Rgba`；118 个以 linear gamma 使用，20 个 sRGB，2 个 default。颜色、normal、roughness、mask、AO 和 packed channels 不能用一个无语义的 RGB loss 等价处理；neural texture asset 必须保存 channel role、transfer function、filter/LOD 与 source provenance。

## 6. Closure 与图结构结论

源码词法审计与 compiled capability 共同显示：

- 126/127 个 module 使用 `microfacet_ggx_smith_bsdf`；唯一例外是 `Aluminum_Anodized`；
- 112 个使用 `weighted_layer`；
- 68 个包含 diffuse reflection；
- 66 个使用 color custom-curve layer；
- 40 个使用 custom-curve layer；
- 30 个使用 directional factor，主要分布在 hammered/knurling、brushed patina、paint 和 mesh；
- normal/texture mixing 极普遍，80 个 module 使用 tangent-space normal texture。

因此 Metal 目录不是 127 种毫无关联的 BSDF，也不是一套固定 PBR channel。更准确的结构是：高度复用的 metal/finish 模板，加上少数复杂 coating/contamination/structure 图。一个质量优先方法有充分理由共享方向 evaluator 与 texture decoder，同时保留 family/module typed compiler 和资产级差异。

## 7. 对“替换 neural texture”的精确定义

审计表明至少有三种不同强度的替换语义：

1. **同 finish、换 metal identity**：例如 `Aluminum_Scratched → Copper_Scratched`。空间纹理 bundle 基本共享，变化主要在金属光学身份与参数；这最适合测试 shared decoder 的组合泛化。
2. **换 finish neural texture asset**：例如 `Scratched → Brushed/Foil/Hammered/Patina`。需要替换压缩的多通道 texture/latent bundle，并让同一个 method 支持不同 channel schema 和 MDL 图条件；这正是用户要求的核心资产替换能力。
3. **换拓扑/输出 capability**：例如 opaque sheet → punched sheet。它不仅替换 texture，还增加 opacity/coverage 和 silhouette 语义；必须扩展公共材质输出与 renderer composition，不能只换 latent texture 后仍调用现有 RGB `evaluate()`。

另有第四种、当前 source 没有原生提供的能力：导入任意外部 texture set，自动映射到某个 Metal template 并编译。这需要单独定义 template/channel authoring schema，不能从现有 837 个 authored exports 自动推出。

## 8. 目标范围的三个可选边界

### A. 全 opaque Metal（推荐）

- 覆盖 127 个 module 中的全部 692 个 opaque exports；混合 module 只纳入其 opaque 子族；
- 145 个 cutout exports 不进入 Metal-v1 catalog；上游 vMaterials source pack 与审计 provenance 保持不动；
- 训练/评测按 module、finish、metal identity、参数状态和 texture asset 组合拆分；
- 优点：覆盖完整 Metal surface scattering 语义，同时不把 opacity/geometry 问题混入 evaluator；
- 代价：source/corpus 和 typed compiler 明显大于当前两个 family，需要系统化 family adapter 与资产 schema。

### B. 全目录含 cutout

- 覆盖全部 837 个 exports；
- 必须同时设计 `opacity/coverage` 输出、训练监督、filtering、viewer composition、PT visibility 与 package ABI；
- 优点：真正覆盖完整目录；
- 代价：任务从 neural scattering 扩张为 scattering + geometry coverage program，模型与代码架构都必须同步扩大。

### C. 代表性 module 子集

- 从 finish matrix 与复杂组各选少数 module；
- 优点：实现与训练范围小；
- 代价：容易退回 fixture，不能支撑“表达 vMaterials Metal 系”的强结论，也削弱 texture/metal/参数组合泛化的价值。

用户已经确认采用 A 的 capability 边界：Metal-v1 只建立 opaque neural catalog；cutout 不登记到该 catalog，也不由当前 RGB evaluator 近似。是否把全部 692 个 opaque authored exports 都纳入首版训练/评测，仍需在 typed 参数目标冻结后确认。

## 9. 尚未由审计决定的事项

- 692 个 opaque exports 是否全部进入首版训练/评测；cutout 已确认不进入 Metal-v1 catalog；
- “换 texture”是否只要求在已 authored Metal assets 之间切换，还是还要求导入任意外部多通道 texture set；
- full opaque 训练是一个全局 shared method，还是共享 runtime + family/module adapter；这属于需求冻结后的模型设计，不由当前代码结构倒推；
- 692 个 opaque exports 的 full decoded runtime、GPU query 和参数极值仍需在实施阶段按冻结 cohort 做 preflight，metadata-only 成功不能代替这些证据。
