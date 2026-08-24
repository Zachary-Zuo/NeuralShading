# 源材质族、reference 与统一神经材质程序

## 真实需求

项目要处理的输入不是预先被归约成某一种层参数的材质，而是多个保持原生语义的**源材质族**。源材质可以是纯数学散射模型、可编辑 shader graph、程序材质、带高分辨率纹理的空间变化材质、测量 BRDF/BSDF/BTF，或以后接入的非局部材质系统。

每个源材质族用适合自己的 reference 实现产生 ground truth（GT，真实目标）外观或散射。后续研究方法再把这些不同来源编译成统一、随机访问、运行成本有界的 neural material program：

```text
源材质资产
  原生参数 + 原生图结构 + 纹理/测量/几何资源
        ↓
该材质族自己的 reference 实现
        ↓
统一查询语义下的方向响应、采样结果或图像 GT
        ↓
训练 / direct fit / 评测
        ↓
跨源材质族的 neural evaluator、latent 与 MethodBundle
```

这里统一的是目标运行时的查询方式：`compile_material → prepare → neural evaluate`，以及按 capability 增加的 matched `sample/pdf` 和专用积分。它不统一 reference 内部的材质表示；不同 reference 不需要共享 `LayerStackIR`、参数布局、求值算法或资源类型。

## GT 原则

### 原生表达就是 GT 的一部分

源材质由什么参数、图结构和资源定义，应由该源材质自身决定：

- 如果源材质是解析模型，它的公式和原生参数是 GT；
- 如果源材质是可编辑图或程序材质，图结构、节点参数和资源连接是 GT；
- 如果源材质包含高分辨率纹理，纹理及其颜色空间、通道语义、坐标、过滤和原生 closure 是 GT；
- 如果源材质是测量 BRDF、BSDF 或 BTF，测量表、参数化和插值规则是 GT；
- 如果源材质原生提供可调参数，项目必须保留这些参数的编辑能力，并让 reference 对编辑后的材质重新正确求值。

除非源材质原本就是层模型，否则它不需要提供层数、层参数或任何人为反演出的分解。项目不能为了迁就当前 `LayerStackIR` 或某个拟合 backend，先把源材质改写成另一种模型后再把改写结果称为 GT。

测量材质可以没有连续可编辑参数；这种情况下，原始样本身份、测量值和有明确定义的变换构成它的可用编辑范围。项目不能凭空声称不存在的物理参数是真值。

### reference 可以有多个实现

`reference` 是“对某个源材质族具有权威语义的求值实现”，不是某一个固定 shader 的专名。项目可以同时存在：

- 多层介质的 Monte Carlo 随机游走 reference；
- 解析 BRDF/BSDF 的公式求值 reference；
- MaterialX、OpenPBR 或其他图系统的原生/受控 reference adapter；
- 程序纹理与高分辨率纹理资源求值 reference；
- 测量 BRDF/BSDF/BTF 的查表和重建 reference；
- 以后为 BSSRDF、纤维或参与介质提供的专用 reference。

reference adapter 只负责把共同查询所需的几何、方向、光谱/颜色、随机流和输出测度对齐。它不得改变源材质的物理语义，也不得先经过项目要研究的 neural material program。

同一源材质族可以保留多个独立 reference 实现用于交叉验证。数据和报告必须记录实际使用的 reference 身份、版本、资源哈希和适用范围。

## 三类数据必须分开命名

为避免把源材质与采集结果混为一谈，后续文档使用以下名称：

1. **源材质资产/源材质语料**：原始材质定义、可编辑参数、图、纹理、测量表和其他原生资源；
2. **reference 响应数据**：reference 对查询采样后产生的方向响应、方差、图像或其他监督；
3. **方法产物**：direct-fit latent/evaluator、共享 decoder、compiler checkpoint、matched sampler、`MethodBundle` 和评测结果；解析 closure 产物属于对照或可选 physical core。

`ncls.reference-dataset@4` 只属于第二类。它不是源材质库，也不规定第一类必须如何表达。

## 候选源材质族

下表先记录覆盖方向，不表示已经承诺实现顺序。每个材质族应保留其原生表达，再通过自己的 reference 接入共同评测。

| 源材质族 | 原生表达与资源 | 合适的 reference | 当前状态 |
|---|---|---|---|
| 多层界面与均匀 slab | 可编辑界面、介质和层顺序 | 随机游走 | 已实现第一版 |
| 解析局部散射 | Lambert、rough diffuse、microfacet dielectric/conductor、sheen/fuzz、retroreflection 等公式及参数 | 解析求值或独立路径积分 | OpenPBR 1.1.1 第一版已接入 |
| 可编辑 layer/mix 图 | 任意合法组合图、权重和原生参数 | 图解释器或原生 shader | 待扩展 |
| 薄膜与虹彩 | 光谱 IOR、膜厚、粗糙度和层结构 | 光谱 thin-film reference | 待扩展 |
| microflake、珠光和汽车漆 | flake 分布、粒径/朝向、颜料和涂层参数 | 专用统计或微几何 reference | 待扩展 |
| 高分辨率纹理材质 | 原生 closure 加 base color、roughness、normal、height、anisotropy 等纹理及物理尺寸 | 完整纹理求值 reference | 8 个 MaterialX/Poly Haven 4K 材质已接入 |
| 程序材质 | 噪声、图案、坐标变换和可编辑程序参数 | 程序图 reference | 待扩展 |
| 测量 BRDF/BSDF | 原始测量表、方向参数化、标定和插值规则 | 测量数据查表/重建 | MERL 100 个测量 BRDF 已接入 |
| BTF 与空间微结构外观 | 随位置、观察和光照变化的测量数据，或高分辨率微几何 | BTF 查表/重建或微几何路径追踪 | 待接入 |
| 透射与 thin-walled 材质 | 原生 BSDF、边界和内部介质参数 | 支持透射路径的 reference | 需扩展 renderer capability |
| BSSRDF、皮肤、蜡、石材 | 非局部散射模型及其原生参数/测量 | BSSRDF reference | 独立 capability 候选 |
| 头发、毛发和纤维 | 曲线/纤维几何及纵向、方位散射参数 | fiber/path reference | 独立 capability 候选 |
| 参与介质 | 密度场、相函数、吸收、散射和发光资源 | volume path reference | 独立 capability 候选 |

高分辨率 PBR/SVBRDF 纹理集可以作为一种源材质族，但纹理 map 本身不完整定义外观；必须同时固定其原生 closure、颜色空间、单位、切线约定、过滤和位移语义。类似地，光学常数数据库可以给解析或薄膜源材质提供原生参数，但不自动定义粗糙度、纹理或材质结构。

## 已完成的源材质接入与当前优先级

本轮不按层数或材质名称堆功能，而用少量语义差异明显的源材质族验证完整架构。以下 1–5 已按顺序完成：

1. **reference package 与源资产登记路径**：根目录 `references/` 已成为唯一入口，并分开管理 reference 身份、第三方实现、原始源材质资产和派生响应数据。
2. **LayerStack reference 的独立验证**：pbrt `CoatedDiffuseBxDF` 与 `CoatedConductorBxDF` 均已覆盖代表性的粗糙度、各向异性、IOR、吸收/散射和方向切片，没有扩展成按 `N=2/3/4/...` 枚举。
3. **OpenPBR 1.1.1 纯数学、原生可编辑材质族**：完整 resolved input 字段、83 个官方 MaterialX 示例索引、常量编辑 round-trip、直接纹理 binding、独立 `eval/sample/pdf` CPU reference 和离线预览已经接入。任意图节点保留原生连接，未实现图求值时显式拒绝。
4. **MERL 测量 BRDF 材质族**：100 个原始密集表、发布包身份、官方 Rusinkiewicz 参数化、RGB scale、向量化查表和离线预览已经接入；没有制造不存在的解析 GT 参数。
5. **原生 MaterialX 高分辨率纹理小集**：8 个 Poly Haven CC0 4K 材质已经锁定并下载，完整保留 `.mtlx`、纹理、物理尺寸、颜色空间、切线 normal 和 displacement 语义；均已在 Falcor 中直接呈现，并通过上游 MaterialX float renderer 的共同相机线性 HDR 图像验收。当前 surface-response 验收不移动 displacement 几何，但没有从原始 GT 删除该图或参数。
6. **先完成 LayerStack 主线**：按 `docs/research/experiment_framework.md` 的 P1–P3 依次回答单材质表达力、共享表示与 source compiler 问题。现有多个源材质族先用于检查合同边界，并在 P4 承担工作流稳健性考核，不在建模尚未确定时继续堆新 reference。
7. **evaluator 成形后扩展 sampling 与环境积分**：matched sampler 必须提供同一 proposal 的 `sample/pdf`；环境/面光能力必须近似 evaluator 定义的积分。
8. **再选择 microflake/汽车漆或 BTF 作为表示压力测试**：两者分别增加统计微结构和位置相关方向外观；空间 latent 测试需要先扩展 UV/footprint/LOD 监督合同。
9. **BSSRDF、纤维和完整参与介质最后按独立 renderer capability 接入**：它们需要先扩展共同查询域，不能只向局部 BSDF tile 增加字段。

OpenPBR 在这里是一个源材质族，不是项目统一方法的目标结构；MERL 和纹理 MaterialX 也各自保留原生 GT，不转换成 OpenPBR 后再冒充原始材质。

## pbrt 与 N 层验证决策

现有 pbrt probe 只原生验证“两界面 coated material”，不是任意 `N≥2` 的 pbrt 材质。它仍然有价值，因为它独立覆盖当前随机游走的界面、slab 和多次散射语义。

不再把“增加层数”本身当成验证目标：

- `N=1` 验证各原子公式及 sample/PDF；
- `N=2` 用 pbrt coated diffuse 和 coated conductor 验证两个真实代表分支；
- `N>2` 验证插入无效界面/介质后的退化关系、互易性、能量与有限值、相同随机流下的新旧实现一致性，以及有明确原生定义的实际材质构造；
- 如果以后接入的 MaterialX、薄膜或其他源材质原生包含多层，则用该源材质自己的 reference 验证，而不是人为嵌套 pbrt 只为得到更大的 N。

coated conductor 验证已经完成；“通用 pbrt N 层 probe”仍不进入计划。当前 smoke suite 的总体 mean 相对误差为 0.414%，max 为 2.554%，误差包含两侧 Monte Carlo 噪声。

## 下载状态与后续计划

本轮只锁定已经进入权威 reference 的上游和资产，未进行无目标的全量素材下载；后续候选仍保持按需决策：

| 资源 | 目的 | 固定版本/规模 | 决策 |
|---|---|---|---|
| ASWF OpenPBR | 原生规范、83 个 MaterialX 示例 | tag `v1.1.1`，commit `f8d6d947dfae4c9b599965a86c22826ea7a8dbfb` | 已固定到 `external/OpenPBR` |
| Adobe `openpbr-bsdf` | 可移植 C++/GLSL/CUDA/MSL/Slang reference | commit `9edf806740d2140846d9bef76e4342fc458e2ef5` | 已固定到 `external/openpbr-bsdf` |
| MERL BRDF Database | 100 个测量 BRDF 的原始查表 GT | ZIP 1,253,117,184 bytes，解压约 3.25 GB，Zenodo record `8101681` | 已下载到 `assets/source-materials/merl-brdf/v1` 并校验 MD5/SHA256 |
| MaterialX | 原生图、shader definition、GLSL generation、float renderer 和官方 viewer | tag `v1.39.4`，commit `270b5cf2ae2be24a3b6ef4b0569f1c93038dda1d` | 已固定到 `external/MaterialX`，并完成独立 renderer 与 Falcor 图像验收 |
| Poly Haven MaterialX 材质 | 高分辨率真实纹理源材质 | 8 个 CC0 4K 材质，manifest 总计 578,787,891 bytes | 已下载到 `assets/source-materials/materialx-polyhaven/v1` 并逐文件校验 MD5 |
| MatSynth | 大规模 PBR/SVBRDF 语料 | 当前完整仓库约 433 GB | 不全量下载；需要扩大语料时只取 manifest 固定的子集 |
| refractiveindex.info database | 光谱 IOR/消光参数 | CC0；在 thin-film/真实导体阶段锁定提交 | 暂不下载 |
| pbrt-v4-scenes | 完整场景与材质示例 | 场景级验证资源 | 当前不需要下载 |

下载动作必须先在对应 `references/<package-id>/` 记录 URL、commit/record、许可证、文件清单和目标路径，再由脚本下载并验证哈希。不能手工下载后只保留一个无法追溯的目录。

## 接入一个源材质族的验收条件

一个源材质族进入正式范围时，至少需要：

1. 原生材质定义和资源的版本化保存或可复现导入；
2. 原生可编辑参数的 round-trip，不能丢失或偷偷重参数化；
3. 一个不依赖统一近似方法的 reference 实现；
4. 明确的查询域、颜色/光谱语义、方向约定和适用范围；
5. 在 viewer 或离线工具中正确呈现原始材质；
6. reference 身份、源资源和参数状态可被数据集逐项追溯；
7. 后续统一方法按输出行为评测，不把内部参数是否相似当作正确性标准。

源材质族可以先于统一 neural material backend 接入。backend 尚未覆盖时必须报告 capability 缺失，但这不影响该源材质及其 reference 作为 GT 存在。
