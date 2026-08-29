# OpenPBR 1.1.1 reference package

这是纯数学、原生可编辑的 OpenPBR 1.1.1 源材质族。GT 是 OpenPBR resolved inputs、几何输入和规范定义的 closure；它不转换为 `LayerStackIR`。

固定两个互补上游：ASWF `OpenPBR` 保存规范、MaterialX 定义和 83 个官方示例；Adobe `openpbr-bsdf` 提供可直接执行的完整 BSDF。`materials.json` 逐个锁定官方 `.mtlx` 的大小、SHA256、颜色空间和 authored parameters。项目 adapter 负责原生 JSON/MaterialX round-trip、常量与直接纹理 resolved input、几何 basis、方向约定和 `eval/sample/pdf` 查询。

查询结果保留源材质的线性模型颜色空间；官方示例为 ACEScg，PNG 预览只在显示阶段转为 linear sRGB。当前固定波长模式使用 Adobe reference 的 RGB 代表波长；需要消除厚薄膜或色散的固定 RGB aliasing 时，后续数据采集必须记录随机波长策略。

当前接入范围包括完整 resolved input 参数、官方常量材质的可编辑 round-trip、直接纹理 binding、`eval/sample/pdf` CPU reference 和离线预览。任意 MaterialX `GraphBinding` 会被原样保留，但在没有显式图求值器时不会冒充已求值常量。volume、emission 和 opacity 保留原生字段；非局部体积/BSSRDF 呈现需要独立 renderer capability。

canonical GPU路径通过`ReferenceBackendCapability.open()`创建与其他四族相同的session，使用typed resolved-input buffer和锁定LUT执行`prepare/evaluate/sample/pdf`。`evaluate()`返回线性RGB BSDF `f`而不含cosine；Windows/Linux平台分别由backend选择D3D12/Vulkan。部署只获取manifest锁定的OpenPBR、openpbr-bsdf与GLM源码，不下载source material资产。
