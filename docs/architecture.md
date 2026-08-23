# 项目目标架构

## 状态

本文记录已经实现并冻结边界的项目架构。它约束新增实现，但不把当前 K2 基线提升为最终表示，也不冻结任何具体拟合方法的内部参数。

详细合同见：

- `docs/contracts/material_program.md`；
- `docs/contracts/scattering_backend.md`；
- `docs/contracts/reference_dataset.md`；
- `docs/contracts/method_bundle.md`；
- 源材质、reference 与统一表示的边界见 `docs/material_scope.md`；
- viewer 行为见 `docs/viewer_spec.md`；
- 执行顺序见 `docs/migration_plan.md`。

项目的根本目标是：接入多种保持原生语义的源材质族，用各材质族自己的 reference 产生 GT，再把它们编译为一种统一、固定成本、可由实时光照系统使用的近似表示，并验证它在未见材质和参数状态上的误差。源材质不需要先被分解成层参数；当前多层随机游走只是第一种已实现的源材质族。

## 统一术语

- **源材质族**（source material family）：共享一种原生材质语义和求值规则的一组材质。它可以由公式、图、程序、纹理、测量表、微几何或其他资源定义。
- **reference**：对某个源材质族具有权威语义的求值实现，用来生成监督和提供 viewer 的 GT 图像。随机游走、解析公式、原生材质图、纹理求值和测量数据查表都可以分别成为 reference；它们只统一查询与输出合同，不统一内部表示。新代码和文档不再使用 `teacher` 作为名称。
- **逐样本直接拟合**（direct fit）：不经过通用预测网络，直接对一个 `(材质, 观察方向)` 优化表示参数，用来测量某种表示的上界。新文档不再把它称为 `oracle`，旧路径只在迁移说明中保留。
- **拟合后端**（approximation backend）：K2 LTC、其他解析 closure、latent 表示、参数预测网络或直接神经求值器等可替换实现。
- **材质程序**（`MaterialProgram`）：面向编辑、存储和交换的可扩展有类型材质图。
- **散射状态**（scattering state）：拟合后端在一个着色点和观察方向上准备出的不透明运行时状态。renderer 不读取其字段。
- **方法包**（`MethodBundle`）：可由评测程序和 viewer 加载的完整方法交付物，包含材质编译器、拟合后端、权重、shader、合同版本和成本信息。

## 三个业务块和公共核心

项目分成三个业务块，外加一层不包含独立工作流的公共核心：

1. **源材质接入与数据采集**：保存或导入源材质资产和原生参数状态，通过该材质族的 reference 获得方向响应或图像 GT，写入可恢复、可验证的数据集。
2. **Python 学习与评测**：逐样本直接拟合、通用编译器训练、测试集评测、TensorBoard、方法导出。
3. **Windows 材质查看器**：在同一场景、相机和灯光下比较左侧累积参考图像与右侧可切换方法。
4. **公共核心**：只保存三块共同遵守的 `MaterialProgram`、散射语义、数据合同、方法包合同和共用 shader，不形成第四条业务链路。

依赖方向固定为：

```text
公共核心 ← 数据采集
公共核心 ← Python 学习与评测
公共核心 ← Windows viewer

源材质资产 → 族专属 reference → 数据集 → Python 学习与评测 → MethodBundle → Windows viewer
MaterialProgram / 原生资源 → 族专属 reference ────────────────────────→ Windows viewer
```

viewer 不依赖训练代码或 PyTorch 运行时。训练侧只能通过 `MethodBundle` 向 viewer 交付方法。

## 目标目录

```text
src/ncls/
  core/
    material/              MaterialProgram、节点注册表和规范化 IR
    representations/       可被学习与导出复用的表示定义
    scattering/            公共散射语义与 backend descriptor
  data/
    generator.py           材质先验、采集调度和可恢复分片 writer
    reference.py           Falcor reference driver
    dataset.py             manifest、reader 和完整性验证
  references/              reference registry/package 解析与身份验证
  source_materials/        OpenPBR、MERL、MaterialX 原生源材质 adapter
  bundle/                  MethodBundle manifest、导出和 loader
  learning/
    direct_fit/            表示上界实验
    models/                通用材质编译器
    training/              训练循环、TensorBoard、checkpoint
    evaluation/            held-out 测试和图像/方向指标
    export/                MethodBundle 导出

shaders/ncls/
  contracts/               共用语义结构和生成的 ABI
  reference/               当前随机游走及后续族专属 reference 共用实现
  data/                    数据生成 compute 入口
  backends/                可替换的拟合后端

apps/viewer/                Windows/Falcor 查看器
references/                 reference package registry、身份和轻量 adapter
configs/                    数据、拟合、训练、评测配置
tests/
  unit/
  integration/
  gpu/
docs/
scripts/
```

以下目录只保存本机生成物并全部忽略：

```text
artifacts/
  runs/
  exports/
  captures/
  benchmarks/
```

参考数据使用单独被忽略的 `data/`；`artifacts/` 只放学习、导出、viewer 和 benchmark 派生产物。

Git 中只保存源码、合同、配置、测试、人工整理的中文结论和轻量 JSON 指标。

## 材质和运行时的分层

### 作者层：MaterialProgram

`MaterialProgram` 是公共输入。它能表达有类型节点、资源和预留的表面/介质/发光/透明度/位移输出。它必须保存源材质原生存在的可编辑参数、图结构和资源引用，不能为了适配当前近似方法而要求所有源材质提供层参数。第一阶段只实现可规范化为线性反射层栈的子集，但以后可以增加其他源材质族和规范化路径而不改变顶层文件格式。

### 规范化层：内部 IR

数据采集和拟合后端不直接解释任意作者图。公共核心先验证节点和资源，再按源材质族规范化为带版本的内部 IR。不同材质族可以使用不同 IR；规范化只消除无关序列化差异，不能改变原生物理语义。第一阶段的主要 IR 是 `LayerStackIR`，它既不是公共材质格式，也不是其他 GT 必须采用的中间表示。

### Reference 层：族专属权威求值

每种源材质族可以拥有独立 reference 和资源求值路径。reference 只需遵守共同的方向、测度、颜色/光谱、随机性和统计输出合同，不要求共享 shader、参数布局或输运算法。同一材质族也可以保留多个 reference 做交叉验证。

reference 必须直接求值源材质原生语义，不能先经过项目要研究的统一近似表示。源材质族可以在尚无 approximation backend 支持时先接入；此时 backend 明确返回 capability 缺失。

### 实现层：ApproximationBackend

每种拟合方法实现同一散射合同，但可以拥有不同的：

- 材质静态编译结果；
- 每个着色点/观察方向的状态布局；
- 解析公式或 neural decoder；
- shader specialization；
- 面光、环境光等专用积分能力；
- GPU 内存和时间成本。

renderer 只使用不透明状态和统一散射操作，不读取 K2、lobe 数量或 latent 维数。

## Deferred 数据流

目标 deferred 路径分成清晰的阶段：

```text
主可见性 / GBuffer
  → SurfaceInteraction + MaterialInstanceID
  → 材质资源求值与 backend.prepare()
  → method-specific opaque scattering state
  → deferred lighting：evaluate / sample / pdf / optional integration
  → 线性 HDR 合成
  → 统一曝光和 tone mapping
```

`prepare()` 是稳定扩展点：解析 K2 可以直接构造解析状态；参数网络可以在这里输出物理参数；小 neural decoder 可以输出 latent，随后在 `evaluate()` 中执行；直接 neural BSDF 也可以把网络完全放在 `evaluate/sample/pdf` 内部。

## 第一阶段范围

以下范围只描述当前已实现的 `LayerStackIR + random-walk-reference@1` 源材质族，不定义项目最终只支持这些 GT：

- 局部表面反射；不渲染穿过整个物体的透射路径。
- 1–8 个界面，底层必须是当前支持的不透明基底。
- 常量材质参数；`MaterialProgram` 合同预留纹理和其他参数来源。
- RGB 线性工作流；材质程序显式记录 `color_model`。
- 支持各向异性，切线坐标系是标准 `SurfaceInteraction` 的一部分。
- 该材质族的随机游走 reference、数据采集和 viewer 左侧复用同一套 reference shader。
- K2 只作为 `legacy-ltc-k2` 后端和对照，不是接口默认值。

## 非局部能力

单一局部 BSDF 接口不能正确覆盖 BSSRDF、完整参与介质传输、真实位移、偏振或全光谱输运。`MaterialProgram` 为这些能力保留输出槽和版本协商；以后通过独立 renderer 阶段和 capability 接入，而不是破坏表面散射合同。

## 配置和实验身份

所有数据、直接拟合、训练和评测都由显式配置启动。每次运行必须保存解析后的配置、Git 提交、参考实现哈希、合同版本、随机种子、依赖版本和输入产物 ID。命令行覆盖只允许修改配置中已声明的字段。

测试集只由独立评测命令读取。训练循环只能使用 train/validation，不能根据 test 结果选择 checkpoint 或超参数。
