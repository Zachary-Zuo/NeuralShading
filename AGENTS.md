# NeuralShading 项目约定

## 根本目标

接入多种保持原生语义的源材质族，用各材质族自己的 reference 产生 GT，再把它们编译成一种统一、固定成本、可直接被实时光照积分的近似表示，并验证它在未见材质和参数状态上的误差。源材质可以来自纯数学模型、可编辑材质图、程序材质、高分辨率纹理、测量外观或其他专用系统。

源材质的原生参数、图结构和资源是 GT 的一部分。除非源材质本身就是层模型，否则不得要求它提供层参数，也不得为了迁就当前 IR 或 approximation backend，先把它反演或改写成层模型后再称为 GT。如果源材质原生可调，项目必须保留这些参数的编辑能力并由对应 reference 正确呈现。完整边界见 `docs/material_scope.md`。

当前已经实现的第一种源材质族是多层界面与均匀 slab，近期工作仍围绕它的最短研究闭环展开：

1. 可信的多层随机游走 reference；
2. 该 reference 生成的方向响应数据；
3. 用逐 tile 直接拟合实验确定固定成本表示的上界；
4. 训练通用编译器；
5. 通过与具体表示无关的散射接口接入 Falcor 延迟着色和 Windows viewer。

当前已实现的基线是“精确计算顶层界面直接反射，再用两个 LTC 瓣拟合剩余响应”。它的方向域相对 L1 误差中位数为 6.73%，第 90 百分位为 31.20%，长尾仍然偏高。任何文档不得再把“闭包表示问题已经解决”当作既定结论。

数据、学习、方法包和 Windows viewer 的基础闭环已经迁到正式架构。当前研究顺序是：先降低当前源材质族中粗糙导体基底和深层栈的表示长尾，确认网络最终需要输出的参数；再实现结构化层组合网络。这个顺序不把 `LayerStackIR` 提升为所有源材质的 GT 表示。不要在输出表示仍可能变化时把某个 backend 的状态布局提升为公共接口。

## 文档与表述

- 本项目维护的 Markdown、实验报告和用户可见说明统一使用中文叙述。
- 统一中文指正文的叙述和逻辑，不要求逐字翻译技术术语。文件名、代码标识符、命令、数学符号，以及 `tile`、`median`、`p90`、`packet`、`closure` 等更便于准确交流的术语可以保留；首次出现不直观的术语时，用中文说明它是什么、为什么存在。
- 不为追求字面上的“全中文”制造生硬译名。判断标准是读者能否迅速明白“它是什么、为什么需要、结论是什么”。
- 写作顺序优先为“它是什么 → 为什么需要 → 当前结论 → 下一步做什么”。避免只堆缩写和指标。
- 物理或工程限制必须说明适用范围，不能写得像普遍定律。例如 v0 为了让 RGB 三通道共用一次自由飞行采样，在有体散射时令三个通道的总消光系数相同；颜色差异仍可由各通道散射反照率表达。这是 v0 的实现约束，不是一般介质都满足的物理性质。
- 新代码、稳定文档和用户入口统一使用 `reference`（对某个源材质族具有权威语义的求值实现）、`direct fit`（逐样本直接拟合）和 backend-specific `ScatteringState`。随机游走是当前层栈材质族的 reference，不是该术语的唯一实现。迁移前报告若从 Git 历史或外部归档中恢复，其原始字段可以保留旧名称，但必须标明它不代表当前接口；这不构成把报告重新持久化到根仓库的理由。

## 根 Git 仓库边界

- 根仓库包含项目自有源码、测试、环境声明、中文稳定文档、版本化验收门槛、资产清单，以及 `references/` 中的 reference registry/package 说明。
- 根仓库不包含 `external/`、`data/`、`build/`、`artifacts/`、`reports/` 和缓存。单次正确性验证、实验报告与运行摘要统一进入 `artifacts/`；第三方 reference 源码固定在 `external/`；原始源材质大资源固定在 `data/source-materials/`；派生响应固定在 `data/reference-responses/`；它们都由 `references/` 中的 package/manifest 追溯。
- 完整规则见 `docs/repository_policy.md`。
- `external/Falcor`、`external/pbrt-v4`、`external/OpenPBR`、`external/openpbr-bsdf`、`external/glm` 和 `external/MaterialX` 是固定提交的独立克隆。当前均为干净工作树，本项目没有修改上游源码。MaterialX viewer 所需的 NanoGUI 及其依赖使用上游 gitlink 固定提交，由获取脚本初始化。
- 若以后确实需要修改上游，先把改动保存为根仓库中的显式补丁和应用脚本，并更新本文件；不得把未说明的修改留在 `external/`。

### Falcor viewer 构建 overlay

- `apps/viewer/` 是根仓库自有源码，通过 `patches/falcor-viewer-overlay.patch` 临时加入 Falcor 的 Samples CMake 树。
- 只使用 `scripts/build_viewer.ps1` 配置和构建 viewer。脚本在应用补丁前验证锁定提交与干净工作树，并在 `finally` 中反向应用补丁；构建结束后 `external/Falcor` 必须重新保持干净。
- overlay 只增加一个 `add_subdirectory()`，不改变 Falcor/Slang 运行逻辑。项目 shader、MethodBundle loader 和应用源码都保存在根仓库。

## 统一 Python 环境

- 项目唯一 Conda 环境名：`neural-shading`。
- 环境声明文件：根目录 `environment.yml`。
- PyTorch 固定为 `2.11.0` 的 CUDA 12.8 wheel，声明在 `requirements-torch-cu128.txt`；它与开发机 CUDA 12.8 toolkit 对齐。
- 所有 Python、pytest 和 pip 命令必须在该环境中执行。非交互命令优先使用：

  ```powershell
  conda run -n neural-shading python <args>
  conda run -n neural-shading python -m pytest <args>
  conda run -n neural-shading python -m pip <args>
  ```

- 不使用 `base`、其他已有 Conda 环境或系统 Python 执行本项目代码。
- 需要导入 Falcor Python 模块时，统一通过 `scripts/run_falcor_python.ps1` 启动；该脚本设置锁定构建的 `PATH`/`PYTHONPATH` 后仍使用 `neural-shading` 环境。
- 新增长期依赖时同步更新 `environment.yml`，确保环境可复现。
- 首次创建环境：

  ```powershell
  conda env create -f environment.yml
  conda run -n neural-shading python -m pip install -r requirements-torch-cu128.txt
  ```

## 锁定的上游源码

- Falcor：`external/Falcor`，tag `8.0`，commit `9dc819c162b2070335c65060436041690b7937f8`。
- Falcor 依赖清单锁定的 Slang：`2024.1.34`。
- pbrt-v4：`external/pbrt-v4`，commit `5f7a606806a4ac7b939131ded9d7a30ebd02416e`。
- ASWF OpenPBR：`external/OpenPBR`，tag `v1.1.1`，commit `f8d6d947dfae4c9b599965a86c22826ea7a8dbfb`。
- Adobe openpbr-bsdf：`external/openpbr-bsdf`，commit `9edf806740d2140846d9bef76e4342fc458e2ef5`。
- GLM：`external/glm`，tag `1.0.1`，commit `0af55ccecd98d4e5a8d1fad7de25ba429d60e863`。
- MaterialX：`external/MaterialX`，tag `v1.39.4`，commit `270b5cf2ae2be24a3b6ef4b0569f1c93038dda1d`；其 NanoGUI gitlink 为 `6452dd6944d2ba5c0c9bc0042a1894f703ce1ace`。

## PowerShell 中文文本编码

- 在 Windows PowerShell 中读取预期为 UTF-8 的文本文件，尤其是包含中文的 Markdown、JSON、配置和源码时，必须显式指定 UTF-8 解码：`Get-Content -LiteralPath <path> -Encoding UTF8`；一次性读取全文时再加 `-Raw`。
- 不因 `[Console]::OutputEncoding` 或 `$OutputEncoding` 已设为 UTF-8 而省略 `Get-Content` 的 `-Encoding UTF8`。
- 如果显式使用 UTF-8 后仍出现乱码，先确认文件实际编码；只有确认是旧式 GBK/ANSI 后才使用 `-Encoding Default`。不得把乱码当作原文继续分析或回写。
