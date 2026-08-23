# NeuralShading 项目约定

## 根本目标

把限定原子层词汇内的可编辑多层材质，编译成固定成本、可直接被实时光照积分的散射实现，并验证它在未见层组合上接近随机游走参考解。

近期工作始终围绕最短研究闭环展开：

1. 可信的多层随机游走参考解；
2. 参考解生成的方向响应数据；
3. 用逐 tile 直接拟合实验确定固定成本表示的上界；
4. 训练通用编译器；
5. 通过与具体表示无关的散射接口接入 Falcor 延迟着色和 Windows viewer。

当前已实现的基线是“精确计算顶层界面直接反射，再用两个 LTC 瓣拟合剩余响应”。它的方向域相对 L1 误差中位数为 6.73%，第 90 百分位为 31.20%，长尾仍然偏高。任何文档不得再把“闭包表示问题已经解决”当作既定结论。

数据、学习、方法包和 Windows viewer 的基础闭环已经迁到正式架构。当前研究顺序是：先降低粗糙导体基底和深层栈的表示长尾，确认网络最终需要输出的参数；再实现结构化层组合网络。不要在输出表示仍可能变化时把某个 backend 的状态布局提升为公共接口。

## 文档与表述

- 本项目维护的 Markdown、实验报告和用户可见说明统一使用中文叙述。
- 统一中文指正文的叙述和逻辑，不要求逐字翻译技术术语。文件名、代码标识符、命令、数学符号，以及 `tile`、`median`、`p90`、`packet`、`closure` 等更便于准确交流的术语可以保留；首次出现不直观的术语时，用中文说明它是什么、为什么存在。
- 不为追求字面上的“全中文”制造生硬译名。判断标准是读者能否迅速明白“它是什么、为什么需要、结论是什么”。
- 写作顺序优先为“它是什么 → 为什么需要 → 当前结论 → 下一步做什么”。避免只堆缩写和指标。
- 物理或工程限制必须说明适用范围，不能写得像普遍定律。例如 v0 为了让 RGB 三通道共用一次自由飞行采样，在有体散射时令三个通道的总消光系数相同；颜色差异仍可由各通道散射反照率表达。这是 v0 的实现约束，不是一般介质都满足的物理性质。
- 新代码、稳定文档和用户入口统一使用 `reference`（随机游走参考解）、`direct fit`（逐样本直接拟合）和 backend-specific `ScatteringState`。迁移前的历史报告允许在文件名或原始实验字段中保留旧名称，但必须标明它不代表当前接口。

## 根 Git 仓库边界

- 根仓库包含项目自有源码、测试、环境声明、中文文档、轻量 Markdown/JSON 实验结果和资产清单。
- 根仓库不包含 `external/`、`data/`、`build/`、`artifacts/`、缓存，以及报告中的 `.npy`、`.npz`、`.pt` 等可重新生成的二进制。
- 完整规则见 `docs/repository_policy.md`。
- `external/Falcor` 和 `external/pbrt-v4` 是固定提交的独立克隆。当前均为干净工作树，本项目没有修改上游源码。
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

## PowerShell 中文文本编码

- 在 Windows PowerShell 中读取预期为 UTF-8 的文本文件，尤其是包含中文的 Markdown、JSON、配置和源码时，必须显式指定 UTF-8 解码：`Get-Content -LiteralPath <path> -Encoding UTF8`；一次性读取全文时再加 `-Raw`。
- 不因 `[Console]::OutputEncoding` 或 `$OutputEncoding` 已设为 UTF-8 而省略 `Get-Content` 的 `-Encoding UTF8`。
- 如果显式使用 UTF-8 后仍出现乱码，先确认文件实际编码；只有确认是旧式 GBK/ANSI 后才使用 `-Encoding Default`。不得把乱码当作原文继续分析或回写。
