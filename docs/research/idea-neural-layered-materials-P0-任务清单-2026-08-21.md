# P0 任务清单：Slang 参考生成器移植与 Falcor 渲染图

> **历史快照说明（2026-08-23）：** 本文保留 2026-08-21 的工程拆解和当时的 closure 路线，不是当前 neural evaluator 建模计划。现阶段先定义 latent、方向编码、`prepare` shared state 和 `evaluate` MLP，再验证单材质容量、共享 decoder 与 compiler；sampler、环境积分和系统 benchmark 后置。当前顺序见 [`docs/realtime_material_compilation.md`](../realtime_material_compilation.md) 和 [`docs/learning.md`](../learning.md)。

> 日期：2026-08-21
> 上游文档：`idea-neural-layered-materials-siggraph-research-2026-08-21.md`（§6.4 / §7.5 / §8 / §9.1 / §14）、`idea-neural-layered-materials-analysis-2026-08-21.md`
> 周期：日历 6–8 周（Claude 静态写码 + Codex@WinDocker 跑 D3D12 headless 测试 + 用户真机审美验收的三层流程）；纯人力串行约 10 周
> 估时单位：人·日，含调试。标 ◇ 的任务可滑到 P1 初，不在 P0 关键路径上。

> **2026-08-22 执行状态：** 本文保留为 P0 的任务定义，不代表所有项目都已完成。C6 的第一轮表示上界实验已经完成：当前 K2 残差表示的方向域 relative-L1 中位数/第 90 百分位为 6.73%/31.20%，K3 为 5.56%/25.24%。这足以确认“精确顶层界面 + 残差瓣”是有效基础，但不足以确定最终闭包词汇。下一项工作是改进困难材质的残差表示，之后才进入结构化网络训练。实时状态以 `reports/p0_setup.md`、`reports/oracle_ceiling_v0.md` 和 `AGENTS.md` 为准。

## 0. P0 的目标与完成标准

P0 只回答一个问题：**研究闭环的基础设施是否可用**——teacher 可信、数据可生成、四列 render graph 可跑、benchmark 可一键出数。P0 不含任何神经网络训练。

P0 完成的六条判据：

1. **teacher 可信**：N ≤ 8 层随机游走 Slang 模块通过 A4 的五项不变量测试；与 pbrt-v4 CPU 原版（2 层）、PFMC（≥ 3 个配置）交叉一致。
2. **v0-oracle 数据集落盘**：512 family × 1 local state × 4 ω_o × 128 bin，adaptive A/B 半样本 + 计数格式，shard + JSON 元数据可被 PyTorch memmap 读取；4.59 GB 的 `v0-train` 延后到 closure 词汇定稿后。
3. **四列 render graph 跑通**：PathTracer 材质级参考列、deferred ours 列（stub decoder）、解析基线列、完整 PT 语境列；单层 GGX 一致性测试通过，并给出 split-sum 积分误差的数字。
4. **benchmark 一键出数**：固定相机路径 → EXR + FLIP/PSNR + 各 pass GPU 时间 CSV。
5. **估算校准**：4090 上 teacher 吞吐（walks/s）、v0 生成耗时、decode/lighting pass 时间均有实测数，回写上游文档 §7.5。
6. **TESTING.md**：列出全部验证命令与期望输出（Claude 不执行，交 Codex/用户）。
7. **oracle 拟合脚本就绪**（C6）：五种 closure 函数族的可微评估与拟合脚本通过单层 GGX 自检，在 v0 数据子集（≥ 500 栈）上出一版初步天花板表；v0 全量报告作为 P1.0 的第一项交付。

## 1. 前置条件（第 1 周并行完成）

| ID | 内容 | 产出 | 估时 |
|---|---|---|---|
| T0.1 | Windows 4090 机上源码构建 Falcor 8.0（VS2022 + D3D12），跑通 `PathTracer` 示例与 `falcor` Python 模块；记录 Slang 版本 | 可运行的 Falcor + 版本清单 | 1–2 |
| T0.2 | 跑通 Falcor 8.0 Python `ComputePass`、structured buffer 与 PyTorch interop；P0 统一使用 Falcor 依赖清单锁定的 Slang 2024.1.34 | compute smoke test + 版本记录 | 0.5 |
| T0.3 | 构建 pbrt-v4（CPU），准备 `LayeredBxDF` 两层测试场景 | 可跑的 pbrt-v4 + 测试 .pbrt | 0.5 |
| T0.4 ◇ | 构建 PFMC Mitsuba 分支（GPL，放 `teacher/thirdparty_gpl/`，仅验证用，不进发布） | 可跑的 PFMC | 1（受阻则放弃，退化为 pbrt-only 交叉） |
| T0.5 | 仓库骨架 `neural-closure/`（teacher/ datagen/ model/ viewer/ baselines/ ue/），CMake/Python 布局，`.gitignore` 排除数据 shard | 空仓库 + README | 0.5 |

## 2. 工作流 C：共享组件（先行，两条主线都依赖）

| ID | 任务 | 输入 | 产出 | 验收 | 估时 | 依赖 |
|---|---|---|---|---|---|---|
| C1 | **层栈描述 schema**：JSON（人读）+ binary pack（GPU）。字段：版本号；`layers[]`（type ∈ {RoughDielectric, RoughConductor, Diffuse, Sheen}、eta / eta+k、alpha_x/alpha_y、tangent frame、spatial texture refs、mask）；`media[]`（sigma_t rgb、albedo rgb、g、thickness）；base material ID + crop transform。Python dataclass + 校验 + pack/unpack | §4.3、§6.3 层族表 | `schema/stack_v1.json`、`schema/stack.py`、Slang 侧 `LayerStack` struct 定义 | round-trip 测试（JSON→pack→JSON 无损）；Slang struct 与 Python pack 字节布局一致（静态 offset 断言） | 2 | T0.5 |
| C5 ◇ | P2 数据 manifest：选择 MatSynth 空间参数图与来源去重规则；P0 不下载材质全集 | §6.1 | `data/manifests/matsynth_p2.json` | material ID、所需通道、来源、许可证与 split 规则完整 | 0.5 | — |
| C2 ◇ | Belcour 2018 移植为 closure 输出（与 B5 的 packet 接口一致），直接走 deferred 路径作解析基线 | Belcour 公开代码（核对许可证） | `baselines/belcour.slang` | 与 Belcour 原实现同参数输出一致；在 B6 管线中可切换 | 3.5 | C1, B5 |
| C3 ◇ | Principled/OpenPBR 参数拟合基线：PyTorch 对 bin tile 拟合 3 lobe，输出同 packet 布局 | A7 数据 | `baselines/fit_principled.py` | 单层 GGX tile 拟合误差 < 1e-3；输出可被 B5 查表模式加载 | 2 | A7, B5 |
| C4 ◇ | stock UE Substrate 成本场景：同资产（glTF 同时导给 Falcor），全 slab vs 参数混合，记录 bytes/px 与 GPU 时间 | UE 5.x、§8.1 | `ue/substrate_cost/` + CSV | 两种模式各出一组 1080p/1440p 数字 | 2.5 | — |
| **C6** | **表示天花板（oracle）拟合脚本**：不训网络，对每个 (stack, ω_o) 的 teacher bin tile 直接用优化器（Adam + 多起点，log-domain 方向损失）拟合 K 个瓣的参数；支持五种函数族 {GGX-K2, GGX-K3, **LTC-K3**, SG-K8, 学习字典-M16}；LTC 瓣 = 3×3 变换（4–5 自由参数）+ RGB 幅值 + frame；输出每 tile 的最优参数与误差，汇总为"材质族 × 粗糙度档 × 入射角档"的方向域 SMAPE 表，并用缓存 tile × HDRI 矩阵乘算 IBL-FLIP（§7.5 机制，不需渲染器） | 上游 §4.6 (4)、§12.2 决策树、§14 P1.0 | `baselines/oracle_fit.py`、`baselines/closure_families.py`（五族的可微评估，LTC/GGX 部分与 `model/closures.slang` 语义对齐）、`reports/oracle_ceiling_v0.md` | (i) GGX-K1 对自身单瓣、identity LTC 对 Lambert 的自检误差 < 1e-3；标准 LTC 是 GGX 的拟合族而非严格超集；(ii) v0-oracle 全量跑完 ≤ 4 h；(iii) 报告表可直接按 §12.2 决策树得出"LTC-K3 是否达标"的结论；(iv) 最优参数可导出为 B5 查表模式的 closure 纹理，在 Falcor 四列中作为"oracle 列"显示 | 4 | A7, B5（仅导出部分） |

**C6 是 P0 → P1 的交接件**：脚本与自检在 P0 完成，v0 全量报告即 P1.0 的第一项交付。它把"表示不够"与"网络学不到"两类风险在训练之前分离开。

## 3. 工作流 A：Slang 参考生成器（pbrt-v4 `LayeredBxDF` → N 层 GPU 随机游走）

设计原则：`teacher/` **自包含**（自带 GGX / Fresnel / 采样 / RNG 的最小实现），不依赖 Falcor 头文件，保证同一模块被 Falcor Python datagen compute kernel 与 `IMaterialInstance` 两处 `#include`，数据端与渲染端 teacher 同源。

| ID | 任务 | 产出 | 验收 | 估时 | 依赖 |
|---|---|---|---|---|---|
| A1 | 精读 pbrt-v4 `LayeredBxDF`（`f / Sample_f / PDF`）与 Falcor `PBRTCoatedConductorMaterial` / `PBRTCoatedDiffuseMaterial`，写算法笔记：随机游走状态机（当前界面索引、方向、throughput、depth）、NEE 式出射估计、介质 HG 散射、Russian roulette、twoSided、PDF 近似策略、两者实现差异 | `teacher/NOTES.md`（1–2 页） | 笔记能独立指导 A3/A4 实现；列出所有与 pbrt 有意偏离之处 | 2.5 | — |
| A2 | Slang 接口与基础库：`LayerInterfaceType` 枚举、`LayerInterface` / `LayerMedium` / `LayerStack`（MAX_LAYERS=8，寄存器数组）；`ILayerBsdf` 的 `eval/sample/pdf`（RoughDielectric GGX 反射+透射、RoughConductor GGX、Lambert、Charlie sheen）；最小 RNG（PCG）、Fresnel、GGX 采样 | `teacher/layer_types.slang`、`teacher/interfaces.slang`、`teacher/sampling.slang` | 每个界面单独通过白炉测试（≤1）与 pdf 一致性（sample 的 pdf == evalPdf，1e-4） | 3 | C1 |
| A3 | **两层**随机游走 `eval/sample/evalPdf` 移植（对照 Falcor `PBRTCoatedConductor`） | `teacher/layered_walk.slang`（N=2 分支） | Falcor 内置实现只在 RR 不触发（`maxDepth≤4`）时作 <2σ 状态机对齐；完整深度与 pbrt-v4 CPU 同参数方向切片一致（目标相对误差 <2%）。Falcor 8.0 RR reject 的已确认低偏见 `reports/two_layer_xval.md` | 3.5 | A2, T0.3 |
| A4 | **推广到 N ≤ 8 层**：按界面索引上下游走；介质按层；v1 reflection-only（底层不透射）；twoSided 关闭 | `teacher/layered_walk.slang`（通用） | 五项不变量：(i) N=1 退化为解析 GGX/Lambert（相对误差 < 1e-3）；(ii) N=2 与 A3 一致；(iii) 插入"空层"（eta=1、厚度 0、无吸收）结果不变；(iv) 互易性统计检验 f(ω_i,ω_o) ≈ f(ω_o,ω_i)；(v) 白炉 ≤ 1 | 4.5 | A3 |
| A5 | 方差与成本控制：RR 策略、`nSamples` 参数化、bin 内 A/B 半样本 fp32 累加、分支发散整理；4090 吞吐基准 | `teacher/bench_walks.py`（Falcor Python `ComputePass`）+ profile 记录 | 3 层栈 ≥ 10^8 walks/s（未达则记录实测并回写 §7.5 估算） | 2 | A4 |
| A6 ◇ | PFMC 交叉验证：3–5 个配置（coat+conductor、coat+diffuse+absorption、3 层含介质） | `teacher/xval/pfmc_report.md` | 能量差 < 2%、峰位一致；差异有解释 | 2.5 | A4, T0.4 |
| A7 | **Falcor Python datagen kernel + driver**：输入 stack pack、ω_o 列表、bin 参数化（half-vector 或 concentric-map，带开关）、seed；输出每 bin RGB A/B 均值 fp16×6 + 计数 u16（14 B）；driver 负责参数先验采样（log-space roughness/thickness、掠射角 oversampling）、无纹理局部层栈状态、shard 写出 + JSON 元数据（teacher hash、先验版本） | `datagen/gen_tiles.py`、`datagen/kernels/tile_kernel.slang`、`datagen/priors.py`、`docs/tile_format.md` | v0 规格在 4090 ≤ 2 h 生成；PyTorch memmap 可读；随机抽 100 tile 同 seed 重算一致；A/B 两半均值之差的分布与计数给出的方差估计一致 | 5 | A5, C1 |
| A8 | 在线 teacher 验证通道：同 Falcor compute kernel 在 A6000 Linux/Vulkan 跑通 | `datagen/online_teacher.py` | 同 seed 输出统计一致（允许后端浮点差异） | 1.5 | A7 |
| A9 | teacher 文档与 `TESTING.md`（WS-A 部分） | `teacher/README.md`、`TESTING.md` | 所有上述验收命令可复制粘贴 | 1 | A1–A8 |

**WS-A 小计：约 25.5 人·日（含 ◇ 2.5）。关键路径 C1 → A2 → A3 → A4 → A7 → v0 数据。**

## 4. 工作流 B：Falcor 渲染图（四列对比与性能测试）

| ID | 任务 | 产出 | 验收 | 估时 | 依赖 |
|---|---|---|---|---|---|
| B1 | Falcor 构建与 Python 模式确认（含 headless 离屏渲染，供 Codex Docker 跑） | 构建笔记 | Python 脚本无窗口渲染一帧 EXR | 2 | T0.1 |
| B2 | **自定义材质类型 `LayeredStackMaterial`**：C++ `Material` 子类（stack buffer 索引 + 纹理句柄 + 参数块）、Slang `LayeredStackMaterialInstance : IMaterialInstance`（`eval/sample/evalPdf` 调 `teacher/layered_walk.slang`）、注册 `MaterialType`、Python 侧创建与赋值 | `viewer/Falcor/Source/Falcor/Scene/Material/LayeredStackMaterial.{h,cpp,slang}` | PathTracer 渲染球体上的 2 层材质，与内置 `PBRTCoatedConductor` 同参数图像差异在 MC 噪声内（FLIP < 0.02 @ 4k spp） | 5 | A3, C1 |
| B3 | 材质级参考列配置：`PathTracer` `maxSurfaceBounces=0` + NEE + env + `AccumulatePass` | `viewer/graphs/ref_material.py` | 单物体 + env 场景下与完整 PT 的差异仅为缺少互反射（可视化确认 + 数值） | 1 | B2 |
| B4 | GBuffer 通道确认/扩展：`GBufferRaster` 的 posW / normW / tangentW / texC / texGrads / mtlData 是否足够；如需 stack id 与 footprint 椭圆则加 channel | 通道清单 + 必要补丁 | `ClosureDecodePass` 能取到 ω_o、切线系、UV 导数、stack id | 1.5 | B1 |
| B5 | **`ClosureDecodePass`（compute）**：定义 `ClosurePacket` 布局（**LTC 风格瓣 ×3**：每瓣 {LTC 4–5 参数 half, RGB 幅值 half×3, frame oct/quat, IBL 等效粗糙度提示}，记录 bytes/px；同时保留 GGX 类型枚举版布局作为消融下界，两版通过编译开关切换）；v0 两种 stub decoder：(a) 直通——从 stack 参数做 Principled 近似；(b) 查表——读预计算 per-stack closure 纹理。后续换真 MLP 只替换 `decode()` 函数 | `viewer/passes/ClosureDecodePass/`、`docs/closure_packet.md` | packet 布局文档化并打印 bytes/px；两种 stub 可切换 | 3 | B4, C1 |
| B6 | **`DeferredLightingPass`**：逐灯循环（point / directional / spot / rect，≤ 128）+ split-sum IBL + **DXR 1.1 RayQuery 阴影**（复用 Scene TLAS，与 PT 的 shadow ray 同源）；LTC 瓣的 eval / 面光闭式积分 / IBL 等效粗糙度查表，以及 GGX 枚举版（Lambert / aniso GGX / clearcoat GGX / Charlie sheen）作为消融，统一放 `model/closures.slang`，训练端 slangtorch 与 C6 的 `closure_families.py` 共享同一语义 | `viewer/passes/DeferredLightingPass/`、`model/closures.slang` | stub 设为"单层 GGX"时，与 PathTracer(maxSurfaceBounces=0) 在同一单层 GGX 材质上的差异 ≤ split-sum 已知误差，并把该数字写进报告；光源数 1/8/32/128 的 pass 时间曲线 | 5.5 | B5, B3 |
| B7 | IBL 预过滤工具（GGX 预过滤 cubemap + BRDF LUT，一次性 compute）+ **closure MC 积分对照模式**（开关：对 packet 中的 closure 直接 MC 积分 env，用于把"表示误差"与"split-sum 误差"分开） | `viewer/tools/prefilter_env.py`、lighting pass 开关 | MC 对照模式收敛后与 split-sum 之差即 split-sum 误差，按 roughness 分档报告 | 2.5 | B6 |
| B8 | 指标接入：`ErrorMeasurePass` / `FLIPPass`，参考 EXR 落盘；Python 离线 LPIPS | `viewer/graphs/eval.py` | 四列任意两列可出 FLIP/PSNR 图与均值 | 1.5 | B6 |
| B9 | **benchmark 脚本**：Python 驱动 Falcor——加载场景、固定相机路径、切换四列 graph、导出 EXR、读取 Profiler 各 pass GPU 时间、输出 CSV；场景：球 / 茶壶 / 平面（近远尺度）+ 1 个 hero asset；1080p / 1440p | `viewer/bench/run_bench.py`、`viewer/scenes/` | 一条命令产出 EXR + FLIP + CSV；同配置重复运行 GPU 时间抖动 < 5% | 3 | B7, B8 |
| B10 ◇ | 编辑钩子：运行时改 stack 参数 → stack buffer 更新 → 参考列与 ours 列同步变化（为 edit-without-retrain 演示铺路） | Python 示例 | 滑块改 coat 厚度，两列实时响应 | 1 | B9 |

**WS-B 小计：约 26 人·日（含 ◇ 1）。关键路径 B1 → B2 → B6 → B9。**

## 5. 周计划（单人；双人可压到 6 周）

| 周 | WS-A | WS-B | WS-C / 其他 |
|---|---|---|---|
| W1 | A1、A2 | B1 | T0.1–T0.5、C1 |
| W2 | A3 | B2（起） | — |
| W3 | A4 | B2（完）、B3、B4 | — |
| W4 | A5、A7（起） | B5 | 运行 NLBRDF / NA 官方代码（文献并行） |
| W5 | A7（完） | B6 | — |
| W6 | A6 ◇、A8 | B7、B8 | — |
| W7 | A9 | B9 | **C6**（拟合脚本 + 五族评估 + 单层自检）、实测数回写 §7.5 |
| W8 | 缓冲 | B10 ◇ | **C6**（v0 子集初步天花板表、导出 oracle 列到 B5）、C2 ◇、C3 ◇、C4 ◇ |

C6 依赖 A7 的 v0 数据，故排在 W7–W8；其 v0 全量报告（≤ 4 h 计算）顺延为 P1.0 第一周的交付。C3（Principled fit）与 C6 共享 `closure_families.py` 的 GGX 评估，C3 实际是 C6 的 GGX-K3 子集，可合并实现。

P0 关键路径合并：**C1 → A2 → A3 → {A4 → A7 → v0 数据 ; B2 → B6 → B9}**。A3 同时是 B2 的前置（B2 需要能工作的两层 teacher），因此 W2 的 A3 是整个 P0 最早的串行瓶颈。

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| Falcor 8.0 材质系统注册新 `MaterialType` 的改动面比预期大 | B2 超时 | 先以 `PBRTCoatedConductorMaterial` 为模板复制改名，最小化触点；实在不行退化为"hack 现有 PBRTCoated 类扩 N 层" |
| Falcor D3D12 与 Linux/Vulkan 对共享 teacher 模块的编译结果不一致 | A8/B2 | P0 先以 Falcor 8.0 自带 Slang 完成 D3D12 闭环；A8 再做同版本 Vulkan 统计一致性测试 |
| RayQuery 在 Falcor compute pass 中的集成细节 | B6 | Falcor 的 RTXDI/ReSTIR 相关 pass 已有 inline RT 用法可参考；退路是用 PT 的 visibility buffer 作为阴影输入 |
| PFMC 旧版 Mitsuba 依赖编不过 | A6 缺失 | 标 ◇；交叉验证退化为 pbrt-v4 CPU（两层）+ 不变量测试 |
| 随机游走 `evalPdf` 只是近似，PT 中 MIS 引入轻微偏差 | 参考列可信度 | 与 pbrt 同策略（pbrt 接受此偏差）；参考列同时提供"BSDF 采样 only"模式做对照 |
| 4090 teacher 吞吐低于 10^8 walks/s | v0 生成 > 2 h、§7.5 估算偏乐观 | A5 有实测后立即回写；必要时降 v0 的 walks/bin 到 32 |
| Windows D3D12 与 Claude WSL 静态环境割裂 | 调试回路长 | 全部验收写成 Codex 可在 Docker headless 跑的 Python 脚本；Claude 只写码与看日志 |

## 7. P0 交付物清单

- `teacher/`：Slang N 层随机游走模块 + NOTES.md + 单测与交叉验证报告
- `datagen/`：Falcor Python `ComputePass` kernel + driver + tile 格式文档 + v0 数据集（~4.6 GB，本地不入 git）
- `viewer/`：`LayeredStackMaterial`、`ClosureDecodePass`、`DeferredLightingPass`、四列 graph 脚本、benchmark 脚本、closure packet 文档
- `model/closures.slang`：训练端与渲染端共享的四种 lobe 评估
- `schema/`：层栈 schema v1
- `baselines/oracle_fit.py` + `baselines/closure_families.py`：五种 closure 函数族的可微评估与 oracle 拟合脚本，`reports/oracle_ceiling_v0.md` 初版天花板表
- `TESTING.md`：全部验收命令
- 上游文档 §7.5 的实测校准回写
