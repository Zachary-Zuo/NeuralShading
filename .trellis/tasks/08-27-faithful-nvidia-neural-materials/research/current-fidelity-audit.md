# NVIDIA 复现忠实度基线审计（修复前）

> 本文件冻结任务启动时的差距证据，路径和“当前”均指修复前 baseline；最终逐项状态以 `correspondence.md` 与 formal artifact 为准。已删除的旧 config 只作为历史实现证据，不是当前入口。

## 审计基准

一手规范为作者版《Real-Time Neural Appearance Models》正文 §4–5、§7–8，以及作者补充材料 §1–2。正文定义完整方法、数据生成、优化和报告规模；补充材料的 Slang 风格 pseudocode 定义可功能复现的 runtime 算术。`D:\01_Workspace\Real-Time Neural Appearance Models` 只作为二手线索，不是 correctness oracle。

本审计把差异分为四类：

- `faithful`：与一手定义相同；
- `source-domain adaptation`：因当前 LayerStack source domain 与论文 textured MaterialX source 不同而必须显式变化；
- `runtime-contract adapter`：为项目线性 `f`、reverse PDF 或错误语义所需，且不改变原方法核心；
- `unfaithful`：未登记地删除、替换或改变论文方法。

## 当前忠实部分

| 项目 | 一手定义 | 当前证据 | 判定 |
| --- | --- | --- | --- |
| latent 维数 | z8 | `models/nvidia_neural_appearance.py:30` | `faithful` |
| learned frame | `z8→12`，无 bias/activation；两个 `(T,B,N)` frame | `nvidia.py:74`、`nvidia_neural_appearance_core.slang:147` | `faithful` |
| frame 算术 | `N=normalize(rawN+(0,0,1))`、`T=normalize(rawT+(1,0,0))`、`B=cross(N,T)` | `nvidia_neural_appearance_core.slang:151-158` | `faithful`，以补充材料 Listing 2 为执行依据 |
| evaluator 输入 | 两个 frame 内的 fixed/query 方向共 12D，再接 z8 | `nvidia_neural_appearance_core.slang:172-199` | `faithful`，项目 `wo/wi` 只是命名互换 |
| evaluator 规模 | 论文正式比较 `2×16`、`2×32`、`3×64` | `nvidia.py:75-82`、model `:35-41` | 当前 `3×64` 是论文正式报告的最大配置，`faithful` |
| evaluator activation/output | hidden ReLU，`exp(raw-3)` | core `:204-207` 与 MLP | `faithful` |
| sampler network | `11→32→32→32→9` | model `:44-51`、core sampler MLP | `faithful` |
| sampler raw/warp | `alphaX/Y,rho,slopeSpec,slopeDiff,wSpec,wDiff` 与 tanh/sinh approximation | `nvidia_proposal.slang:50-70` | 主参数化 `faithful` |
| FP32 master / FP16 pack | 训练 master FP32，加载时量化 FP16 | model 参数与 `nvidia.py:252-258` | 权重量化边界 `faithful` |
| 公共 response adapter | 论文网络返回 `f·cos`，项目 ABI 返回线性 `f` | core `:210-217` | `runtime-contract adapter`；需 exact parity |
| reverse PDF | 项目 scattering ABI 需要 forward/reverse | `nvidia_neural_appearance.slang:107-109` | `runtime-contract adapter`；不是论文 claim |

## 不忠实或尚未证明的部分

### P0：方法身份缺失

1. **完整 encoder 被删除。** 正文 Figure 6 与 §5.1 把 native material parameters → `64×4→8` encoder 作为完整方法和高分辨率可训练性的关键；当前 `create_trainable()` 只接受 `state_count`（`methods/nvidia.py:184-186`），模型直接创建 per-state free latent（model `:30-31`）。这是 `unfaithful`，不是小预算差异。
2. **hierarchical latent texture 与过滤被删除。** 论文按原纹理分辨率建立两张 RGBA FP16 MIP textures，按 UV footprint 计算 LOD，在相邻 MIP 间 stochastic selection，再做 level 内 bilinear fetch。当前 material 只打包一个 32-byte z8（`methods/nvidia.py:134-145,292-303`）。这是 `unfaithful`；uniform LayerStack 只能登记为 1×1 退化的 `source-domain adaptation`。
3. **encoder bootstrap → latent materialization → latent finetune 生命周期不存在。** 当前 config 使用 `evaluator→joint→sampler` 三段（`configs/learning/nvidia-offline-v1.json:5`），既不是论文的同时训练，也没有 encoder drop/latent finetune。这是 `unfaithful`。
4. **formal data domain 不对应论文。** 论文在 UV 空间采样 native layer parameters，按 half/difference vectors 采样方向，并在线求 reference；当前 live source 从固定 LayerStack states 均匀采样两个 hemisphere directions（`batch_sources.py:161-162`），没有 UV、footprint、native parameter features 或 filter level。这是混合的 `unfaithful` 与待设计的 `source-domain adaptation`。
5. **formal training budget 未表达。** 论文为 300k iterations，每步 evaluator/sampler 各 65k 独立 batch；当前 offline 为 25k steps、`batch_size=16`（config `:5-6`），live smoke 为一组 `1×64` directions（live config `:3-5`），live producer 还限制 query-group batch `≤64`（`batch_sources.py:101-104,158-159`）。当前 identity 已正确写成 diagnostic，但不能晋升。
6. **独立 evaluator/sampler batch 在架构重置中回归丢失。** 旧 pipeline 曾有 `auxiliary_training_batch()`；当前 runner 每步只取一次 batch（`runner.py:85`），NVIDIA objective 用同一 values 同时算 evaluator 与 sampler（`methods/nvidia.py:209-222`）。这是 `unfaithful`。
7. **joint optimizer lifecycle 被替换。** 论文同时优化 evaluator 与 sampler以建立共享 latent，KL 只 detach latent；当前顺序还包含 evaluator-only 和 sampler-only phases，并在每个 phase 重建 optimizer（`runner.py:73-80`）。这是 `unfaithful`。
8. **optimizer 与 schedule 不对应。** 一手要求 Adam `β=(0.9,0.999)`、`eps=1e-7`、zero decay、global cosine `1e-3→1e-4`；公共 config 强制 AdamW 且只暴露 weight decay（`training/config.py:62-65`），runner 没有 scheduler（`runner.py:77-81`）。这是 `unfaithful`。
9. **directional mollification 不存在。** 一手要求前 20k iterations `10°→0°` cosine，单 target 256 cone directions；当前 live producer没有 step-aware recipe或 cone sampling。旧离散 HDF5 levels 也不能替代 formal continuous online schedule。
10. **sampler objective 不是已证明的一手公式。** 当前把同一批离散方向上的 learned response 归一化为 target mass，再做 weighted cross-entropy；它对 sampler 参数可等价于一个离散 KL，但没有实现独立 sampler query stream，也没有证明对应论文“samples drawn from learned sampler”的描述。判为 `unfaithful/author-underspecified`，需要 correspondence 和 gradient oracle。
11. **BRDF log loss 使用 `log1p`。** 论文只公开“L1 in log space”，未公开 epsilon/offset；当前 `log1p`（`methods/nvidia.py:210-213`）是冻结过的数值选择，但不能标成作者精确公式。判为 `source/protocol adaptation`，必须进入 recipe identity。
12. **sampler 额外混入 `1/32` cosine safety lobe。** `nvidia_proposal.slang:9,93-100,124-132` 改变了论文 two-lobe PDF。它可作为单独的 robust runtime adaptation，但当前唯一 sampler identity 不能同时声称论文忠实。正式 faithful route 应使用原 two-lobe；项目 null/hemisphere 行为单独验证。
13. **随机数映射改变。** 补充材料 Listing 3/4 用 2D `u`，其 `u.x` 既选 mixture 又 remap 后采样；当前 runtime 用 `sampleNext3D`（`nvidia_neural_appearance.slang:122`），以独立一维选 lobe。分布可相同，但不是 exact functional mapping；需标为 adapter或改回 Listing。
14. **runtime 只量化权重，算术中间值仍为 `float`。** 补充材料明确用 regular FP16 shader math 提供 functional reproduction；当前 MLP arrays/accumulation 为 `float`。这不影响 FP32 training core，却使 runtime precision/performance identity 不忠实。
15. **resume/selection/validation 只是 schema，没有 lifecycle。** `validation.interval/batches` 与 `tail_guard` 被 config 强制，但 runner 从不执行 validation；checkpoint 只保留最后一个 phase optimizer，`run()` 没有 resume入口或 scheduler/query-stream state。无法证明正式训练可恢复。

### P0：最新统一架构尚不能承载论文 runtime

16. **package resource 只写不绑定。** Python manifest/writer 能记录 `material.resources`，但 viewer C++ 只 consume runtime/material blobs（`apps/viewer/ScatteringPackage.cpp:95-121`），latent textures 无法成为通用 binding。
17. **deferred context 丢失已有 UV/gradient。** scattering ABI 已有 `surface.uv/uvDx/uvDy`，visibility G-buffer 也输出 texcoord/gradient；`PackageBackend.slang:24-41` 与 `Prepare.cs.slang:34-35` 却没有填充它们。论文 LOD fetch 在 deferred 不可达。
18. **PT footprint 还没有进入 package backend context。** 现有 reference PT 能为 MaterialX 算 ray-cone UV footprint，但 package backend prepare helper没有等价输入，NVIDIA latent LOD 无法在 PT 与 deferred 共用。
19. **method 仍出现在 viewer CMake。** `apps/viewer/CMakeLists.txt:30-34,46` 显式列出 NVIDIA shader/proposal，违背“新增方法不修改 CMake、package module 是真实加载源”的最新架构合同。应删除具体方法清单并只保留 host ABI/include root。
20. **capability 声明没有变成 neural PT。** `ComparisonSlot` 会把 PT 判定为需要 `sample|pdf`，但当前 `NclsViewer` 根本没有使用 `ComparisonSlot[2]`；唯一的 `ReferencePathTracer.cs.slang` 只调用 source-family reference，package backend 只进入 `Prepare.cs.slang`、`DeferredRenderer.cs.slang` 和 parity probe。UI 显示 package 提供 `sample/pdf` 不等于 renderer 调用了它。当前 neural method 没有真实 PT 证据。
21. **viewer 仍是单 reference / 单 method 分屏。** 当前 C++ 只有一个 `mSelectedProgram`、一个 method approximation texture和一个 reference accumulation，尚未实现稳定文档规定的两个完全对称 slot；package loader也没有产生拥有 typed resources 与独立 pass/state 的 `ScatteringBinding`。这会让 source reference 与 neural method 继续走不同 transport 主线。
22. **MaterialX neural package 在 viewer 被硬拒绝。** `allMaterialsSupportedBy()` 只接受 `ncls.layer-stack-ir@1` 且要求 snapshot hash 等于单个 compiled state，因此即使训练/导出 spatial MaterialX latent，当前 viewer 也无法选择它。source 支持必须改由 package/source identities 与 method adaptation 结果决定，不能硬编码 LayerStack。

### P1：验证证据回归

23. **NVIDIA fidelity tests 在架构重置中被删除。** 当前产品测试只断言 registry 有 NVIDIA（`tests/unit/test_method_definition.py:10-11`）；先前 exact frame/input、autograd warm-order、Falcor PDF/pack parity、sampler oracle tests 已被 commit `7a10a5f` 删除。没有 correspondence regression gate。
24. **稳定 viewer 文档仍描述旧 MethodBundle/03 轨道。** `apps/viewer/README.md` 与当前 `ScatteringPackage@1` 身份不一致，容易再次把旧 artifact 当作当前实现。

## 一手材料未公开、不能伪造精确值的项目

- encoder 阶段切换到 latent finetune 的确切 iteration；正文只说 decoder sufficiently converged；
- “L1 in log space”的具体 epsilon/offset；
- sampler KL 的完整估计式、归一化与采样 estimator；
- 作者训练代码、精确 texture/filter implementation、所有 reference shader assets 与初始化细节；
- 自定义 DXC tensor-core intrinsic 的可直接复用实现。

这些未知项必须在最终 correspondence 中标成 `author-underspecified`，采用预先冻结、可追溯的实现选择；不能把选择写成作者事实，也不能因此回退删除已公开的完整架构。runtime 功能忠实性应以补充材料未优化 FP16 pseudocode 为目标，tensor-core 性能结果只做 observed report。

## 规划结论

当前实现不是“差一点训练到位”，而是只保留了论文 decoder/sampler 的主要数学骨架。真正修复必须同时恢复 encoder/latent lifecycle、formal online recipe、独立双 batch、optimizer/mollification、论文 sampler identity，以及统一 package/viewer 的 generic resource + footprint 路径。只把 25k 改成 300k 会继续训练错误的方法身份。
