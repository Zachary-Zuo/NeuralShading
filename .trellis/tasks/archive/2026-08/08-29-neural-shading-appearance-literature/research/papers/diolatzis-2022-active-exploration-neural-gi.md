---
paper_id: "diolatzis-2022-active-exploration-neural-gi"
title: "Active Exploration for Neural Global Illumination of Variable Scenes"
authors: "Stavros Diolatzis; Julien Philip; George Drettakis"
year: "2022"
venue: "ACM Transactions on Graphics 41(5), Article 171"
doi: "10.1145/3522735"
report_status: "evidence-reviewed"
main_source: "https://www-sop.inria.fr/reves/Basilic/2022/DPD22/active-exploration.pdf"
supplemental_status: "available"
official_code_status: "audited"
official_code_commit: "626adbf50703bee8f86ac17543fd05cc9e3e37ec"
author_worker: "/root"
reviewer: "/root/lightformer2024_review"
last_verified: "2026-08-29"
---

# Active Exploration for Neural Global Illumination of Variable Scenes

## 1. 研究对象与报告边界

本文研究的核心不是一种新的局部 BSDF 表示，而是：当一个场景的相机、物体、材质和光源均可变化，如何在昂贵的 path-traced 训练样本中主动选择更能推动神经渲染器学习的场景状态。作者把数据生成与训练交错执行，以 MCMC 在归一化场景参数空间中搜索高价值状态，再用自调节复用和逐步提高训练图像分辨率降低数据生成成本。[P §1、§5–§6]

论文同时给出一个 scene-specific PixelGenerator：输入当前场景参数向量与首交点 G-buffer，输出该像素的全局光照结果。它是场景级 transport surrogate，不是 `evaluate(wo, wi)`、material compiler、BSDF sampler 或跨场景泛化模型。本报告覆盖作者版正文、4 页补充材料和官方 GitLab commit；比较 CNSR、ANF 与 Uniform 时只恢复论文实际使用的 protocol，不把这些 baseline 的方法细节由本文转述扩展成完整报告。

## 2. 来源、版本与 source ledger

| 来源 | locator/版本 | 获取日期 | 本地 hash/commit | 用途与边界 |
|---|---|---|---|---|
| Main paper `P` | [作者项目页](https://repo-sam.inria.fr/fungraph/active-exploration/)；[作者托管 PDF](https://www-sop.inria.fr/reves/Basilic/2022/DPD22/active-exploration.pdf)，18 页 | 2026-08-29 | SHA-256 `66187B8642C41C55E59BEB98BF4C8464BA63E1003EFA61D330A69BFA94824896` | 正式方法、实验、消融和限制；PDF 页脚仍是预出版模板，书目身份以 DOI/项目页为准 |
| Supplemental `S` | [作者项目页 supplemental](https://repo-sam.inria.fr/fungraph/active-exploration/supp/supplemental.pdf)，4 页，submission id 300 | 2026-08-29 | SHA-256 `41D9FBD125F4A5C4C7473D5A31D7F5E744BC6B7D5E5856540FAF35C6FEAC087C` | 网络图、128-feature 变体、MCMC lifespan、复用概率推导与额外 CNSR/GT 结果 |
| Official code/config/data `C` | [GitLab project](https://gitlab.inria.fr/fungraph/active-exploration)；public archive | 2026-08-29 | commit `626adbf50703bee8f86ac17543fd05cc9e3e37ec`；archive SHA-256 `B65B68912B9905CEE32A8627F74396F0BD1BAEB95395E4D64B1E1959FEF0E5F3` | 训练、MCMC、renderer、七场景 XML、七个 checkpoint 与 preview；未执行 Mitsuba 训练/渲染 |
| Author page/talk/correction `A` | 项目页 README、视频入口与 DOI `10.1145/3522735` | 2026-08-29 | 页面无版本号 | 作者身份、公开资产入口、推荐命令；没有单独勘误 |
| NeuralShading evidence `N` | `docs/research/experiment_framework.md`、`docs/research/model_candidates.md`、本任务 `current-nvidia-correspondence.md` | 2026-08-29 | 当前工作树 | 只用于迁移分析，不回填论文事实 |

官方 archive 包含定制 Mitsuba 2 源码和 submodule 声明；本报告只做静态审计与 checkpoint shape 读取。没有使用 Git clone、SSH、token 或登录，也没有把第三方副本当作正文。

## 3. 原论文的问题、假设与贡献边界

[P] 可变场景的显式状态记为归一化向量 `v`，所有可能配置构成超立方体 `D`。均匀采样在维数增加且“重要状态”只占小区域时效率很低。作者的三个方法部件是：

1. 以 `loss × 下一次 Adam 总更新步长的范数` 作为动态 target，用大步/小步 MCMC 主动探索 `D`；
2. 依据新样本与已见样本 loss 的 EMA 差异，自调节“生成新样本/复用旧样本”的 Bernoulli 概率；
3. 保持 patch 为 `32×32`，逐渐提高来源图像分辨率，使 patch 覆盖的视场区域变小，从粗到细学习高频 transport。

[P] 论文的能力声明是单场景训练后可交互改变显式参数并呈现难光路；不是零样本新场景、任意图结构材质、物理可分解 transport、可查询 BSDF，亦不产生与 evaluator 匹配的 `sample()/pdf()`。

## 4. 输入、输出、坐标与 query domain

| 项 | 论文定义 | shape/domain | evidence locator |
|---|---|---|---|
| Scene input | 可变相机、变换、emitter、材质的显式归一化参数向量 `v`；固定属性留在 generator/G-buffer | `v∈[0,1]^D`；实验 `D=6–11`，另有 128-variable Chess 极限测试 | P §4.1、§7.1、§8；Fig.3–4、18 |
| Runtime query | 每个像素的首交点 world position `x`、normal `n`、BSDF reflectance 与 roughness、outgoing direction `wo`，以及整帧共享 `v` | P 所列非 emission G-buffer 合计 13 scalars，其中 position 既走独立预条件支路，又保留在 skip-concat 的 13D 原输入中；`v` 沿 H/W 重复 | P §4.2；S §5 Fig.3；C `PositionalPixelGenerator.forward()` |
| Training exploration state | 场景参数再加 patch 的二维图像位置 | `u∈[0,1]^(D+2)`；16 条 chain | P §5；C `run_markov_chain()`、`dimensions=total_parameters()+2` |
| Coordinates | `x` 按场景 bbox 归一化；normal 由 `[-1,1]` 映到 `[0,1]`；scene variables 已归一化 | world/bbox 与 scene-native ranges | C `VariableRenderer.get_custom_render*()`、`utils.stack_inputs*()` |
| Output quantity | 首交点发射 `Le` 加 hemisphere integral，目标为 outgoing radiance `Lo`；网络学习非直接发射部分 | RGB image-space radiance，训练在 `log1p` domain | P Eq.(1)、§4.2、§6；C `tonemap.py`、preview inverse transform |
| Validity/domain restrictions | 需要能渲染首交点 G-buffer 的单个已训练场景；variable type 主要由少量 float 表示 | 不支持数千维或一般 deformation | P §8；C `docs/variations.md` |

`C` 中方向 buffer 命名为 `wi`，而 `P` 明确写 outgoing direction `wo`；报告保留这个命名冲突，不据变量名改写论文语义。

## 5. Representation、逐层网络与数据流

### 5.1 总数据流

1. 给定归一化场景状态 `v`，定制 Mitsuba 2 修改 camera/shape/emitter/BSDF 参数并 path trace 当前 crop；
2. AOV 同时输出 GT、emission、normal、position、方向、albedo 和 roughness/alpha；GT 与 emission 做 `log(1+x)`；
3. 将 `v` 在所有像素重复；world position 单独通过一层 `3→512` 预条件化；
4. 令 `q=[normal, raw position, wi, albedo, alpha, v]`。首个融合层输入 `concat(position_feature,q)`；之后 8 个公开 `hidden` block 和 output layer 都反复拼接同一个 `q`，而不是再次直接拼接 512D `position_feature`；
5. MLP 输出三通道 log-domain radiance，preview 以 `exp(y)-1` 回到线性域；
6. 训练时 MCMC 根据当前模型的动态 target 决定接受哪种场景/patch 状态，复用器决定本步使用缓存 patch 还是重新渲染。

### 5.2 持久化表示

每个实验场景有一套独立 MLP 权重；没有跨场景 shared decoder、per-material latent texture 或显式 light-field grid。运行时还需要当前 `v` 与逐帧 G-buffer。官方七个 checkpoint 的序列化大小约 `9.83–9.94 MB`；静态读取 state dict 得到约 `2.455–2.483 M` 个参数，差异来自 `D`。这些是 `C` 中公开 checkpoint 的观察，不是正文给出的模型大小口径。

### 5.3 网络逐层配置

| 模块 | 输入 | 层/运算 | activation/normalization | 输出 | shared/per-asset | locator |
|---|---|---|---|---|---|---|
| Position preconditioner | `x:3` | Linear `3→512` | LeakyReLU；negative slope 未显式传参，沿 PyTorch default | 512 | per-scene weights | P §4.2；S §5 Fig.3；C `PositionalPixelGenerator.inner_pos` |
| First fused layer | position feature 512 + 非 emission buffer 13 + `v:D` | Linear `(525+D)→512` | LeakyReLU | 512 | per-scene | C `inner`；checkpoint shapes |
| Repeated fused layers | previous 512 + 同一 13+D | 每层 Linear `(525+D)→512`，每层 skip-concat 原输入 | LeakyReLU | 512 | per-scene | P §4.2；S §5；C `hidden` |
| Output | previous 512 + 13+D | Linear `(525+D)→3` | 无 final activation | RGB log-radiance | per-scene | C `outer` |
| Emission merge | network output 与 emission AOV | 论文：emission 直接传到 output；代码：先对 emission 做 `log1p`，再逐 RGB 通道以 `emission>1.0` 选择纯 emission，否则选择 `output+emission` | 无 | 最终 log-domain RGB | per query | P §4.2；C `VariableRenderer.get_custom_render*()`、`PositionalPixelGenerator.forward()` |

`P/S` 说“8 hidden layers、512 hidden features”。但 `C` 在 `hidden_layers=8` 时还额外创建 `inner_pos` 和 `inner`，再创建 8 个 `hidden` block，最终有 10 个带 LeakyReLU 的 Linear stage 后接 output；checkpoint keys 也包含 `inner_pos`、`inner`、`hidden.0…7`。因此“8”在论文图与公开实现中的计数边界不一致，不能仅凭论文句子猜出精确 topology。

### 5.4 条件化、坐标变换与物理先验

- Position preconditioning 是主要结构先验：空间位置先形成 512D 条件，再与其他 signal 融合；不是 Fourier feature。作者试过 Fourier feature，但训练数据噪声导致 artifact。[P §4.2]
- `v` 是显式、可解释、per-frame global conditioning；没有 encoder 推断 scene latent。[P §4.1]
- G-buffer 把可见首交点属性交给网络，但 visibility、间接传播和难光路仍被隐式储存在 per-scene weights 中；没有物理分解或 reciprocal constraint。
- emission 不由 transport MLP 独立学习，但代码 merge 并非无条件 passthrough：默认 positional 分支对 **已经 `log1p` 的每个 RGB emission 通道**以 `>1.0` 为条件，满足时直接选 emission，否则返回 `prediction+emission`。非 positional `PixelGenerator` 分支使用同一算子但 threshold 为 `0.2`。论文没有披露任一 threshold，也没有解释在 log domain 相加的残差语义。[C `variable_renderer.py`; `positional_pixel_generator.py`; `pixel_generator.py`]

## 6. 数据、GT/reference 与 query/sampling recipe

| 项 | 具体配置 | locator |
|---|---|---|
| Scenes | 修改自 Bitterli rendering resources：Bathroom、Living Room、Bedroom、Veach Door、Sphere Caustic、Spaceship、Veach Egg；相机除 Sphere 外均可变 | P §7.1、Fig.8；C `scenes/` |
| Scene dimensions | Bathroom 8、Living 7、Bedroom 6、Veach Door 6、Spaceship 8、Veach Egg 9；正文称 Sphere Caustic 11 且固定相机 | P §7.1 |
| GT renderer | 定制 Mitsuba 2 `gpu_rgb` forward path tracer；论文声称 integrator-agnostic，但实验只用 forward PT | P §6.2；C training/renderer |
| Training spp | Sphere 200、Living 400、Bedroom 400、Veach Door 600、Bathroom 800、Spaceship 1200、Veach Egg 24000 spp | P Table 2 |
| Patch batch | 16 条 chain 并行渲染 `32×32` crop，论文称一次典型生成约 2.5 s；来源图像初始 `128×128`、FoV 90° | P §5.1、§6.1–6.2 |
| MCMC global/local | large step 概率 `p_LS=0.3`，从整个单位超立方体均匀重采样；其余为 local perturbation | P Eq.(2)；C `large_step_prob=0.3` |
| Adaptive resolution | patch 始终 `32×32`；来源图像每 2000 iterations 增加 4 像素，直到 `600×600` | P §6.2；C README example + epoch update |
| Validation/test | 定量视图从每个 scene/path 选 10 个困难配置；完整随机 split、独立 seed、重复次数未报告 | P §7.2.1、Fig.13；S §1 Fig.1 |
| Online/offline | GT patch 训练中现生成；生成后的 patch 可保存在内存并按旧 loss 加权复用 | P §6.1；C `samples/sample_weights` |

`C` 的七场景 XML 与 checkpoint 维度一致，唯一与论文维数不一致的是 Sphere。Sphere XML 只有 `var_sphere_mat=1` 与 `var_light=2` 显式写出 `num_parameters`，但不能据此把 XML 误读成 3D：定制 Mitsuba 对未显式填写的 BSDF 和 shape/instance 分别默认 `num_parameters=3`。因此两面 variable wall BSDF 各贡献 3、sphere instance 贡献 3，再加显式的 1+2，代码 scene total 是 `D=12`；公开 checkpoint 的 first-fused input `537=512+13+12` 与之吻合。未闭合冲突是 **P 的 11D 与 C/XML/checkpoint 的 12D**，不是 XML、checkpoint、P 三个互异版本。[P §7.1；C `ext/mitsuba2/src/librender/{bsdf,shape}.cpp`、`src/shapes/instance.cpp`、`scenes/sphere-caustic/scene.xml`、checkpoint `inner.0.weight`]

## 7. Loss、optimizer 与训练 lifecycle

| 项 | 正式配置 | locator |
|---|---|---|
| Target transform | GT 与 emission 做 `log1p`；preview `expm1` | C `VariableRenderer`、`tonemap.py`、preview |
| Reconstruction loss | 论文：`L1 + structural dissimilarity`；代码 `2·abs(pred-gt) + (1-SSIM)` 后求均值 | P §6；C `DssimL1Loss` |
| Optimizer | Adam，learning rate `1e-4` | P §4.2；C CLI default |
| Target function | `p(u)=Loss(u)·||ΔAdam(u)||`，其中 `ΔAdam` 是考虑 momentum/RMSProp 后的下一次总更新步 | P §5.1、Fig.6、17；C `compute_adam_grad_norm_reset()` |
| Acceptance | 标准 MH Eq.(3) 因 target 持续变化而太慢；正式方法采用 `p(v)>p(u)` 才接受的 0/1 aggressive policy | P Eq.(3–4)；C `acceptance_policy` 使用 `>=` |
| Reuse bootstrap | P 称前 100 个“samples”全部新生成并存储，但没有消除 sample 是 patch 还是 16-chain generation event 的歧义；C 的 `total_samples` 每次 `generate_samples()` 只加 1，因此 `bootstrap_samples=100` 实际计 100 个 generation events，而非 100 个已存 patch | P §6.1；C `train_dynamic_markov_reuse.py` |
| Reuse probability | `p_s=sigmoid(Loss_exist-Loss_new+β)`；`β=4.6`，两 loss 相等时约 0.99；只用 large-step 样本更新两 EMA | P Eq.(5)、§6.1；S §7；C |
| Reuse selection | 已存 patch 按最后一次记录的 per-patch loss 成比例采样，复用后更新其权重 | P §6.1；C `random.choices(...weights=sample_weights)` |
| Batch/query count | P 定义 16 patches/16 chains；C 初始化 generation batch 为 16，之后每条 rejected chain 额外加入 rejected proposal，所以 fresh batch 是 `16 + rejected_chain_count`，范围 16–32；reuse batch 固定 16 | P §5–6；C `generate_samples()` |
| Replay capacity/accounting | P 未给正式容量。C default `memory_samples=1e5`，但计数器按 generation event 增 1，而 `samples` 按 16–32 个 patch 增长；触发淘汰后每 event 只 `pop(0)` 一个 patch，且没有同步 `pop` `sample_type`。是否有正式 run 达到该分支未报告 | C `train_dynamic_markov_reuse.py` CLI 与 eviction block |
| Steps/epochs | 正文没有总 iterations/epochs；README example 设每 epoch `training_samples=2000`，代码默认 1000 epochs，并在每个 epoch 开头将 resolution `+4`。因此 XML 初始 `128×128` 时，未 resume 的首个 optimizer step 实际使用 `132×132`，不是 128 | P §6.2；C README、epoch loop、scene XML |
| Seed/model selection | code default `seed=0` 并设置 NumPy/Python/Torch；正式 seed、重复训练和 checkpoint selection 未报告 | C CLI；P 未报告 |
| Checkpoint/resume | P 未报告 resume。C resume file 虽保存 model、optimizer、MCMC samplers 和 resolution，loader 却只恢复 model、optimizer、resolution；不恢复 `mcs`、replay patches/weights/types、EMA、sample counters、epoch 或 RNG state | C save/load blocks in `train_dynamic_markov_reuse.py` |
| Hardware/time | 单 RTX 6000，七场景约 5–18 h；Bedroom Fig.9 展示 5/11/18 h | P §7.1、Fig.9 |

代码 default 与论文配置必须分开：`reuse_bias` 默认 `3.0`、`ema_alpha` 默认 `0.95`，README example 使用 `4.6/0.9`，正文只固定 `β=4.6`，未给 EMA coefficient。README 示例调用不存在的 `train_dynamic_markov_reuse_grad_res.py`，archive 实际脚本名为 `train_dynamic_markov_reuse.py`。此外，论文称 small step 是 normal perturbation，代码实际在每维随机选正负并从 `[1/25,1/20]` 对数均匀取幅度、越界时周期回绕；这是实质 proposal 差异。

## 8. Inference、部署与成本

| 项 | 正式配置/测量范围 | locator |
|---|---|---|
| Runtime call path | 每帧先生成全分辨率 G-buffer，再对所有像素并行执行同一 per-scene MLP；`v` 在 H/W 重复 | P §4.1、§6.2；C preview |
| Parameter count | 正文未报告；official checkpoints 静态计数约 `2.455–2.483 M` | C checkpoint audit |
| State bytes | official FP32 checkpoint 文件约 `9.83–9.94 MB`；frame-local `v` broadcast tensor 与 G-buffer 未给总 bytes | C `models/` |
| Texture/feature fetch | 不使用 learned texture fetch；需要 raster/path-traced G-buffer AOV | P §4.2 |
| Precision | 论文未报告；official preview 将 model 和 input 转为 FP16，checkpoint 本身以 FP32 state 序列化 | C preview/model archive |
| Hardware/backend | Python + PyTorch + Mitsuba 2，RTX 3090，`900×900` | P §6.2、§8 |
| Time | unoptimized prototype 4–6 fps，包含约 15 ms Mitsuba G-buffer overhead；没有分离 MLP latency | P §8 |
| Smaller network | 128 hidden features 质量下降但仍可接受，Python prototype 13 fps；supplemental 文本同时写“lower inference speed”和更高 FPS，措辞内部冲突 | S §5 Fig.4 |
| Amortization | 5–18 h per-scene training 不计入 frame time；scene change 在已定义 `v` 内无需重训，新增变量/物体通常需要重新训练或 future fine-tune | P §7.1、§8 |

运行成本不是 `evaluate(wo,wi)` 的固定小 MLP 成本：它包含 scene-specific 约 2.46M 权重、每帧 G-buffer 与把全局向量扩展到每像素的内存流量。论文明确说 5000 variables 会产生 `128×128×5000` tensor，因而设计上不能处理数千变量。[P §8]

## 9. 实验 protocol、baseline、指标与结果

| Experiment | Protocol | Baselines | Metrics | Result | locator |
|---|---|---|---|---|---|
| Table 1 五场景质量 | 七个主场景都有 qualitative result，但 Table 1 只量化 Fig.8 中的五个场景；每场景独立训练 5–18 h | GT | DSSIM/MAPE/MAE/LPIPS | Veach Egg `.0141/.079/.20/.0245`；Sphere `.0012/.031/.01/.0023`；Living `.0068/.048/.04/.0220`；Bathroom `.0029/.033/.03/.0089`；Bedroom `.0149/.074/.05/.0512`；未给 Spaceship/Veach Door 的 Table 1 数值 | P Fig.8、Table 1 |
| Uniform same time | 保留 sample reuse；主 uniform 不使用会伤害它的 multi-res；Fig.11 同训练时间 | Uniform | 同上 | Living：Ours `.0141/.079/.20/.0245`，Uniform `.0241/.162/.28/.0652`；Veach Egg：Ours `.0116/.065/.22/.0477`，Uniform `.0147/.076/.25/.0738` | P §7.2.1、Table 3 |
| 困难视图随时间 | Bedroom/Living/Veach Door，各选 10 个困难 frames，曲线从 2 h 开始 | Uniform | DSSIM、MAPE | 终点 Ours/Uniform：Bedroom `.0097/.0458` vs `.0160/.0791`；Living `.0180/.0892` vs `.0212/.1058`；Veach Door `.0117/.0712` vs `.0135/.0752` | P Fig.13；S Fig.1 |
| CNSR same quality | ArchViz 71D；CNSR 9000 sample points，每点 16 batches×3 observations，`64×64`，1M iterations | Granskog et al. 2020 | qualitative/time | Ours 256 features 且不做 resolution enhancement；Fig.12/S Fig.6 标 Ours 24 h、CNSR 11 days；正文另写 Ours 36 h，存在内部冲突 | P §7.2.2、Fig.12；S Fig.6 |
| CNSR same time | Living/Bedroom/Veach Door，16×3 observations/query，均 `64×64` | CNSR public code | qualitative/MAPE | Living 18 h：Ours `.082`，CNSR `.823`；另给 CNSR 4-day `.655`。作者明确说目标不同，此比较只作效率指示 | P Fig.12、§7.2.2 |
| ANF | 使用相同生成像素 fine-tune；8-frame sequences；inference 8 spp + buffers | pretrained ANF 与 fine-tuned ANF | DSSIM/MAPE/MAE/LPIPS | Spaceship Ours `.0155/.047/.001/.0176`，两 ANF 约 `.046–.048/.067–.068/.023/.066–.069`；Living Ours 全胜；Veach Egg 中 ANF 在 MAPE/LPIPS 更好或相当 | P §7.2.3、Table 4、Fig.15 |
| Hybrid RT | Bedroom 的 specular bounces 由 ray tracing 生成并作为 G-buffer position 等输入，MLP 学其余 shading | full neural | qualitative/time | 约 30 min 可得 acceptable result，但 carpet 高频细节仍需更多训练 | P Fig.10 |
| Variable-count study | Salon 5/7/9/10/25D，逐步加入 furniture、light、roughness、albedo | Uniform | DSSIM difference/qualitative | 收益取决于会制造高频 transport 的变量，不只取决于维数；加 light position 的一维比 10→25 个 albedo 维度更关键 | P §7.3、Fig.14 |

作者没有报告多 seed、误差条、置信区间或正式随机 split；不同 table 也使用不同配置，不能把 Table 1 与 Table 3 的同名 scene 数值视作同一 test aggregate。

## 10. 消融、失败尝试与负结果

| 分类 | 尝试/配置 | 观察 | 作者解释 | 本项目解释（若有） | locator |
|---|---|---|---|---|---|
| `author-negative` | Uniform sampling | 同时间遗漏 sharp shadow/reflection/caustic，最终落入低质量 local minimum | 高价值配置只占 `D` 的小区域 | `[I]` active selection 的收益来自 data coverage，不证明 generator 容量更高 | P §5、§7.2.1、Fig.11–13 |
| `author-negative` | Fourier features | noisy training data 下出现 artifacts | 高频编码放大/拟合训练噪声 | `[I]` 不能把 scene result 直接外推到低方差局部 BSDF query | P §4.2 |
| `ablation-inferior` | 仅用 loss 作为 MCMC target | Living `.0149/.085/.22/.0306`，弱于 full `.0141/.079/.20/.0245`；mirror state 卡住，bottle caustic 欠细节 | loss 高不代表模型仍能改善；Adam step norm 排除不可表示状态 | `[I]` 若迁移到 online reference，不能用 loss-only 难例挖掘冒充原机制 | P Table 7、Fig.17 |
| `ablation-inferior` | 无 position preconditioning | `.0184/.098/.25/.0393`，弱于 full | raw high-frequency wood albedo 干扰阴影/caustic 学习 | `[I]` 是 scene G-buffer factorization 证据，不是 local direction coordinate 证据 | P Table 5、Fig.16 |
| `ablation-inferior` | 无 multi-res | Living `.0201/.135/.23/.0590` vs full `.0141/.079/.20/.0245` | 固定低分辨率限制小反射/阴影 | — | P Table 6 |
| `author-negative` | Uniform + multi-res | DSSIM/LPIPS 略变、MAPE 改善但 MAE `.38` 变坏，作者判定整体更差 | uniform 下 patch 覆盖单点概率进一步下降 | `[I]` curriculum 与 selector 有耦合，不能单独搬一半 | P §6.2、Table 6 |
| `ablation-inferior` | 128 hidden features | 13 fps，但比 512 features 更模糊 | 容量—速度折中 | — | S §5 Fig.4 |
| `author-negative` | ArchViz 上 512 features + resolution enhancement | 产生高频 artifacts；作者改用 256 features、关闭 enhancement | 复杂 71D scene 对 base method 构成挑战 | 比较 CNSR 时方法配置已改变，不是默认 full method | P §7.2.2 |
| `known-limitation` | 学所有路径，包括 pixel-perfect mirror/high-frequency effects | 可交互但不能精确复现 | 网络表达与训练覆盖不足；可用 RT hybrid 或 neural textures | — | P §8 |
| `stress-test-inferior` | Chess 128 variables | 18 h 后 plausible，但缺部分 shadow/highlight | 显式向量广播与容量受限 | — | P Fig.18、§8 |

## 11. Paper ↔ supplemental ↔ code correspondence

| 主题 | Paper | Supplemental | Code/config | 结论/冲突 |
|---|---|---|---|---|
| Architecture | 512 features、8 hidden layers、position first | 图示 position `3→512`、随后 concat，且正文称 total hidden=8 | `inner_pos + inner + 8 hidden + outer`；checkpoint 有 `hidden.0…7` | P/S 的“total 8”与 C 的构造计数边界不一致；不可压缩成“精确 8-layer MLP” |
| G-buffer | `x,n,reflectance,roughness,wo`；emission passthrough | 图中把 position 画作独立 3D 支路，同时标 13D G-buffer concat | explicit tensor order `[emission, normal, position, wi, albedo, alpha]`；去 emission 后 13D 仍含 raw position | `wi` 名称与论文 `wo` 冲突；position 同时进入预条件支路与 raw skip path |
| Emission | 直接传到 output | 未补充 | `log1p` 后逐通道 threshold：positional `>1.0`，non-positional `>0.2`；否则在 log domain 做 `output+emission` | code 不是无条件 passthrough；threshold 与 log-domain residual 语义未由 P/S 解释 |
| Small-step proposal | normally distributed perturbation | 未给实现 | 对数均匀幅度、随机正负、周期 wrap | proposal family 冲突 |
| Acceptance | `>` 的 aggressive 0/1 policy | 未补充 | `>=` | 只在相等 tie 上不同 |
| Reuse | β=4.6、100 samples、large-step EMA、loss-weighted replay | 由 likelihood ratio 推出 alpha=99/β | default β=3.0，README β=4.6；EMA default/example `.95/.9` | 正文 β 与 README 对应；EMA 未由论文裁决 |
| Resolution | 从 `128×128` 开始，每 2000 iterations `+4` 到 600 | 未补充 | 每 epoch 进入 loop 前先 `+4`；README `training_samples=2000` | 间隔依赖 example 才对应 P；代码首步是 132 而非 128 |
| Batch | 16 patches/16 chains | architecture/visual only | fresh batch 为 `16 + rejected_chain_count`，即 16–32；reuse 固定 16 | 实际 work unit 与 P 的固定 16 不一致 |
| Replay capacity | 正式上限未报告 | 未补充 | default `1e5` 以 generation event 计数，patch list 以 16–32 增长；淘汰只移除一个 patch 且遗漏 `sample_type` | cap 的代码记账不一致；未有证据证明正式 run 触发该路径 |
| Resume | 未报告 | 未补充 | save 写入 `mcs`，load 却忽略它；replay/EMA/counters/epoch/RNG 均不保存或不恢复 | official resume 不能续接同一 active-exploration/reuse state |
| Precision/runtime | 4–6 fps、15 ms G-buffer | 128-feature 13 fps | preview model/input FP16；无优化 kernel | paper latency scope不足，不能从 FPS 推单 query cost |
| Assets | 七场景与维数；Sphere 11D | selected views/additional figures | 定制 Mitsuba defaults + Sphere XML 得 12D；checkpoint input shape 也对应 12D | Sphere 是 P=11 与 C=12 的两方冲突；XML 和 checkpoint 彼此一致 |
| Training entry | 方法叙述，无命令 | — | README 指向不存在的 `_grad_res.py` 文件 | 需要人工改为 archive 中实际脚本，公开入口不可原样复现 |

## 12. 作者声明的限制与未报告信息

### 12.1 `known-limitation`

- 未优化 Python inference 只有 4–6 fps，且 15 ms G-buffer 只是总成本的一部分。[P §8]
- 高光、镜面反射、caustic shadow 等高频路径可能无法精确重建；hybrid ray tracing 可缓解，但改变了输入与成本。[P Fig.10、§8]
- 每个变量被归一化到 `[0,1]`，方法不建模变量真实 range 与 transport 难度差异，也不自适应各维 mutation width。[P §8]
- 支持的变量主要是少量 float；parametric deformation 仍是未来工作。[P §8]
- 每场景仍需最高 18 h 训练；新增变量/对象的快速 fine-tune 没有实现。[P §8]
- global vector 在所有像素重复，使数千变量内存不可行；128-variable Chess 已出现质量下降。[P §8]
- 没有跨场景泛化；所有正式质量结果来自 per-scene training。

### 12.2 未报告/材料不可得

- 正式 total optimizer steps、每场景实际新生成/reuse 数量、最终 replay-buffer 大小和 renderer wall-time breakdown；
- 正式 seed 数、split、checkpoint selection、置信区间、metric color space/聚合的全部细节；
- LeakyReLU slope 的论文值、weight initialization、optimizer 其余超参数与 LR schedule；
- MCMC target 的数值稳定策略、target normalization、chain mixing/ESS、burn-in 长度与 acceptance-rate 原始日志；
- official resume 不恢复 MCMC/replay/EMA/counters/epoch/RNG；作者没有给出可恢复同一 selector/reuse trajectory 的替代入口；
- official submodule commit 的完整可重建 manifest、训练命令逐场景配置和七个 checkpoint 与 paper figure 的一一对应；
- 论文 24/36 h ArchViz、11/12D Sphere、8 hidden layers 计数、emission merge 与 small-step proposal 等冲突的官方勘误；
- 模型 MAC/FLOP、frame 内显存峰值、FP16 与 FP32 质量差异、MLP/G-buffer/display 的独立 latency。

## 13. 本项目分析 `[I]`

### 13.1 容量真正放在哪里

运行容量主要在每场景约 2.46M 的 dense MLP 权重，而不是显式 transport cache；当前 scene state `v` 只选择这套已烘焙函数中的一个切片。训练侧另有昂贵容量：path tracer、16-chain selector、动态 Adam-step target 与 replay buffer。论文的主要收益来自把 GT 预算投到“模型仍能改善且难采到”的状态，不是提出了更强的 per-pixel representation。

### 13.2 成功所依赖的假设

1. 场景可用少量、连续且已知 range 的参数显式化；
2. G-buffer 已把 visible geometry/material/view 条件变成与输出高度相关的 per-pixel signal；
3. target 虽随训练变化，仍在 chain 局部停留期间足够平滑，使“只向更高 target 移动”的 aggressive policy 有用；
4. renderer 比网络训练昂贵，因此 replay 能显著节省 wall time；
5. 难光路在 scene-state/patch 空间形成可被 local mutations 找到的 pockets。

### 13.3 可迁移机制与不能迁移的部分

可迁移的是训练 query 选择思想：在 native source state、`wo/wi`、LOD 和 query bucket 上定义候选状态，以“当前误差 × 可实现的优化步影响”主动分配 online reference work。不可直接迁移的是 scene G-buffer、image patch、per-scene radiance MLP、emission merge 和 128→600 image curriculum；它们与 local material ABI 的 query measure 不同。

自调节 reuse 也不能无条件套进当前 formal recipe。NeuralShading 目前要求 GPU-resident online reference query、固定 identity 与可恢复 RNG stream，不持久化 response batch。若引入 replay，必须新建 recipe identity，并明确：是仅复用 query state 后重新求 reference，还是允许有界 GPU response cache；后者会改变 reference work、resume 与随机 reference 方差语义。

### 13.4 与本项目 runtime contract 的关系

Active Exploration 本身是 training-only data policy，训练后可零额外 runtime cost，因此适合作为 `query-recipe diagnostic`，不适合作为 evaluator、compiler、sampler 或 runtime representation。论文的 PixelGenerator 不满足当前小型随机访问 `evaluate(wo,wi)` 合同：它依赖 scene G-buffer、整帧状态广播和 scene-specific 大网络。它应留在独立 scene-transport identity，不能作为给当前 local NVIDIA evaluator 增加隐藏输入的理由。

## 14. 对当前 NVIDIA 复现的影响 `[N/I]`

| 轴 | 分类 | 影响 |
|---|---|---|
| 2024 evaluator topology/runtime | `not-applicable` | Active Exploration 没有为 `20→64→64→64→3`、z8、learned frames 或 bare-`f` ABI 提供替代证据 |
| Online query recipe | 新 candidate 的 `intentional-deviation` | 当前 `functional-f@2` formal config 的 evaluator/sampler 各 65k typed route、300k lifecycle 与独立 RNG 已冻结；active selection 必须用新 identity matched 对照，不能静默改变 faithful baseline |
| Hard-query refinement | 候选机制 | 论文表明 loss-only target 可能黏在高误差但不可改善状态；若当前 hard-query 仅按响应/误差排序，应加入 gradient-influence control，而不是直接宣称 defect |
| Compute feasibility | `budget-adaptation` 必需 | 原方法为每个 proposed/current patch 各做 backward；对 65k query batch 逐样本算 Adam update norm 不可行。迁移需 group/bucket influence 或小 candidate pool，并把选择开销计入训练时间 |
| Replay | `interface-adaptation` | cache GT 会改变 online reference 与 resume 合同；需要单独冻结 cache residence、eviction、stochastic reference SE 和预算口径 |
| Scene transport | `not-applicable` 于 local evaluator | PixelGenerator 依赖 scene buffers，不应进入 `evaluate(wo,wi)`；它只补充第二波 scene-level 方法比较 |

当前没有 `functional-f@2` 300k formal artifact，因此本文只能提出未来 matched 诊断，不用旧 `functional@1` 200k 结果评价 Active Exploration。

## 15. 可证伪的迁移假设 `[I]`

| Hypothesis | Direct evidence | Transfer assumption | Minimum matched control | Frozen axes | Metrics | Runtime class | Falsification condition |
|---|---|---|---|---|---|---|---|
| AE1：在相同 reference-query 总数下，grouped `loss×Adam-step influence` 能提高最难 source/query bucket 的质量 | P Fig.6、13、Table 7 | local material 的难峰也在冻结 source/query space 中形成可搜索 pockets | current uniform/frozen recipe vs grouped active selector；另含 loss-only selector | model、optimizer、steps、总 reference evaluations、seed、train/eval split、checkpoint rule | G1；参数式 family 才测 G2/G2s；peak/tail/energy；bootstrap CI；selection time | training-only | active 与 uniform 的困难 bucket CI 无改善，或改善完全由更多 selection compute/reference work解释 |
| AE2：`loss×influence` 比 loss-only 更不易反复选择当前容量无法改善的 query | P Table 7/Fig.17 的 mirror-vs-caustic 负结果 | local evaluator 也存在高误差但梯度边际收益低的 query | uniform、loss-only、loss×influence 三组；same candidate pool | network、candidate states、reference work、optimizer state、mutation | accepted-state lifespan、重复 bucket 比例、validation improvement per reference query、wall time | training-only diagnostic | influence 组没有降低 stale-state 占比或没有提高单位 reference work 的 validation improvement |
| AE3：有界自调节 reuse 在 iso-wall-time 下优于全 fresh query，且不损独立 evaluation | P Eq.(5)、Table 3；S §7 | 当前 reference 成本足够高，且旧 target 在短期仍有效 | fresh-only vs fixed reuse rate vs EMA self-tuning；同时报告 iso-reference 与 iso-time | query distribution、cache bytes、eviction、stochastic reference sampling、resume policy、seed | G1/G2、time-to-quality、fresh/reuse ratio、reference calls、cache bytes、CI | compiler/training only | 优势只来自减少 reference calls但同 calls 质量不升，或 replay 过拟合使独立 eval 显著变差/破坏可恢复性 |
| AE4：按 native parameter range/difficulty 自适应 mutation 比所有维共用固定幅度更有效 | P §8 明确列为未解决限制；Fig.14 表明变量重要性不等于维数 | source registry 可提供 native range 与类型但不泄漏 test | fixed toroidal mutation vs type/range-aware mutation | target、large-step rate、candidate count、source split、reference work | acceptance/lifespan、coverage、G2、peak/tail、selection time | training-only | 自适应 mutation 不改善 coverage/质量，或只因违反 native domain/改变 prior 获益 |

这些假设是报告输出，不构成执行训练、扩大 seed/预算或修改当前 formal protocol 的授权。

## 16. 证据索引

- `P §4.1–4.2, Eq.(1), Fig.3–4`：显式 scene vector、G-buffer、PixelGenerator 与输出语义。
- `P §5.1, Eq.(2–4), Fig.5–6`：MCMC target、large/small step、标准与 aggressive acceptance、16 chains。
- `P §6.1, Eq.(5)`：100-sample bootstrap、EMA reuse、β=4.6、loss-weighted replay。
- `P §6.2, Fig.7`：`32×32@128→600` adaptive resolution 与 Uniform 失败。
- `P §7.1, Fig.8–10, Table 1–2`：七场景、SPP、训练时间、基本质量与 hybrid RT。
- `P §7.2, Fig.11–15, Table 3–4`：Uniform、CNSR、ANF protocol 与结果。
- `P §7.3, Fig.14,16–17, Table 5–7`：维数、position、resolution、target function 消融。
- `P §8, Fig.18`：runtime、high-frequency、变量 range/type、18 h、数千维/128D 限制。
- `S §5 Fig.3–4`：网络图、LeakyReLU、128-feature/13 FPS；`S §6–7`：state lifespan 与 reuse 推导。
- `C neural_rendering/generators/*.py`：精确公开 topology 与 emission merge。
- `C train_dynamic_markov_reuse.py`、`samplers/mcmc_sampler.py`、`utils.py`：target、acceptance、proposal、reuse、resolution 与 CLI defaults。
- `C data_generation/variable_renderer.py`、`tonemap.py`：AOV packing、normal/position normalization、`log1p/expm1`。
- `C ext/mitsuba2/src/librender/{bsdf,shape}.cpp`、`src/shapes/instance.cpp` 与 Sphere XML/checkpoint：未显式 `num_parameters` 的默认值和 Sphere 12D 资产身份。
- `C models/`、`scenes/`、preview：checkpoint shape/bytes、scene XML、FP16 inference。
- `N docs/research/experiment_framework.md`、`model_candidates.md` 与 `current-nvidia-correspondence.md`：当前 online typed route、泛化轴、runtime ABI 和 identity 边界。

## Evidence review

```text
author_worker: /root
reviewer: /root/lightformer2024_review
reviewed_at: 2026-08-29
sources_rechecked:
  - author paper 18/18 pages, including equations, captions, tables and limitations
  - supplemental 4/4 pages
  - official GitLab archive at commit 626adbf50703bee8f86ac17543fd05cc9e3e37ec
  - official README, training/generator/sampler/renderer/preview code, scenes and checkpoint shapes
findings_closed:
  - corrected the Sphere asset audit: custom Mitsuba defaults make XML and checkpoint both 12D; only paper 11D remains in conflict
  - made the exact C topology explicit and preserved the P/S total-eight versus C construction-count ambiguity
  - narrowed the G-buffer statement to the actual 13D non-emission vector, including the raw-position skip path
  - recorded log-domain per-channel emission thresholds instead of calling C an unconditional passthrough
  - quantified fresh C batches as 16–32 and separated the paper sample unit from the code generation-event counter
  - recorded the replay-cap bookkeeping mismatch and the code's 132 rather than 128 first training resolution
  - recorded that official resume restores weights/optimizer/resolution but not the selector, replay, EMA, counters, epoch or RNG trajectory
  - verified ArchViz 24 h in both figures versus 36 h in main prose as an unresolved paper-internal conflict
remaining_evidence_gaps:
  - exact formal run configs, seeds, checkpoint-to-figure identity and raw logs unavailable
  - hidden-layer count, small-step proposal, emission merge, Sphere 11/12D and ArchViz 24/36 h conflicts have no author correction
  - formal runs may not have reached the replay-cap eviction branch; its empirical impact is unknown
review_status: evidence-reviewed
```

### 完成检查

- [x] main paper 已完整阅读，关键公式/图/表/图注/脚注已视觉核对；
- [x] supplemental/appendix/勘误的可用性已检查；
- [x] official code/config/data 的可用性与 commit 已检查；
- [x] architecture、training、runtime 和主要结果均有 locator；
- [x] 失败尝试与较差消融正确分类；
- [x] paper/code gap 和“未报告”保留；
- [x] `I` 分析晚于事实层，没有改写作者结论；
- [x] NVIDIA 影响引用真实 `N` 证据；
- [x] 假设包含 matched control、部署类别和证伪条件；
- [x] 独立 evidence review 已完成。
