# NVIDIA Neural Appearance functional reproduction correspondence

## 身份边界

执行规范是一手论文《Real-Time Neural Appearance Models》正文 §4–5 与补充材料 Listings 1–4。后续作者公开仓库 [NVlabs/neuralappearance](https://github.com/NVlabs/neuralappearance)（审计 commit `305b4b9c12e679398c487603dd8245c3f348526c`）只用于补足 2024 论文没有公开的 estimator/stage 选择；这些项仍标为 `author-underspecified`，不改写成论文事实。

当前实现身份：

- `correspondence_id = nvidia-rta2024-functional@1`；
- formal `recipe_id = nvidia-rta2024-materialx-formal-300k-stage100k@1`；
- MaterialX `source_adaptation_id = materialx-standard-surface-spatial@1`；
- LayerStack `source_adaptation_id = layer-stack-uniform-1x1@1`；
- smoke/profile/formal 使用不同 recipe/config/checkpoint identity；实现满足公开方法合同后可称为 functional reproduction，但本次用户冻结的 200k 经验结果不得标成“300k formal protocol完成”。

## 逐项对应

| 一手定义 | 当前实现符号 | 回归证据 | 分类与结论 |
|---|---|---|---|
| 正文 §4/Fig. 6：native parameters 经 `K→64→64→64→64→8` encoder | `NvidiaNeuralAppearanceModel.encode`；MaterialX K38、LayerStack K217 native layout | `test_nvidia_encoder_materialization_and_checkpoint_state_roundtrip` | `faithful`；修复了 per-state free latent 替代 encoder 的旧偏差 |
| 正文 §4、补充 Listing 1：hierarchical z8，两张 RGBA texture，footprint LOD、相邻 mip stochastic selection、level 内 bilinear | `latent_texels` + mip offset；`nclsNvidiaNeuralFetchLatent`；`latent0.dds/latent1.dds` | `test_nvidia_material_compiler_emits_two_full_rgba16f_mip_chains`、typed DDS tamper tests、Falcor package parity | 语义 `faithful`；训练期扁平 parameter 是不改变 texel/filter 的存储 adapter |
| 补充 Listing 2：z8 投影两个 learned frame；normal/tangent 加 canonical axis 后 normalize，bitangent=`cross(N,T)` | `_learned_frames`；`nclsNvidiaNeuralLearnedFrame` | `test_fp32_torch_training_core_matches_slang_functional_oracle` | `faithful`；Torch FP32 与 Slang oracle固定容差一致 |
| 补充 Listing 2：两个 frame 内 fixed/query direction，各 6D，再接 z8 | `_evaluation_input`；`nclsNvidiaNeuralEvaluationInput` | 同上 exact-vector parity | `faithful`；项目 `wo/wi` 命名由注释明确映射，不改变排列 |
| 正文 §4：正式最大 evaluator 为 `20→64→64→64→3`，hidden ReLU，输出 `exp(raw-3)` | `evaluate_w0/w1/w2/out`；`response`；Slang evaluator MLP | formal static validator、Torch↔Slang parity | `faithful` |
| 正文 §4/补充 Listing 3：sampler `11→32→32→32→9` | `sampler_raw`；Slang sampler MLP | formal static validator、joint gradient test | `faithful` |
| 补充 Listings 3–4：9 raw值解码 anisotropic GGX、correlation、spec/diffuse slope和两权重；tanh/sinh approximation | `_sampler_pdf_from_raw`；`nclsDecodeNvidiaProposalUnchecked` | Torch↔Slang PDF parity | `faithful` |
| 补充 Listings 3–4：仅两个 learned lobes，使用同一 `float2 u`；`u.x` 选择并在分量内 remap | `nclsSampleNvidiaProposal` | `test_formal_sampler_shader_has_only_two_lobes_and_2d_random`、Falcor sample/pdf parity | `faithful`；已删除旧 `1/32` cosine safety lobe和独立第三随机数 |
| 正文 §5.1：先直连 encoder训练，随后 materialize全部 mip texel，移除/冻结 encoder并 finetune latent | `configure_lifecycle`、`materialize`、`TrainingRunner.run` | lifecycle unit、MaterialX step1→step2真实磁盘 resume smoke | 结构 `faithful`；精确切换点属下述未公开选择 |
| 正文 §5.1：GPU online生成数据；uniform UV、half/difference directions、指数 mip、Gaussian footprint且空间样本随面积增加 | `MaterialXLiveReferenceBatchSource`、`MaterialXNativeFeaturePyramid.sample_torch`、`MaterialXGpuQueryRuntime` | `test_materialx_training_geometry`、live Falcor GPU tests、formal preflight | 公开部分 `faithful`；MaterialX standard_surface及 LEAN coarse feature是 `source-domain adaptation`；rate=1/cap=64是 `author-underspecified` recipe choice |
| 正文 §5.1/补充 §1：前20k对 outgoing `ωo` 做 `10°→0°` cosine mollification，每target 256 cone samples | `_mollified_views`、`_reference_target`、formal config | formal static validator；preflight step1 `sample_count=102,162,944` | `faithful`；flat tiles只分 dispatch，不改 logical estimator |
| 正文 §5：300k iterations；每步 evaluator/sampler 两个独立 65k batch | `TrainingRouteRequest`、`TrainingRunner._batches`、formal config | static reject matrix、stream identity/resume tests、formal metrics `work_units=130000×step` | `faithful` |
| 正文 §5：Adam `(0.9,0.999)`, eps `1e-7`, zero decay；cosine `1e-3→1e-4` | `_optimizer_and_scheduler` | config validator、interrupted-vs-uninterrupted lifecycle test | `faithful`；CUDA fused Adam只合并kernel，不改变参数 |
| 正文 §5.2：evaluator/sampler simultaneous optimization；sampler KL target是当前 learned BRDF，latent detached | `NvidiaMethodDefinition.training_objective`、`sampler_forward_kl_score` | joint gradient ownership、finite joint gradients | 公开 detach/target语义 `faithful` |
| 论文只写 forward KL，未给完整 estimator | 从当前 learned proposal取样；`-stopgrad(L(f·cos)/p)·log p` | 后续作者 `training/train.slang` 对照；score-path GPU smoke | `author-underspecified`，采用 author-code-informed `learned-sampler-forward-kl-score@1`；没有 batch self-normalization |
| 论文只写 log-space L1，未给 offset | `L1(log1p(response), log1p(target))` | loss identity/static validator | `author-underspecified`，冻结为 `log1p-l1@1` |
| 论文未公开 encoder→latent 精确 step | formal `materialization_step=100000` | recipe/static validator、checkpoint lifecycle | `author-underspecified`；取后续作者公开默认 BSDF encoder phase，明确不是 2024 隐藏事实 |
| 补充 pseudocode 使用 FP16 runtime weights/intermediate；训练 master FP32 | FP32 Torch model；`nvidia_neural_appearance_fp16.slang` 的 half input/weights/bias/accumulator/activation/output；`package_validation()` 的独立 CPU oracle | Torch FP32↔Slang训练 oracle；`test_nvidia_deployment_shader_uses_regular_fp16_mlp_path`；pre-formal D3D12 package parity | regular functional path `faithful`；容差在 formal 前冻结，见 `fp16-runtime-parity-calibration.md`；不声称复现未公开 tensor-core intrinsic |
| 论文 evaluator返回 cosine-weighted response；项目 ABI要求线性 `f` | `nclsNvidiaNeuralResponseToBareF` | scattering contract exact-vector tests | `runtime-contract adapter`；只除一次 `max(wi.z,minCos)` |
| 项目 renderer要求 reverse PDF、null/error与固定 state ABI | backend `prepare/evaluate/sample/pdf`；reverse重新以交换方向 prepare | package/Falcor parity与 viewer PT | `runtime-contract adapter`；不改变 forward proposal |
| 作者训练资产、shader graphs与论文图像未公开 | formal 使用 registry内 `american_walnut_veneer` MaterialX snapshot；LayerStack另做1×1 smoke | snapshot/config/source identity | `source-domain adaptation`；只对实际 snapshot/reference负责，不声称逐图像复刻作者结果 |
| neural方法必须进入真实 renderer transport | `PackagePathTracer.cs.slang`、`DeferredRenderer.cs.slang`、`ComparisonSlot[2]` | capture v4：同 package 分别为 PT/deferred且 status ready | 当前架构 `runtime-contract adapter`；PT命中实际调用 package `prepare/sample/pdf/evaluate` |

## 修复结论

修复前实现只忠实保留 decoder/sampler的大部分算术；encoder、hierarchy、lifecycle、双65k online route、published optimizer/mollification、exact two-lobe sampling、typed latent binding和 neural PT都缺失或被预算适配替代。当前代码已经逐项补齐公开合同，并把 stage、log offset、KL estimator、spatial cap、source asset与常规 FP16 path等无法逐 bit确认的项目放入版本化 identity。

本次经验结果按用户决定冻结在 step 200k：checkpoint `ee3e6fb3…12fe`，package `6950aeb2…02073`。64×4096 方向诊断中，packed-FP16 runtime 对 MaterialX reference 的方向 normalized L1 median/p95 为 `0.02548/0.07018`，能量相对误差 median/p95 为 `0.01157/0.04272`；packed runtime 对 FP32 master 的 log1p L1 mean 为 `4.49e-5`。同 package 的 viewer PT/deferred slot 与 source-reference/neural PT slot 都为 `ready`。完整报告位于 `artifacts/nvidia-faithful/materialx-recorded-200k/formal-report.json`；它明确排除 step 200k 以后日志，不声称完成原 300k protocol。
