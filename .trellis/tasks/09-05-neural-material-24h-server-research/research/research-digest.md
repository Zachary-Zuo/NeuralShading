# 服务器研究的预读摘要

## 如何使用

启动时读本文、PRD 与实验卡；每次事件只补读当前假设相关的详细报告。本文把已有深入研究转成此次可试的机制，未声称重新完成所有论文复现。详细报告中的论文正文、补充材料、代码与作者通信有不同证据等级，无法闭合的事实仍保持未闭合。

总入口为 [论文目录](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/catalog.md) 和 [可复现实验假设](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/implications/reproducible-hypotheses.md)。前者保留逐篇 primary source 链接与缺口；后者提供 H-O/H-R/H-F/H-Q/H-D/H-C 等机制映射。本次应优先 E00/E01/E02，再按证据取用后面的机制。

## 先前项目证据

| 来源 | 有用事实 | 本次行动与不能推出的结论 |
|---|---|---|
| [前继诊断](../../archive/2026-09/09-05-neural-material-spatial-optimization-research/research/evidence-and-diagnosis.md) 与 [模型审计](../../archive/2026-09/09-05-neural-material-spatial-optimization-research/research/model-design-audit.md) | 多 UV/native 值、summary 信息瓶颈、训练/cook/读取错配是可定位风险；C1–C6 有具体修复合同 | E00 先验证 native，E02 分段跟踪；不能把旧性能当当前 raw 质量证据 |
| [当前实施验证](../../archive/2026-09/09-05-neural-material-spatial-optimization-research/research/implementation-validation.md) | raw 链路、共享资源、ABI 和 Windows step 2 跑通，实际青铜 12 reads、176 B state、约 181.33 MiB latent | 服务器测 Linux 和真实训练成本；summary 未实现、全 D0/D1 未执行，图像不是滤波质量或 Pareto 证明 |
| [历史结论](../../09-04-metal-neural-budgeted-redesign/research/final-conclusions.md)、[特征 probes](../../09-04-metal-neural-budgeted-redesign/research/characteristic-probes.md) 与 [模型综合](../../09-04-metal-neural-budgeted-redesign/research/model-redesign-synthesis.md) | role/center/dual-local 改动未稳定恢复高频；fixed batch/提高 LR 也没隔离根因；输入和 native UV 后来又发现问题 | 保留为 E02 的检查线索，全部按新 raw 合同 fresh rerun。日志 `mean(p²)` 是均方，不是振幅；很小的正值不能写成精确零 |
| [纹理共享分析](../../archive/2026-08/08-30-vmaterial-metal-neural-system/research/texture-encoder-sharing.md) | 692 exports 只有 52 个 texture-set；42 个 set 只对应一个 module，若按 module 切分易泄漏 | E07 按 texture-set 分组，原生参数编辑另作一轴；不能把 module 数叫独立空间材质数量 |
| [五卡监管记录](../../09-04-metal-neural-budgeted-redesign/research/ddp5-supervision.md) | 历史 GPU 5–9 为 A6000，曾有空 review、旧入口/配置及运行生命周期问题 | 本次实机重查 UUID/接口；空决策失败、deadline 独立于模型、先单卡并行，不复制旧 DDP5 命令 |

## 优先可迁移的论文机制

### Taming Optimization Variance（2026）→ E04/E08/E10

[深入报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/bitterli-2026-taming-optimization-variance.md) 分开研究 target remap、稳定坐标、LeakySmeLU 和 successive training。可直接形成单轴消融：log 对 cube-root、坐标、activation；初始化方差明显时再试少数实例筛选。

原配方 K=64→16→4→1、batch=1k→4k→16k→64k，K×B=65,536，100k steps；共享查询使总网络求值约 6.5536B 而独立 reference queries 约 2.176B，和单实例 query 成本不等同。24h 内的小规模改编必须独立命名，不能称忠实复现或同 query 预算。报告保留 Table 1/正文 trial 数等矛盾；不要替论文补齐未确认数字。

### Real-Time Neural Appearance Models（2024）→ E01/E05/E07 与阶段部署

[深入报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/zeltner-2024-real-time-neural-appearance-models.md) 给出原生属性的 pointwise encoder、LEAN/coarse mip、latent bake/真实读取 fine-tune、view-conditioned prepare 复用和 proposal。对本次最有用的是训练输入到部署读取的对应以及 spatial code 与方向 decoder 的分工。

它不证明 raw CNN 必需、8 通道足够或任意新源材质零样本成功；本项目采用 bare 线性 f，论文响应量若含 cosine 必须显式换测度。此次先稳定 evaluator，proposal/sampler 的论文能力不能视作当前方法已具备。

### NeuMIP（2021）→ E02/E05

[深入报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/kuznetsov-2021-neumip.md) 的核心是空间 latent 与独立层级、连续 LOD 读取，以及自由 latent 训练中的渐进 Gaussian blur。小 decoder 的效果依赖存入空间表示的信息。

可试 derived/independent mip、真实量化读出与渐进过滤；不能由该论文推出当前 encoder 能编译未见纹理。它的 per-material free code、offset 和几何边界不是任意 native MDL 的合法重写；增加 offset 必须另证 source 语义，当前不优先。

### Neural Graphics Texture Compression Supporting Random Access（ECCV 2024）→ E01/E05/E08

前继 [诊断报告](../../archive/2026-09/09-05-neural-material-spatial-optimization-research/research/evidence-and-diagnosis.md) 已分析 raw CNN、coarse grid 与随机访问压缩路径；[论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05476.pdf)。启发是让学习层看到原始空间结构、预算化空间存储与读取。纹理值压缩成绩不等于角度响应拟合或未见资产 compiler 成绩。

### Filtering After Shading（2024）与 Neural Prefiltering（2023）→ E00/E05

[随机纹理过滤研究](https://research.nvidia.com/publication/2024-05_filtering-after-shading-stochastic-texture-filtering) 与 [预过滤深入报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/weier-2023-neural-prefiltering-lod.md) 提醒：先平均参数再非线性着色通常不等于响应平均，相关变量也不能任意独立过滤。E00 使用 reference response footprint witness；E05 的随机 latent 读取只是一种候选，不能仅凭随机性宣称无偏等于原生 GT。

### Hybrid Neural Microfacet BRDF（2026）→ E02/E03

[深入报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/2026-hybrid-neural-microfacet-brdf.md) 把物理 core 与 correction 分工，core loss/full loss 消融说明训练分工本身会影响结果；更深的 Disney core 也可能不稳定。

可据 residual 符号和饱和检查 bounded head 与分阶段冻结。论文的 measured/isotropic 范围、positive correction 以及输入 padding 等未闭合细节，均不能直接证明本项目 signed correction、原生多 UV 或 NVIDIA 全系统等价；E03 是项目假设。

### Hierarchical Neural Materials（2024）→ E06

[深入报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/xue-2024-hierarchical-neural-materials.md) 使用图像缓冲上的 Inception CNN、Fourier 特征、空间梯度损失和 HDR 变换。可借训练时空间辅助 loss 与层级监督；图像 CNN evaluator 不满足本项目随机单次 query 的 runtime 合同。论文印刷 `I^-4` 与文字 root 表述有冲突，fourth-root 若试验必须标为项目定义，不盲抄公式。

### Angular Parameterization（2025）、Neural Biplane BTF（2023）→ E08

[角度坐标报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/xu-2025-improving-angular-parameterization.md) 只覆盖短 poster 的两类 BTF 和十种坐标，适合触发控制实验，不提供通用排名。[Biplane 报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/fan-2023-neural-biplane-btf.md) 的空间 U/half-vector H plane 给出 factorization 候选；额外 reads/bytes 需重新登记，无 LOD/新资产 encoder 证据不能自行补齐。

### NBRDF、Neural Layered BRDFs、MetaLayer、Neural Material Adapter → E02/E07

[NBRDF 报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/2021-neural-brdf-representation-importance-sampling.md) 提醒 weight-space 匹配存在不可辨识性，不能把 decoder 权重 L1 当函数接近。[Layered BRDFs](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/fan-2022-neural-layered-brdfs.md)、[MetaLayer](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/2023-metalayer.md)、[Material Adapter](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/2026-neural-material-adapter.md) 提供 latent/参数到表示的思路，但各有 source-family 与训练条件边界。E07 验证函数响应及真实原生参数编辑，不强迫非层材质提供层参数。

### Active Exploration Neural GI（2022）→ E09

[深入报告](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/diolatzis-2022-active-exploration-neural-gi.md) 的 per-scene MCMC patch 选择使用 loss×优化更新信号；loss-only 可能长期追逐难拟合区域。局部材质可改编为 train-only query bucket 的 recipe/EMA 选择，必须核算 selector 与新 GT 成本；不得借此恢复持久 batch/replay 数据管线。

## 保留启发但本窗口不启动的分支

- [Compositional Neural Scene Representations](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/papers/granskog-2020-compositional-neural-scene-representations.md) 是 scene observations/G-buffer/全局 latent/图像生成问题，可借语义角色辅助监督，不能直接充当局部 BRDF 网络。
- NeLiF、NeLT、Superposed Deformable Feature Fields、Dual-Band GI、LightFormer、Neural Light Probes 与匿名体传输论文的深入报告均在 [目录](../../archive/2026-08/08-29-neural-shading-appearance-literature/research/catalog.md)。组合、频带、物理辅助与 light-transport 分解可保留为后续假设；本次不新增 scene-level GI 协议或承诺其质量收益。
- Comprehensive Neural Materials 的形状/offset/silhouette 混合输出、mobile/VR 的特定部署结论、importance-baking 的 sampler 改造均不自动进入当前 evaluator 实验。
- Belcour 的层模型 core 和 Gaussian-product 的层栈 reference 加速只在该源族语义与实际瓶颈适用时考虑，不能替换 native MDL GT。matched sampler、环境积分、多灯/PT/UE 留给 evaluator 稳定后的独立阶段。

## 历史材料的接口转换

历史报告出现的 `MethodDefinition`、`functional-f@2`、`ncls learn ...`、旧数据 shard、旧 backend/profile 名称和 checkpoint 只保留为溯源。当前实现只走 `Method`、在线 reference 与 `python -m ncls train/validate/export`；source/native 参数与运行时 `ScatteringState` 合同以当前源码、AGENTS 和 spec 为准。

根目录 `docs/research/experiment_framework.md`、`model_candidates.md` 的泛化/分层指标/有界成本原则仍有效，但历史命令和方法规模不直接复制。本任务明确的 bounded signed control 不取消旧死区警告，任何新候选先读 `.trellis/spec/project/method-constraints.md`。无需在每次模型调用重写整个文献库；有新发现时更新对应决策与证据链接。
