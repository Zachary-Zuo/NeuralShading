# NVIDIA neural appearance baseline 方法对应关系

## 权威来源与适用边界

一手来源固定为 NVIDIA 作者版论文 [Real-Time Neural Appearance Models](https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_paper.pdf)、[作者补充材料](https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_supplemental.pdf) 与 [NVIDIA Research 项目页](https://research.nvidia.com/labs/rtr/neural_appearance_models/)。结构与运行时算术以补充材料 §2 的 Slang 伪代码为最高可执行依据；论文正文 §4–5 冻结方法语义、训练目标和运行规模。

`D:\01_Workspace\Real-Time Neural Appearance Models` 仅作只读二手对照。它确认了若干结构线索，但其 LayerStack GT、softening 配对、`dual_batch` 数据流和 sampler objective 没有独立证明，不能作为 correctness oracle，也不能把其中的 Torch 前向复制到本项目。

## 逐项 correspondence

| 项目 | 一手定义 | 03 正式实现合同 | 明确适配/偏差 |
|---|---|---|---|
| latent | 每个空间/LOD 查询读取 8D latent；两张 RGBA FP16 texture | 当前 uniform state 保存独立 `z8`，packed asset 使用 FP16 | 当前事件域没有空间纹理；省略 encoder/texture pyramid，不改变单个 uniform query 的 decoder 函数 |
| learned frame layer | `z8→12` 单线性层，无 bias、无激活 | 完全同形 | 无 |
| frame 构造 | 每 6 个 raw：`N=normalize(rawN+(0,0,1))`、`T=normalize(rawT+(1,0,0))`、`B=cross(N,T)`；补充材料明确 `(T,B,N)` 不正交化 | 完全同算术；frame 只依赖 latent，不依赖 view | 设计文档早期的 Gram–Schmidt 描述已判为二手复现漂移，不进入实现 |
| evaluator 输入 | 先写两个 frame 中的 fixed incident / queried outgoing 方向，共 12D；latent 写入最后 8D，合计 20D | 项目 fixed `wo` 映射论文 `wi`，queried `wi` 映射论文 `wo`；顺序为 `frame0(wo,wi), frame1(wo,wi), z8` | 仅变量命名适配，保证 sampler 仍以 fixed project `wo` 为条件 |
| evaluator 规模 | 论文比较 `2×16`、`2×32`、`3×64`；任务冻结原规模 `3×64` | `20→64→64→64→3`，各 hidden 使用 ReLU | 补充材料伪代码展示 `2×32`，这里只取论文已正式报告的最大配置，不称为伪代码默认配置 |
| evaluator 输出 | `exp(raw−3)`；伪代码返回 `f(wi,wo)·dot(n,wo)` | 内部网络输出项目 response `y=f·wi.z`；公共 `evaluate()` 返回 `f=y/max(wi.z, ε)`，训练与评测只乘回一次 cosine | 项目公共接口强制返回线性 `f`；response/API 转换及 grazing epsilon 进入 correspondence 与 parity，不改网络监督量 |
| albedo | evaluator 可选输出额外 RGB directional albedo | 关闭可选 albedo head | 一手来源明确为 optional；当前 03 没有 albedo runtime capability |
| sampler 输入 | fixed incident direction 3D 写在前，latent 8D 写在后，共 11D | project `wo.xyz + z8` | 仅方向命名适配；不得复用 view-conditioned evaluator trunk |
| sampler MLP | `11→32→32→32→9`，hidden ReLU、输出线性 | 完全同形 | 无 |
| sampler raw 顺序 | `alphaX, alphaY, rho, slopeSpecX/Y, slopeDiffX/Y, wSpec, wDiff` | 完全同顺序 | 无 |
| sampler warp | `tanh_approx(x)=x/sqrt(1+x²)`；`sinh_approx(x)=x*sqrt(1+x²)`；alpha 加 `1e-4`；两个权重指数归一化 | 复用 01 `nvidia_proposal.slang` 的同式 decode | 为 full support 加固定 `1/32` cosine safety 分量，是 03 强制 adaptation；报告原两 lobe 与 safety mass |
| sampler解析分布 | tilted-cosine diffuse + non-centered anisotropic GGX specular；sample 后用同一 mixture PDF | 只调用 01 公共 proposal `sample/pdf`；below-surface specular 为显式 null | 三态与 epsilon safety 是 01/03 运行时合同，单独做 oracle/correctness |
| evaluator loss | 线性输出在 log space 使用 L1 | `mean(abs(log1p(y_pred)-log1p(y_target)))` | 一手正文未给 log epsilon/offset 的更细公式；`log1p` 是冻结的数值解析，进入 config/hash，不混入 linear/energy/peak loss |
| sampler objective | evaluator 与 sampler 联合训练、各用独立 batch；当前 learned BRDF 作为 KL target；KL 对 latent detach，不能干扰 evaluator/共享 latent | baseline reproduction 保留 joint stage：BRDF loss 更新 evaluator/latent，KL 只更新 sampler head；另为2×2 comparison注册 frozen-evaluator GGX9/LTC adaptation | 离散 64-direction group 是 02 冻结 corpus 的适配；零能量 group 使用 cosine target；冻结 stage 不替代 joint reproduction |
| optimizer | Adam `β=(0.9,0.999)`、`eps=1e-7`、zero weight decay；LR `1e-3→1e-4` cosine | 原 baseline formal config 完全同 optimizer/schedule | 当前统一 runner 需要显式支持 Adam epsilon 与 zero decay，字段进入 config/checkpoint hash |
| mollification | 前 20k iteration，cone `10°→0°` cosine；每 target 256 cone samples | 只经 training entry `47ef…5a89` 使用 02 冻结的四个正半径 level，`.875` 后回 base-v5，20k 后继续 base | 离线有限 level 及现有 reference sample 数是数据合同适配，不冒充连续在线 256-sample schedule |
| 原论文预算 | 300k iteration；evaluator/sampler 各自 batch 65k；FP32 master、加载时 FP16 | 03 使用预先冻结的有限 corpus、formal step/batch 类别与多 seed；FP32 master、导出 FP16 | 训练数据量/预算不等同论文，convergence 只证明本任务复现运行稳定，不宣称重现论文图像质量 |

## 当前实现迁移判定

旧 `nvidia-frame-two-lobe-realtime-v1` 的 `z16`、view-conditioned `prepare 23→64→64`、17D feature、`32×2` evaluator、LeakyReLU、softplus/output floor 和复合 appearance loss 不符合上述 baseline，对应 run 只能登记为 `superseded-diagnostic`。不得只改名称后继续复用 checkpoint。

正式 baseline 使用新 pipeline/config identity；exact-core candidate 保留 `z16 + top-interface + positive residual` 私有形态。两者只共享 Falcor-free Slang 基础算术与公共 runtime contract，不共享 latent/state/网络宽度，也不靠 padding 伪造 matched bytes。

## 验收证据

- exact-vector unit 锁定 frame base offset、非正交 cross basis、20D/11D 顺序、`exp(raw−3)` 与 sampler raw 顺序；
- SlangPy cold/warm autodiff 锁定 evaluator 与 sampler 各训练角色的 callable identity；
- joint stage 锁定 BRDF loss 更新 evaluator/latent、sampler KL 只更新 sampler head且 latent detach；frozen matched stage 使用独立 checkpoint/config identity；
- Falcor/SlangPy 使用相同 FP32 与 FP16-packed 参数做完整 evaluator/PDF parity；
- convergence report 分开判 implementation、有限性、初始化到 best 的 validation 改善、late trend、多 seed 一致性和 checkpoint 恢复；
- quality/cost 只描述本任务数据上的结果，不反向修改 correspondence。
