# 公共散射数学 inventory 与公式来源

## 结论

当前仓库没有真正的公共散射数学层：LayerStack reference 文件同时承担 RNG、cosine、GGX/VNDF 与 Fresnel；两个旧 backend 又各复制 LTC basis。01 应先把纯数学迁到 `shaders/ncls/scattering/`，让 reference 与 backend 都变成调用者，再增加 NVIDIA proposal 和 null-event oracle。

## 仓库 inventory

| 现有位置 | 当前所有权 | 01 处理 |
|---|---|---|
| `shaders/ncls/reference/sampling.slang` | PCG RNG、cosine、GGX `D/G/lambda`、VNDF、HG、Fresnel | 纯 cosine/GGX/VNDF/Fresnel 迁公共层；RNG/HG 留在 reference；增加预抽 `u` 重载并让 RNG API 做薄包装 |
| `shaders/ncls/reference/interfaces.slang` | LayerStack interface evaluate/PDF/sample、layer frame | 改为包含公共 frame/microfacet/Fresnel；保持 `NclsRng` 消费顺序与 public entry 语义 |
| `shaders/ncls/backends/legacy_ltc_k2/legacy_ltc_k2.slang` | 一份 LTC basis、局部/世界 frame、cosine proposal | packed ABI 与 identity 暂留，但全部调用公共 LTC/frame/cosine；身份在 06 删除 |
| `shaders/ncls/backends/lobe_residual/lobe_residual_mlp.slang` | 第二份 LTC basis 与解码 | 只保留 backend-private head/state 解码；basis 调公共 LTC；整个旧 identity 在 06 删除 |
| `shaders/ncls/backends/lobe_residual/lobe_residual_core.slang` | 未完成 sampler/PDF TODO | 01 不补旧方法；只让仍使用的 pure math 指向公共层 |
| `tests/gpu/test_legacy_ltc_k2_gpu.py` | evaluate parity | 保持通过；新增独立公共数学 GPU oracle |

仓库扫描确认目前不存在 LTC `sample()`，也没有任何 PDF normalization、sample histogram 或 null-event 测试。

## NVIDIA 官方 sampler 公式

一手来源：NVIDIA Research 的 [Real-Time Neural Appearance Models supplemental](https://research.nvidia.com/labs/rtr/neural_appearance_models/assets/nvidia_neural_materials_author_supplemental.pdf)，相关内容为 Listing 3/4（PDF 第 5–7 页，版面页码 33:5–33:7）。

### 9 raw output range warp

- `alphaX/Y = 1e-4 + 0.5 * (1 + tanh_approx(raw))`
- `rho = tanh_approx(raw)`
- specular/diffuse 两组 slope 使用 `sinh_approx(raw)`
- 两个 lobe 权重使用 `exp(raw)` 后归一
- `tanh_approx(x) = x / sqrt(1 + x²)`；论文名为 `sinh_approx` 的函数实际是 `x * sqrt(1 + x²)`

实现采用减去最大 logit 的 stable softmax；对极端有限 raw 增加不会改变正常区间结果的 correlation/denominator 数值 guard，并在 03 的 baseline provenance 中登记。

### Tilted cosine

预测 slope 构造 `n = normalize((-slope.x, -slope.y, 1))`，从该 normal 的 cosine hemisphere 采样；PDF 是把查询方向变到该 frame 后的 `local.z / pi`。本项目只支持 canonical upper hemisphere，因此 sample 落到 `wi.z <= NCLS_MIN_COS` 时进入显式 null bin，不能重采。

### Non-centered anisotropic GGX NDF

标准 slope 采样：

```text
s = sqrt(u.x) / sqrt(1 - u.x)
sx_std = s * cos(2*pi*u.y)
sy_std = s * sin(2*pi*u.y)
```

椭圆相关变换与中心偏移：

```text
sx = alpha_x * sx_std + mu_x
sy = alpha_y * (rho*sx_std + sqrt(1-rho^2)*sy_std) + mu_y
h = normalize((-sx, -sy, 1))
wi = 2*dot(wo,h)*h - wo
```

PDF 逆变换查询 half-vector slope，计算：

```text
p22_std = 1 / (pi * (1 + sx_std^2 + sy_std^2)^2)
p22 = p22_std / (alpha_x * alpha_y * sqrt(1-rho^2))
p_h = p22 / h.z^3
p_wi = p_h / (4 * abs(dot(wo,h)))
```

这是 NDF slope proposal，不含现有 VNDF 的 `G1(wo)` 因子。reflection 落到 canonical upper hemisphere 外时是显式 null event。

### 项目适配

父任务在论文两项混合外加入固定 `epsilon=1/32` canonical cosine safety component。它不是 learned output；9 参数 identity 保持不变。完整 proposal 的 continuous PDF 为三项完整 mixture，因此开放 upper hemisphere 内严格为正。

## LTC 变换

旧代码的 inverse transform 是旋转后的上三角矩阵：

```text
q = A * R(-angle) * wi
A = [[sx, shx, shy],
     [ 0,  sy, shz],
     [ 0,   0,   1]]
```

其 normalized density 为：

```text
p(wi) = det(A) * max(q.z, 0) / (pi * |q|^4)
```

sample 先取 normalized cosine `v`，再闭式计算 `wi = normalize(R(angle) * inverse(A) * v)`。由于 `A` 保持 z 分量符号且正对角有界，LTC 不产生 reflection null event。旧 response basis 等于此 PDF 乘 RGB amplitude。

## 验证含义

- cosine/LTC：固定 quadrature 的 continuous integral 应为 1。
- tilted/NDF：固定 quadrature 得到 continuous mass，GPU sample 统计 null mass；两者之和应为 1。
- VNDF：先验证 sampled half-normal 对 visible-normal PDF；再单独统计 reflection mapping 的 null event，不能拿方向连续积分冒充 normal 分布归一化。
- 完整 NVIDIA proposal：统计 safety/diffuse/specular 三分量的完整 mixture，不按被选分量 PDF 判定。
