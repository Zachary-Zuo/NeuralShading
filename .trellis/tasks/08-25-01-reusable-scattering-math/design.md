# 01 可复用散射数学原语设计

## 1. 边界与依赖方向

本任务建立纯 Slang 数学组件，不新增公共 ABI 字段。依赖固定为：

```text
contracts/scattering_contract.slang
              ↓
shaders/ncls/scattering/*          （唯一公式所有者，Falcor-free）
       ↙                 ↘
reference/{sampling,interfaces}    backend private core / GPU oracle
```

`reference/sampling.slang` 继续拥有 `NclsRng`、PCG 与 Henyey-Greenstein；它通过薄包装把预抽 `float2 u` 传给公共 sample 函数。这样 LayerStack reference 的随机数消费顺序保持不变，neural backend 又能直接传入 `ISampleGenerator` 取得的随机数。

## 2. 模块布局

新目录 `shaders/ncls/scattering/` 作为公共核心 shader 目录，并在 `docs/architecture.md` 登记：

| 文件 | 单一职责 |
|---|---|
| `common.slang` | `NCLS_PI`、`NCLS_INV_PI`、`NCLS_MIN_COS`、safe normalize、有限值 guard、公共 directional sample 状态 |
| `frame.slang` | canonical/local/world frame、slope normal frame、rotation；以 `NclsShadingFrame` 为公共 world adapter |
| `cosine.slang` | cosine 与 tilted-cosine `sample/pdf` |
| `ltc.slang` | `NclsLtcTransform`、参数有效性、闭式 inverse sample、normalized PDF、RGB response basis |
| `ggx.slang` | GGX `D/G/lambda`、visible-normal sample/PDF、non-centered anisotropic NDF sample/PDF |
| `fresnel.slang` | dielectric/conductor Fresnel，供 LayerStack interface 与 analytic control 共用 |
| `mixture.slang` | fixed K=3 权重归一、component select/remap 与完整 mixture PDF |
| `nvidia_proposal.slang` | 9 raw output decode、固定 cosine safety component、两项论文 proposal 的完整 `sample/pdf` |

新 analytic control 位于 `shaders/ncls/backends/ltc_k2_analytic_control/ltc_k2_analytic_control_core.slang`。它只实现 exact-top + K2 LTC 的 Falcor-free evaluate core；通用合同包装、bundle 和 viewer 注册由 04 接管。

## 3. 公共 sample 状态

公共 distribution sample 使用一个统一结果：

```text
NclsDirectionalSample
  direction
  pdf          // continuous solid-angle PDF；null/invalid 为 0
  valid        // 参数与数值构造有效
  nullEvent    // 有效离散 null bin；不是实现失败
```

状态解释：

- continuous：`valid=1, nullEvent=0, pdf=full_pdf(direction)`；
- null：`valid=1, nullEvent=1, pdf=0`，保留有限 direction 仅用于诊断；
- invalid：`valid=0, nullEvent=0, pdf=0`。

调用方只可对 continuous sample 计算 `f*|cos|/pdf`；null 返回零贡献。任何分布 sample 最后都调用自己的完整 `pdf()`，防止 component density 与 mixture density 混淆。

## 4. 数学合同

### 4.1 Frame 与方向

- public API 使用 NCLS `wo`/`wi`：均从着色点向外；canonical normal 为 `+z`。
- slope normal为 `normalize((-sx,-sy,1))`；frame 构造在接近切线轴退化时选择备用轴，保持正交与右手性。
- LTC rotation 采用旧实现已冻结的 `R(-angle)` query / `R(angle)` sample 对偶；LayerStack tangent rotation 结果不变。

### 4.2 Cosine 与 tilted cosine

canonical cosine：`p(wi)=wi.z/pi`，只在 `wi.z>NCLS_MIN_COS` 有效。tilted cosine 在 slope frame 中应用同一 density，同时要求 canonical `wi.z>NCLS_MIN_COS`；倾斜后落到 canonical 下半球的概率进入 null bin。

### 4.3 LTC

`NclsLtcTransform` 保存正 inverse scale、三个 shear 与 tangent rotation。query 计算 `q=A R(-angle)wi`，使用 `det(A)*q.z/(pi*|q|^4)`；sample 对 cosine direction 应用上三角闭式 `A^-1` 与 `R(angle)` 后 normalize。参数 head 的 raw warp 仍是 backend-private；公共层只校验 decoded transform。

### 4.4 GGX VNDF

`nclsSampleGgxVisibleNormal(wo,alpha,float2 u)` 是纯函数；旧 `NclsRng` overload 只做一次 `nextFloat2()` 后调用它。normal PDF 仍为 `G1(wo)*abs(dot(wo,h))*D(h)/abs(wo.z)`。若后续把 sampled normal reflection 成 canonical-domain direction，below-surface 结果显式标 null；不在 VNDF normal PDF 内重采。

### 4.5 NVIDIA 9 参数 proposal

实现严格按 `research/math-inventory.md` 中官方 Listing 3/4。数值稳定适配只有：stable softmax、随机数上界钳到 `<1`、`rho`/denominator 避免 float round-off 奇点、所有输出有限值检查。正常 raw 区间中的公式不变。

完整 proposal：

```text
p = epsilon * p_cos
  + (1-epsilon) * (w_spec * p_noncentered_ggx + w_diff * p_tilted_cos)
epsilon = 1/32
```

sample 用 `u.x` 选择 K=3 分量并 remap 到被选区间，`u.yz` 驱动方向 sample；返回后统一计算上式。diffuse/specular 域外 sample 是 null，safety cosine 永不 null，因此 continuous upper hemisphere full support 且严格正。

## 5. 迁移策略

1. 先新增公共模块和 GPU oracle，不改变现有调用者。
2. 将 `reference/sampling.slang` 的 public 名称改为 include/re-export 或同名薄包装；`interfaces.slang` 改用 pure `u` overload，但保持每个 branch 的 RNG 调用次数和顺序。
3. `legacy_ltc_k2` 保留 packed struct/descriptor，frame/cosine/LTC response 改调公共函数。
4. `lobe_residual_mlp` 的 private lobe decode 保留，response basis 改调公共 LTC；不实现其旧 TODO sampler。
5. 建立新 analytic control core，并通过 GPU compile/evaluate probe。

迁移不改 scattering ABI/schema，失败时可以按文件组回退到最近通过 gate 的步骤；不能通过复制公式回退。

## 6. 测试设计

### 6.1 独立方向集与 quadrature

- 固定 `Gauss-Legendre(z∈[0,1]) × uniform azimuth` 网格，至少 `128×256` 点。
- cosine/LTC continuous integral 误差 `≤2e-4`。
- tilted/NDF/NVIDIA proposal：continuous integral 与至少 `2^17` GPU samples 得到的 null frequency 相加，误差 `≤1e-2`；同一固定 seed 可重复。

### 6.2 Histogram 与 re-evaluation

- 使用 `(z,phi)` 等 solid-angle bins；将 quadrature 期望质量与 GPU samples 比较，total variation `≤0.04`，低期望样本 bin 合并。
- 每个 continuous sample 断言 `sample.pdf` 与独立 kernel `pdf(sample.direction)` 在 `rtol=2e-5, atol=2e-6` 内一致。
- mixture 测试只比较完整 PDF，另检查 component 选择频率。

### 6.3 边界

覆盖 grazing `wo.z`、`alpha={1e-4,1}`、`rho≈±1`、大 slope、LTC scale/shear/rotation 边界和 `u` 接近 0/1；所有合法输出有限，invalid 与 null 不混淆。

### 6.4 Reference 与旧 ABI

- 实现前在完整开发机上运行固定 seed LayerStack probe，把 current HEAD 数值写入 `artifacts/`；实现后用同命令对比。
- 运行 `tests/integration/reference/test_reference_physics_gpu.py`、reference shard smoke、`test_legacy_ltc_k2_gpu.py` 和 packed ABI unit test。
- `git -C external/Falcor status --short` 必须为空，HEAD 必须仍为锁定 commit。

## 7. 失败规则

- 任何 distribution 无法给出同一 transform 下的 exact sample/PDF：不进入公共 family，任务回到设计。
- normalization/histogram/re-evaluation 任一失败：不得启动 02。
- LayerStack 固定 seed probe 漂移：先定位 RNG 消费或公式变化，不能提高容差掩盖。
- 锁定 Slang 编译不接受某种语法：只改纯函数语法/布局，不维护第二份数学实现。
