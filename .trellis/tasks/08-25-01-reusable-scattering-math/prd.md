# 01 可复用散射数学原语

## Goal

建立唯一的 Falcor-free Slang 散射数学层，使 NVIDIA sampler baseline、LTC 候选、LayerStack interface、GPU oracle 和后续通用 MethodBundle backend 共用同一份方向变换、分布参数解码、`sample()` 与 solid-angle `pdf()`。本任务先证明数学合同与现有 LayerStack reference 不漂移，后续任务才能安全训练 sampler 和接入 path tracing。

## Background And Confirmed Facts

- 本任务是父任务 `08-25-unified-scattering-method` 的第一个执行 child，无前置 child；`02` 只能在本任务通过 gate、提交并归档后启动。
- `shaders/ncls/reference/sampling.slang` 当前拥有 cosine、GGX、VNDF、Fresnel 与 reference RNG；`interfaces.slang` 直接依赖这些公式。
- `legacy_ltc_k2.slang` 与 `lobe_residual_mlp.slang` 各复制了一份 LTC density/response basis，当前 LTC 没有 `sample()`，全仓也没有 sampler/PDF 的归一化、histogram、null-event 测试。
- NVIDIA 官方 supplemental Listing 3/4 明确给出 9 个 sampler raw output 的 range warp、tilted-cosine、non-centered anisotropic GGX slope sample、half-vector density与 reflection Jacobian；实现以该一手公式为准，并登记本项目的 upper-hemisphere/null-event 与 full-support 适配。
- 现有 GGX VNDF 是 visible-normal 分布；NVIDIA proposal 是 non-centered anisotropic GGX NDF slope 分布。二者不是同一 proposal，必须分开命名和测试。
- 父任务已经冻结首个方法的事件域为 LayerStack upper hemisphere、reflection-only、non-delta。任何 tilted/specular sample 落到该连续域外时都属于显式 null event，不能静默 rejection/resampling。

## Requirements

### R1. 唯一公共源码所有权

- 在 `shaders/ncls/scattering/` 建立按语义拆分的公共 Slang 模块，统一拥有常量/有限值、frame/direction、cosine/tilted cosine、LTC、GGX/GGX VNDF、non-centered anisotropic GGX NDF 与 fixed-size mixture 原语。
- 公共源码只依赖项目 `contracts/` 和其他公共 math 模块，不依赖 Falcor、MethodBundle、LayerStack IR、reference RNG 或具体 neural backend。
- 迁移完成后，旧 `legacy_ltc_k2`、`lobe_residual` 和 LayerStack reference 只调用公共原语，不保留公式副本。

### R2. 统一方向、测度与 sample 结果

- 输入/输出方向均从着色点向外；分布 PDF 一律相对于 solid angle，upper-hemisphere 连续支持域使用 `wi.z > NCLS_MIN_COS`。
- 公共 sample result 必须区分“有效连续方向”“合法 null event”和“无效参数/数值失败”；null event 返回零连续 PDF，调用方不得重采样。
- `sample()` 与 `pdf()` 必须共用同一参数解码、frame 和变换实现；sample 返回的 PDF 必须通过完整分布 `pdf(sample.wi)` 重新求值，而不是只返回被选分量密度。

### R3. Cosine、tilted cosine、LTC 与 mixture

- cosine hemisphere 提供 normalized `sample/pdf`。
- tilted cosine 按预测 slope 建 frame；只把同时落在 canonical upper hemisphere 的方向计入连续密度，其余概率进入显式 null bin。
- LTC 使用一个非奇异上三角 inverse transform 加 tangent rotation；正对角、有限 shear/rotation 和 determinant 检查由统一参数合同保证。`sample()` 使用闭式逆变换，`pdf()` 与 response basis 使用同一个 Jacobian。
- fixed-size mixture 至少覆盖本任务实际需要的三分量选择、CDF 区间 remap、权重归一化和完整 mixture PDF；不引入动态数组或无界循环。

### R4. GGX/VNDF 与 NVIDIA NDF proposal

- 现有 GGX `D/G/lambda` 与 visible-normal sample/PDF 迁入公共层，并新增接受预抽 `float2 u` 的纯函数；reference RNG 版本保留为薄包装，随机数消费顺序不变。
- NVIDIA proposal 忠实实现官方 supplemental 的 9 raw output range warp：`alpha_x/y`、`rho`、specular/diffuse slope 与两项 mixture logit；softmax 和相关系数边界采用数学等价的有限值稳定写法。
- non-centered anisotropic GGX NDF 先在标准 slope 空间采样，再施加椭圆相关变换与 mean slope，转换 half vector后 reflection；PDF 使用 slope density、`1/cos^3(theta_h)` 和 `1/(4|dot(wo,h)|)` Jacobian。
- 论文两项 proposal 外加父任务冻结的固定 `epsilon=1/32` canonical cosine safety component；epsilon 不是第十个 learned 参数。连续 upper hemisphere 内 PDF 严格为正。

### R5. Analytic control 与 reference 迁移

- 建立新身份 `ltc-k2-analytic-control` 的 Falcor-free core，使用 exact top-interface response 与公共 LTC basis；本任务不把它注册成 viewer fallback。
- `legacy_ltc_k2` 在 `06` 删除身份前改为调用公共 LTC/frame/cosine 原语，保持现有 packed ABI、descriptor 和输出语义。
- LayerStack `interfaces.slang`/`sampling.slang` 改为薄适配，现有 reference 固定 seed 的数值、随机数消费和 shard 采集语义不得漂移。

### R6. 工程边界

- 所有循环和数组静态有界；边界输入、grazing 和极端各向异性返回有限值。
- 不修改锁定的 `external/`，不改公共 scattering ABI/schema，不引入 Torch 生产前向或第二份 Python 生产公式。
- 一次性基线探针只放本任务 `scratch/`，运行证据只写 `artifacts/`。

## Acceptance Criteria

- [x] 公共模块只有一份 cosine/LTC/GGX/NVIDIA proposal 公式，仓库扫描证明旧 backend/reference 不再复制这些实现。
- [x] 固定 Gauss-Legendre × azimuth quadrature 证明 cosine 与 LTC 连续 PDF 归一化；tilted cosine、non-centered GGX 与完整 NVIDIA proposal 满足“连续积分 + 显式 null mass = 1”。
- [x] cosine、tilted cosine、LTC 和完整 NVIDIA proposal 的 GPU sample histogram 与独立 PDF quadrature 一致；`sample.pdf == pdf(sample.wi)`。
- [x] GGX VNDF 的 sampled normal histogram 与 visible-normal PDF 一致；reflection 方向落出 upper hemisphere 的 null-event 频率被单独统计，不混入 normal-PDF 结论。
- [x] grazing、`alpha`/`rho`/slope/shear 的冻结边界样本无 NaN/Inf，合法连续 PDF 非负，NVIDIA proposal 在整个开放 upper hemisphere 严格为正。
- [x] 同一公共 Slang 源通过锁定 Slang 2024.1.34/Falcor GPU 编译与数值 oracle；没有仅测试第二份 Python 实现。
- [x] 迁移前后的 LayerStack reference 固定 seed probe 在冻结容差内一致，reference physics/shard 回归通过。
- [x] `ltc-k2-analytic-control` core 可编译，旧 `legacy_ltc_k2` Python/Slang parity 继续通过且 packed ABI 未变。
- [x] `external/Falcor` 与其他锁定上游保持原提交和干净工作树。
- [ ] 质量检查、spec 判断、scoped local commit 与归档完成后，父任务才进入 `02`。

## Out Of Scope

- neural evaluator/sampler head、loss、训练、directional mollification 数据判断与 corpus 重采。
- MethodBundle exporter/loader、backend generic specialization、viewer deferred/PT 和 capture/replay。
- transmission、delta、volume-boundary proposal；LayerStack reference 已有的内部 transmission 行为只做语义保持。
- 删除 `legacy_ltc_k2`、`lobe_residual` 或 Film 方法身份；替代链路验收后的物理删除属于 `06`。
- 修改未版本化数据或 artifacts；本任务只生成可重建的测试/诊断证据。

## Planning Convergence

- 用户价值、范围、兼容性、风险和可观察验收已由父任务冻结；本 child 没有新的用户所有决策。
- 技术未知项已通过仓库 inventory 与 NVIDIA 官方 supplemental Listing 3/4 解析；实现细节不会改变父任务产品边界。
- Blocking open questions：无。
