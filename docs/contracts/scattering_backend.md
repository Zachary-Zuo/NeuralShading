# Scattering backend 合同

公共 scattering ABI 是 `prepare(context, compiledMaterial) -> State`，以及 `State.evaluate/sample/pdf`。context 包含 shading/geometric frame、outgoing direction、UV、`uvDx/uvDy`、material instance 和 stochastic sample；`prepare` 可过滤 latent 并缓存 view-conditioned state。`evaluate` 输出不含 cosine 的线性 `f`，连续事件的 PDF 以 solid-angle measure 表示，sample event 明确 reflection/transmission/delta/null。

用于 path tracing 的 backend 必须同时提供四个入口；descriptor 缺任一 capability 时 fail closed。连续 sample 的数学合同是 `weight = f * |n_s·wi| / pdf`，且 `sample()` 与 `pdf()` 属于同一 proposal；invalid/null sample 终止当前路径，不切换 generic proposal。对 source-native API，sample 返回的 direction/event/PDF/throughput weight 是不可拆分的数值 tuple，adapter 必须原样保留。极窄峰在掠射反射时可能依赖 sampler 内部未舍入的 half-vector；禁止用已舍入 `wi` 再调独立 `pdf/eval` 重建 tuple，否则 float32 切向量相消会制造巨大权重。独立 `pdf(wi)` 仍由同一 native proposal 负责并用于 NEE。renderer 只依赖 canonical state，不解释私有 state、latent、closure、proposal 或 source family。

source reference 与 neural package 共享上述 scattering ABI，但不共享虚假的交付身份：source 由各自权威 backend 和 scene composer 进入 integrator，`ScatteringPackage@1` 则通过稳定 `NclsPackage*` host binding 交付已编译方法。两者都不能靠兼容 adapter 补出缺失的材质语义。
