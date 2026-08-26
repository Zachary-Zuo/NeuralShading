# Scattering backend 合同

公共 ABI 是 `prepare(context, compiledMaterial) -> State`，以及 `State.evaluate/sample/pdf`。`evaluate` 输出线性 `f`，PDF 以 solid-angle measure 表示，sample event 明确 reflection/transmission/null。capability 必须与实际入口一致；renderer 只依赖 binding，不解释私有 state、latent、closure 或 source family。reference 与 neural 都通过 `ScatteringPackage@1` 进入同一 host ABI。
