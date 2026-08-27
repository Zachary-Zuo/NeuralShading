# 材质范围

源材质的原生参数、图结构和资源属于 GT。LayerStack、OpenPBR、MERL、MaterialX、MDL 各自保留 source contract 与权威 reference；除非源本身是层模型，不得先反演成 LayerStack。可调 source 必须通过 typed editor 保留编辑能力。方法产物经 `TrainingCheckpoint@2` 与 `ScatteringPackage@1` 交付。

MDL source family 保存 module、exact export signature、class-compiled typed arguments、传递依赖与原生纹理资源。MDL SDK 是语言、标准 closure 与 target code 的上游语义核心；项目 bridge 和当前 Falcor 8 是唯一正式执行路径。锁定 falcor2 使用同版 MDL SDK，只用于 renderer integration parity，不是第二条 GT 或运行时 fallback。MDL V1 对外只声明 surface-BSDF `evaluate`，纹理查询固定 `ExplicitLod(0)`；viewer 已直接接入相同 compiled artifact，并在内部调用同一 target code 的 `sample/pdf` 正确积分尖锐 flakes/coat。这个 viewer transport 细节不把训练/provider 的 source capability 扩成公共 matched sampler。独立 image parity 与 derivative filtering 仍保持未声明。
