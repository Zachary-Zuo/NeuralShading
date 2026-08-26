# 材质范围

源材质的原生参数、图结构和资源属于 GT。LayerStack、OpenPBR、MERL、MaterialX 各自保留 source contract 与权威 reference；除非源本身是层模型，不得先反演成 LayerStack。可调 source 必须通过 typed editor 保留编辑能力。方法产物经 `TrainingCheckpoint@2` 与 `ScatteringPackage@1` 交付。
