# 材质范围

源材质的原生参数、图结构和资源属于GT。LayerStack、OpenPBR、MERL、MaterialX、MDL各自保留source contract与权威reference；除非源本身是层模型，不得先反演成LayerStack。可调source必须通过typed editor保留编辑能力。训练状态经`TrainingCheckpoint@1`保存，部署产物通过三段式`ScatteringPackage@2`交付。

MDL source family 保存 module、exact export signature、class-compiled typed arguments、传递依赖与原生纹理资源。MDL SDK 是语言、标准 closure 与 target code 的上游语义核心；项目 bridge 和当前 Falcor 8 是唯一正式执行路径。锁定 falcor2 使用同版 MDL SDK，只用于 renderer integration parity，不是第二条 GT 或运行时 fallback。

MDL V1的runtime reference与其他source一样完整实现canonical `prepare/evaluate/sample/pdf`，四个入口都落到同一compiled target code；缺任一入口时descriptor fail closed。online训练通过公共backend session调用其中的`prepare/evaluate`，source `sample/pdf`用于transport与数值验证。MDL SDK compiler只是该program的内部toolchain provider；不存在MDL专用公共backend/query/batch入口，也不通过viewer adapter或fallback拼接能力。
