# 共享 Slang scattering backend

每个 program 私有 module 实现 `INclsScatteringBackend` 及稳定 `NclsPackage*` host ABI。训练、Falcor parity 与 viewer 使用同一 module closure；offset 来自编译器反射，运行循环静态有界。package 从自身绝对路径加载，renderer 不识别 program key。测试必须覆盖 evaluate/sample/pdf 有限性、sample→pdf、独立 oracle 和 package 编译。
