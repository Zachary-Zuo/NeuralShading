# 多层传输随机游走参考

`teacher/` 包含一套自包含 Slang 实现，用来计算由 N 个界面和 N−1 个层间介质组成的局部材质响应。数据生成 compute pass 和验证路径包含同一份源码，避免训练数据与渲染参考使用两套不同公式。

主要文件：

- `sampling.slang`：PCG 随机数、余弦半球采样、各向异性 GGX 可见法线采样和 Fresnel；
- `interfaces.slang`：粗糙介电、粗糙导体、Lambert 漫反射和 Charlie 绒面的求值、采样与概率密度；
- `layered_walk.slang`：支持 1–8 个界面的通用随机游走，以及参考 pbrt 的下一事件估计（NEE）和多重重要性采样（MIS）；
- `two_layer_reference.slang`：只用于交叉验证的 Falcor 8.0 内置两层实现；
- `xval/pbrt_probe`：直接调用 pbrt-v4 `CoatedDiffuseBxDF` 的 CPU 探针；
- `bench_walks.py`：包含上传、GPU 执行和回读的端到端吞吐测试。

验证顺序为：先确认单界面解析公式，再对齐两界面随机游走，随后推广到任意层数，最后检查白炉、互易性、插入空界面和 A/B 方差。

当前实现已经通过确定性、有限且非负、单界面解析一致性、采样/概率密度一致性、互易性、吸收与 HG 散射介质、Falcor 交叉检查以及 1–8 层执行测试。深路径与 pbrt-v4 的真空、纯吸收和 HG 散射方向切片最大相对差分别为 0.182%、0.108% 和 0.267%。

RTX 4090 上，三层材质、4096 个方向、A/B 每组 64 samples、`maxDepth=64` 时，端到端吞吐为每秒 `1.223e8` 条随机游走样本。计时包含上传、dispatch 和四个统计量回读。

固定的上游参考位置：

- `external/Falcor/Source/Falcor/Rendering/Materials/LayeredBSDF.slang`
- `external/pbrt-v4/src/pbrt/bxdfs.h`
