# P0 环境与完成项记录

日期：2026-08-22

| 组件 | 锁定版本 |
|---|---|
| Falcor | tag 8.0，`9dc819c162b2070335c65060436041690b7937f8` |
| Falcor 使用的 Slang | 2024.1.34 |
| pbrt-v4 | `5f7a606806a4ac7b939131ded9d7a30ebd02416e` |
| Python | 3.10.21，Conda 环境 `neural-shading` |
| PyTorch | 2.11.0+cu128 |
| CUDA 工具包 | 12.8.61 |
| 显卡 | NVIDIA GeForce RTX 4090 |

已通过：

- FalcorPython Release 构建；
- Falcor D3D12 设备创建；
- PyTorch CUDA 设备创建；
- Python `LayerStack` 打包与解包测试；
- Falcor compute shader 正确读取 752-byte `LayerStack`；
- 自包含的粗糙介电、粗糙导体、Lambert 漫反射和 Charlie 绒面 Slang 接口编译与 GPU 验证；
- 真空夹层两层随机游走的确定性、非负性和互易性验证；
- pbrt-v4 `CoatedDiffuseBxDF` CPU 探针构建与三方方向切片交叉验证；
- 定位 Falcor 8.0 `LayeredBSDF.eval()` 在俄罗斯轮盘赌拒绝路径时丢弃累计贡献的问题；
- 两层真空、纯吸收和 HG 散射方向切片与 pbrt-v4 的最大相对差分别为 0.182%、0.108% 和 0.267%；
- 支持 v0 材质族的 1–8 层随机游走状态机及不变量测试；
- 每方向 14 bytes 的 A/B tile 生成与内存映射回读；
- RTX 4090 三层随机游走实测每秒 `1.223e8` 条路径，达到 P0 的每秒 `1e8` 条目标；
- 多 tile dispatch、材质族级数据划分、分片写入和源码/先验/schema 哈希记录；
- 8192-tile、固定 64-spp 旧路径的端到端标定为 3.568 秒；该数字不外推自适应 oracle；
- 多层“直接连接最底层界面”的下一事件估计与批量自适应采样；
- GGX、LTC、SG 和共享字典的逐 tile 表示上界实验；
- `v0-oracle`：512 个材质族、2048 个 tiles、全部有限、410/51/51 材质族划分、生成耗时 612.769 秒；
- 12 个 Poly Haven CC0 HDRI 的固定清单、MD5 校验和 48 个环境光探针；
- 当前 K2 基线“精确顶层界面 + 两个 LTC 残差瓣”的方向域 median/p90 relative-L1 为 6.73%/31.20%，真实 HDRI 为 1.74%/10.62%；
- 增加第三个残差瓣后，方向域改善到 5.56%/25.24%，真实 HDRI 改善到 1.42%/9.34%，仍未消除长尾；
- 176-byte 三槽 packet 的 PyTorch 与 Falcor 语义全量对齐；
- `v0-train`：5000 个材质族、160,000 个局部状态、2,560,000 个 tiles、40 个分片、全部有限，材质族划分为 4000/500/500；writer 支持按分片断点续写；
- 38,642 参数循环编译器基线在高采样测试集上的 median relative-L1 为 22.70%。

下一个里程碑：先降低导体基底和深层栈的 closure 表示长尾，确认网络最终输出；之后再实现带反射/透射结构的组合算子。当前循环模型保留为学习式组合对照。
