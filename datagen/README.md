# 方向响应数据生成

v0 保存局部多层材质的方向响应。一个 tile 固定一个层栈和一个观察方向，在入射半球的离散方向上记录 `BSDF × 入射余弦`。每个方向保存 A、B 两组独立随机游走均值，用两者的差判断参考数据有多噪。

P0 在 RTX 4090 上通过 Falcor 8.0 Python `ComputePass` 生成数据。v0 只研究局部、均匀材质参数；空间变化的 PBR 参数图到 P2 才加入。

小规模路径包含确定性的 v0 参数先验、等立体角光照方向、偏向掠射角的观察方向、独立 A/B 随机数流、自适应采样，以及每方向 14 bytes 的内存映射记录。运行命令：

```powershell
./scripts/run_falcor_python.ps1 ./datagen/gen_tiles.py --output ./data/pilot_v0 --stacks 4 --views 2 --bins 128 --max-samples 512
```

`metadata.json` 记录数据布局版本、参数先验版本、随机种子、维度、样本数统计和每个层栈的 SHA-256。逐 tile 自适应采样主要用于生成低噪声 oracle 和检查置信区间。

可扩展 writer 会在一次 GPU dispatch 中处理多个 tiles，把同一材质族的全部局部状态放在同一个数据划分中，并分片写入磁盘：

```powershell
./scripts/run_falcor_python.ps1 ./datagen/gen_v0.py --output ./data/pilot_v0_batched --families 8 --local-states 4 --views 4 --bins 128 --samples-per-half 64 --tile-batch 64
```

工程试运行、8192-tile 吞吐标定和 512-family 自适应 oracle 都已完成。正式 `v0-train` 包含 5000 个材质族、160,000 个局部状态和 2,560,000 个 tiles。

长时间 D3D12 任务支持 `--resume` 断点续写。只有 tile 文件和索引文件同时存在，并且形状与样本数检查通过，才把一个 shard 视为完成。全盘检查所有响应是否有限，并抽样统计 A/B 噪声：

```powershell
conda run -n neural-shading python -m datagen.validate_v0
```
