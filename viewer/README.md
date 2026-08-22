# Falcor 查看与实时验证

查看器将让三条路径复用同一份主可见性结果：随机游走参考、逐 tile 优化得到的 oracle closure，以及网络预测的 closure。这样比较时只有材质求值方式不同，几何、相机和光照保持一致。Falcor 固定为 `external/Falcor` 中的 8.0。

`oracle_lookup.py` 和 `kernels/oracle_lookup.cs.slang` 是目前已经跑通的第一条实时路径。它们上传固定 176-byte packet，并调用未来延迟光照 pass 也会使用的 Slang 求值函数。

```powershell
./scripts/run_falcor_python.ps1 -m viewer.validate_oracle_lookup
```

2048 个 oracle tiles 全量验证后，Falcor 与归档 FP16 预测之间的 relative-L1 为 median 0.017%、p99 0.024%、最大 0.099%。下一步不是继续做查表，而是在表示方案确认后，把网络预测和延迟光照接到同一条路径上。
