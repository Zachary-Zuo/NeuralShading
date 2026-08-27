# Reference 与 batch

四个正式 family 共享 query stream 和 `TrainingBatch@1`。offline reader 与 live Falcor executor 只是 producer；live tensor 必须驻留 CUDA、显式同步并受 lease 保护。proposal、target estimator、seed 与 sharding 是 recipe。HDF5 只由 offline corpus sink 写入。旧平行采集/reader 不存在。详见 `../project/unified-pipeline.md`。

## Falcor/CUDA live producer 生命周期

- CUDA 输入使用 Falcor `Buffer.from_torch()` 提交 device-to-device copy，再由 `wait_for_cuda()` 建立 CUDA→Falcor 顺序；不得长期持有输入 buffer 的 `to_torch()` 映射并在训练循环中原地覆写。
- Falcor 输出可由 shared buffer 的 `to_torch()` 映射零拷贝消费，但 compute dispatch 后必须先调用 `wait_for_falcor()`，下一次 Falcor 读取共享输入前仍由 `wait_for_cuda()` 纳入此前 CUDA consumer。
- 一个 logical training iteration 是一个 Falcor frame。只有当该 iteration 的全部 route lease 都释放后才调用一次 `device.end_frame()`；不得跨 iteration 累积 tiled dispatch，也不得在仍有 active lease 时轮转 transient heap。
- 新增 live producer 的测试必须同时覆盖同步/lease 单元合同和包含真实 forward、backward、optimizer 的连续 GPU soak；短 smoke 不能替代 frame 生命周期验证。

## 场景：从不可变 payload 上传 Falcor texture

### 1. Scope / Trigger

从 package bytes、memory-mapped 文件或其他不可变 payload 解码 NumPy mip，再通过 Falcor Python 上传 texture 时适用。该边界容易把“C contiguous”误当成“可上传”；Falcor uploader 还要求数组 backing storage 可写。

### 2. Signatures

```python
falcor.Texture.from_numpy(data: np.ndarray, mip_level: int = 0) -> None
```

### 3. Contracts

- `data` 必须是与 texture format/当前 mip extent 一致的 C-contiguous ndarray。
- `data.flags.writeable` 必须为 `True`。`np.frombuffer(immutable_bytes, ...)` 返回的 view 即使 contiguous 仍是只读。
- `np.ascontiguousarray(read_only_view)` 可能直接返回原 view，不能作为获得可写 storage 的保证；payload decoder 在上传边界使用 `.copy()`。

### 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| dtype、channel 或 mip extent 与 texture 不符 | 上传拒绝；测试失败，不做隐式 reshape/cast |
| ndarray contiguous 但 backing storage 只读 | pybind overload 拒绝并抛 `TypeError` |
| ndarray contiguous、可写且格式匹配 | 上传当前 mip；shader fetch 必须读回预期值 |

### 5. Good / Base / Bad Cases

- Good：DDS bytes 用 `np.frombuffer()` 解码后 `reshape(...).copy()`，逐 mip 上传并用 shader probe读回。
- Base：由 NumPy运算新建的、已可写 contiguous array可直接上传。
- Bad：只调用 `np.ascontiguousarray()` 处理 immutable bytes view，然后把 pybind `TypeError` 误判为 texture format不支持。

### 6. Tests Required

- RGBA16F人工 mip texture至少包含两个取值不同的 level。
- shader probe断言显式 mip fetch、相邻 mip stochastic selection与 level 内 bilinear结果；测试必须真实调用 `Texture.from_numpy()`，不能只检查 ndarray flags。

### 7. Wrong vs Correct

```python
# 错：contiguous 仍可能只读
level = np.frombuffer(payload, dtype="<f2", count=count, offset=offset)
texture.from_numpy(np.ascontiguousarray(level.reshape(height, width, 4)))

# 对：上传边界显式取得可写 backing storage
level = np.frombuffer(payload, dtype="<f2", count=count, offset=offset)
texture.from_numpy(level.reshape(height, width, 4).copy())
```
