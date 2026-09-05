# 跨语言 JSON identity

## 1. 适用范围

Python producer 与 C++ viewer 共同验证 catalog/package identity 时适用。JSON 数值相等不保证序列化字节相同。

## 2. 签名

```text
ncls.core.identity.canonical_json(value) -> UTF-8 text
ncls::canonicalJson(nlohmann::json) -> UTF-8 std::string
ncls::sha256Json(nlohmann::json) -> lowercase SHA-256
```

## 3. 合同

对象 key 排序、UTF-8 非 ASCII 原样、紧凑分隔符；数组顺序不变。有限 double 用 shortest round-trip，十进制指数在 [-4,15] 使用定点，否则 scientific；浮点整数保留 `.0`，保留 `-0.0`。整数保留 int64/uint64 精度。NaN/Inf 拒绝。C++ 统一走 Hash.cpp，不在 reader 内复制 `json.dump()` hash。

## 4. 错误矩阵

| 条件 | 行为 |
|---|---|
| 非有限浮点 | 拒绝 canonical identity |
| 合法包但 JSON 浮点文本不同 | 修 serializer，不跳过 package hash |
| 内容篡改 | identity/hash 验证拒绝 |

## 5. 正常、基础与错误案例

正常：Python 的 1e-5 在 C++ 输出同一指数格式。基础：无浮点的对象仍保持原 identity。错误：以 epsilon 比较 hash 输入，或重写已发布 manifest。

## 6. 必要测试

`tests/integration/test_viewer_json_identity.py` 编译独立 C++ helper，比对 4096 个有限随机 double、指数边界、正负零、Unicode、int64/uint64；另用真实 producer 包确认 viewer 可加载。

## 7. 错误与正确

```cpp
// 错：nlohmann::json::dump() 的浮点表示不等于 Python repr。
sha256(json.dump());
// 对：使用共享的跨语言 canonical serializer。
ncls::sha256Json(json);
```
