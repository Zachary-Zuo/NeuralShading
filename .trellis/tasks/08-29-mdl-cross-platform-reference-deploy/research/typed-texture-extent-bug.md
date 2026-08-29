# Bug Analysis: scalar texture3d extent轴解析错误

## 1. Root Cause Category

- **Category**：B（Cross-Layer Contract）+ D（Test Coverage Gap）+ E（Implicit Assumption）。
- **Specific Cause**：program payload已按`[depth,height,width]`描述scalar 3D texture，但公共binder使用为`[depth,height,width,channels]`编写的负索引取width/height。`[33,64,65]`因而被创建为`64×33×33`；549120-byte Float32 payload与278784-byte Falcor subresource精确不符。

## 2. Why Fixes Failed

1. 之前的MDL fixture只有2D texture或无纹理，未经过scalar 3D BSDF-data上传路径。
2. 既有3D数据的三个extent都由manifest正确记录，但没有纯函数测试验证shape到GPU extent的映射；typed descriptor验证只覆盖元素数量，没有覆盖resource creation参数。
3. 如果只观察“payload size mismatch”，可能误判为MDL artifact或decoder问题；实际bytes与错误extent的乘积给出了区分轴解析错误的直接证据。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | 用`_texture_extent()`集中解析spatial-first shape，2D/3D只允许明确rank | DONE |
| P0 | Test Coverage | unit覆盖scalar/RGBA 2D/3D，3D使用互不相等的33/64/65 | DONE |
| P0 | Integration | 重跑含SDK BSDF-data texture的carpaint/copper formal parity | DONE |
| P1 | Documentation | 在reference-query spec与cross-layer guide固化axis owner和验证点 | DONE |

## 4. Systematic Expansion

- **Similar Issues**：2D scalar、2D RGBA与3D RGBA当前也经过同一helper；unit显式覆盖，防止后续为某个dtype再次分叉。
- **Design Improvement**：typed descriptor负责声明shape，binder只做一次spatial-first解释；pixel format选择不能反向改变axis含义。
- **Process Improvement**：任何新增texture kind/dtype必须同时提供非方形/非立方extent测试和至少一条真实GPU上传证据。

## 5. Knowledge Capture

- [x] 更新`.trellis/spec/data/reference-query.md`。
- [x] 更新`.trellis/spec/guides/cross-layer-thinking-guide.md`。
- [x] 更新稳定合同`docs/contracts/reference_query.md`。
- [x] 新增`tests/unit/test_reference_texture_binding.py`。
- [x] formal MDL parity通过：`artifacts/reference-parity/mdl/windows-unified-backend-formal-framecosfix/report.json`；carpaint/copper与PDF均通过冻结gate。

本项目不存在`src/templates/markdown/spec/`镜像，因此没有可同步的spec template。任务仍需Linux/A6000后半段，按task计划不在Windows阶段单独commit。
