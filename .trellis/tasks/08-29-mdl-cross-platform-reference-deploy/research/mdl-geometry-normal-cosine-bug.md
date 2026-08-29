# Bug Analysis：MDL geometry normal 的公共 f measure 漂移

## 1. Root Cause Category

- **Category**：B（Cross-Layer Contract）+ D（Test Coverage Gap）+ E（Implicit Assumption）。
- **Specific Cause**：MDL target code 的`bsdf_diffuse + bsdf_glossy`是`f·|N_source·wi|`。统一query迁移把它除以`init()`后的material-local normal cosine得到`f`，但renderer/parity再按公共输入frame的`|N_input·wi|`乘回response。无normal map时两者相同；铜材质的`geometry.normal`使二者不同，response失败而PDF保持逐值一致。

## 2. Why Earlier Gates Missed It

1. `formal-stb-v6`产生于统一query迁移之前，走的是已删除的MDL专用provider shader；它不能证明公共session的measure转换正确。
2. 既有native fixture只有constant diffuse，`N_source == N_input == +Z`，错误除数被完全掩盖。
3. carpaint formal packet没有触发足以区分两套normal的路径；copper才同时提供normal texture和稳定response/PDF对照。

## 3. Fix and Prevention

| Priority | Mechanism | Action | Status |
|---|---|---|---|
| P0 | Architecture | MDL backend用输入`context.surface.shadingFrame.normal`移除公共transport cosine，把source normal的cosine ratio保留在等价`f`中 | DONE |
| P0 | Independent test | 新增`tilted_normal_diffuse.mdl`，比较SDK native response与`public f × input-frame cosine` | DONE |
| P0 | Formal gate | carpaint/copper按原冻结tolerance重跑，不覆盖旧失败artifact | DONE |
| P1 | Contract | 在reference-query、MDL spec与cross-layer guide记录normal/cosine owner | DONE |

## 4. Evidence

- GPU fixture：`tests/gpu/test_mdl_native_crosscheck.py`，2/2通过。
- Formal：`artifacts/reference-parity/mdl/windows-unified-backend-formal-framecosfix/report.json`。
- carpaint response最大绝对误差`5.960464477539063e-08`；copper为`7.450580596923828e-09`；两者PDF通过。
- `references/mdl-vmaterials2-v1/parity-gate.json`未修改。
