# Metal matched sampler 验证记录

## 交付形态

- full method保持`metal_fused_full_v1`，未缩减texture codec、typed compiler、6+4 evaluator lobes、angular bank或proposal branch；
- proposal固定为6个analytic core、4个positive residual和1个full-hemisphere fallback，共11个component；
- `PreparedState`保存`float32[11,8]`proposal state，字段和component enum由JSON生成Slang常量；
- Python实现独立`sample()/pdf()`、forward/reverse PDF和throughput weight；Slang实现同一folded-preimage mixture；
- method capability为`PREPARE | EVALUATE | SAMPLE | PDF`，20个required component、13个parameter group。package compiler仍由下一runtime child完成，在此之前只对package export fail closed。

## 数学与异常边界

- 每个局部分布采样后把renderer-z负方向折回上半球；密度显式累加原方向和z镜像方向两个preimage。11种单component+fallback mixture用`128 × 256`确定性半球积分验证归一化，误差界为0.3%。
- 一个`float2`同时完成CDF选择、区间内radial重映射和azimuth；sample后只执行一次directional evaluator。
- valid sample重新调用独立`pdf()`后forward/reverse PDF一致，weight严格由`f * max(wi.z,0) / pdf.forward`形成。
- binary topology `active`与连续compiler activity分离；连续activity仍参与可微mixture mass。
- zero/nonfinite energy、weight/active矛盾、错误enum/frame、退化authored frame、grazing方向、非法随机tuple与prepared invalid全部fail closed且不产生NaN/Inf；合法axis/tangent孤立共线由确定性fallback tangent处理，不误判坏state。

## 训练证据

环境为完整Windows，RTX 4090，`neural-shading`环境与Falcor Release可用。最终运行：

```powershell
scripts/run_falcor_python.ps1 -m ncls.cli learn train `
  configs/learning/metal-fused-full-windows-smoke.json `
  artifacts/08-30-metal-matched-sampler/full-method-verified-v2.pt
```

- full profile三phase各4步：codec FP32、joint evaluator BF16、proposal BF16；真实vMaterials Metal MDL authoritative online reference；
- 12步、60个work units、峰值显存393,684,992 bytes、训练主体7.31秒；
- checkpoint SHA-256：`558887b456c64b37f85e7e6d1779c62c1960795359e35c5781a252d09cd37753`；
- implementation：`5ab52b21e714fc4bd799d61258ac52654d06ce9c5eac547d430bc3027bae6c9b`；
- descriptor：`5b608950370775e44df8077e68a80bb9e6c4afec4bddb1e060bafdf55a65c082`；layout：`9f7efbad4213391974f16a37615da62bfefb59ff0d740aa321ed2e4fc3b9e64a`；
- complete checkpoint有20个component、328个state tensor；13/13 parameter groups均记录finite gradient、nonzero gradient和parameter update，`proposal_sampler.last_audit_step=11`；
- 每个proposal训练step与validation的`proposal_valid_fraction=1`、`proposal_identity_error=0`。不同online query的总loss不要求单调；固定target GPU回归另以12次optimizer update验证density loss下降和目标方向density上升。

## Backend、闭包与回归

- `171 passed`：`tests/unit`；
- `42 passed`：`tests/gpu`，经Falcor launcher运行；
- Slang parity使用256个跨state/frame/`wo`/random tuple probes，比较sample direction、selected component、forward/reverse/direct PDF，11个component全部命中；
- full-cohort preflight覆盖692 exports、178 graphs、52 texture sets、4类texture role和20个required component，identity为`52ba5a123c39938f9a62f6f62f0832a9809813214412d1b1a6ed2c462d4edcdf`；
- generated layout `--check`、`compileall`、`git diff --check`通过，`external/Falcor`工作树干净。

## 静态成本边界

- prepared state上限2816 bytes，proposal本体为11 × 8个float；
- `sample`上限48个静态step、2个随机值、1次evaluator调用；`pdf`上限32个静态step；
- `pdf()`只遍历固定11个component，不读取texture、不运行typed compiler/evaluator；
- full profile总texture/angular read上限仍为106。以上是ABI静态合同，不是formal runtime benchmark或效率hard gate。

## Matched controls 与后续统计边界

- `analytic-only@1` control冻结为只保留component 0–5和fallback 10、再归一化同一state；不得用它冒充产品sampler。
- `source-reference@1` control冻结为通过同一`ReferenceExecutionPlan@1`调用source-native `sample/pdf`，保持完整native tuple；不得作为proposal训练teacher。
- 本child不执行formal scene PT variance。runtime child先把full method装入package/viewer，Linux long训练后再由formal evaluation在同checkpoint、scene、spp和integrator下比较matched neural、analytic-only与source-reference。该项明确递交，不以当前短run结果替代。
