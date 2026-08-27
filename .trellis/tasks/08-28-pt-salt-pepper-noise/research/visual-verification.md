# 视觉复核：primary path pool

## 1. 冻结对照

所有主对照固定为 960×540 composite、单 slot 480×540、1024 spp，相机、环境、曝光和源材质 identity 不变。raw EXR 全部 finite。正式 after 产物：

- MDL ceramic：`artifacts/diagnostics/pt-salt-pepper-noise/after-v5-falcor-uniform-primary-path-pool-mdl-ceramic/`
- MDL car paint：`artifacts/diagnostics/pt-salt-pepper-noise/final-mdl-carpaint-960x540-1024spp/`
- OpenPBR car paint：`artifacts/diagnostics/pt-salt-pepper-noise/final-openpbr-carpaint-960x540-1024spp/`

视觉检查结论：

- ceramic 底座原先分散的单像素白点明显减少；球体连续高光与蓝白 tile 边界不漂移。
- MDL car paint 的孤立亮点显著减少；灰色 flake 区域仍保持连续的微结构，不被当成 firefly 清除。
- OpenPBR car paint 的青色主体、底座 flakes 与高光形状保持；残留高分标记主要贴着结构边缘和合法高光。
- 没有使用 radiance/throughput clamp、denoiser、temporal/spatial filter 或 source-family 分支。

## 2. 其它材质审计

R7 capture 全部为 320×240、1024 spp，位于 `artifacts/diagnostics/pt-salt-pepper-noise/audit-*/`。

| 材质 | finite | 视觉分类 |
|---|---|---|
| OpenPBR aluminum | 是 | 金属底座互反射仍有高频结构；未见与旧 ceramic 相同的随机稀疏白点回归。 |
| OpenPBR glass | 是 | 仍有明显的 delta 折射/内部多跳颗粒；属于另一类标准单向 PT 高方差，需独立任务处理。 |
| MERL chrome | 是 | 主要是连续暗金属反射、亮环境反射与几何边缘。 |
| MaterialX textured material | 是 | 纹理斑块与高光连续；local residual 指标会把纹理边误计，但视觉无随机孤点。 |
| LayerStack | 是 | 暗材质上的连续青色高光与接触边；少数 residual 标记落在强边缘。 |

这一步明确区分了“修复已覆盖的 primary-continuation 长尾”和“仍需专门 transport 技术的 specular caustic/内部折射方差”。后者不能作为接口兼容问题或 MIS 权重问题处理。

## 3. Source / package 公共 transport 复核

为避免只证明 `ReferencePathTracer`，任务在
`artifacts/diagnostics/pt-salt-pepper-noise/package-path-smoke-v4/` 生成了一个
probe-only LayerStack ScatteringPackage，并让同一 source material 同时进入 source 与 package
两个真实 scene PT slot。该 probe package 只适配当前 viewer source-state identity 和 parity
buffer binding，不进入项目资产或产品 discovery。

结果：

| slot | transport | status | spp | finite | local residual 标记数 |
|---|---|---|---:|---|---:|
| 0 | source reference PT | ready | 1024 | 是 | 0 |
| 1 | package PT | ready | 1024 | 是 | 0 |

两个 slot 的 local residual p99/p99.9/p99.99 与 top-512 residual sum 在 float32 输出精度内一致，显示图也没有左右结构差异。这验证了 `PackagePathTracer` 的真实 Falcor scene specialization、canonical package `sample/pdf` 调用和共享 primary path pool，而不只是静态检查 shader 文本。

## 4. 质量门

- `conda run -n neural-shading python -m pytest tests/unit -q`：97 passed。
- `scripts/run_falcor_python.ps1 -m pytest tests/gpu -q`：35 passed。
- `conda run -n neural-shading python -m pytest tests/integration -q`：3 passed，1 个 Falcor import 用例在纯 Conda 入口按设计 skipped；真实 GPU suite 已单独覆盖。
- `scripts/build_viewer.ps1 -Configuration Release`：通过。
- Falcor、pbrt-v4、OpenPBR、openpbr-bsdf、glm、MaterialX 六个锁定上游：全部 clean。
