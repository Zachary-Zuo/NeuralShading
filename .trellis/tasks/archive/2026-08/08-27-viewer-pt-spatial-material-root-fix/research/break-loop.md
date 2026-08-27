# Bug Analysis：PT 空间材质被最粗 mip 平均化

## 1. Root Cause Category

- **主类别：E - Implicit Assumption**：把 Falcor `cameraV` 的长度误当成纯 image-plane slope；实际上 `cameraU/V/W` 共享 `focalDistance` 尺度，只有比值才与投影有关。
- **伴随类别：B - Cross-Layer Contract**：camera basis、世界空间 ray cone、triangle UV/world Jacobian、normalized UV derivative 和 texture-owned dimension 之间没有写明单位边界。
- **伴随类别：D - Test Coverage Gap**：既有测试验证资源、ABI、方向求值与 slot 生命周期，没有验证真实 scene surface 的空间字段和 mip 选择。

## 2. Why Fixes Failed

1. **把低 spp 当主因**：提高到 512 spp 后平均色仍存在，只排除了噪声，不能区分 UV、LOD 与纹理绑定。
2. **只看加载日志与 source flag**：它们证明纹理存在且启用，但最粗 mip 同样会成功加载和采样，因此与错误现象相容。
3. **只看 walnut**：天然低对比纹理让最粗 mip 与低频外观难区分；换成带大尺度接缝的 denim 才确认空间 lookup 退化。
4. **只比较 PT 输出**：source 与 neural 同时失败容易误指向场景/资产；同一 neural package 的 deferred 保留纹理后，才把范围缩到 PT 公共 surface 链。
5. **最终有效手段**：raw first-hit probe 绕过 BSDF、灯光与 tone mapping，先证实 UV 非常量，再把 LOD 拆为 Jacobian、ray-cone width 和 normal projection 三项，一次定位到 camera scale。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | reference/package PT 统一调用 `PathSurface.slang`，移除重复 hit/frame/footprint 逻辑 | DONE |
| P0 | Test Coverage | GPU oracle 验证 camera basis scale invariance、resolution/distance/angle 单调性与有限 fallback | DONE |
| P0 | Documentation | `.trellis/spec/viewer/path-surface.md` 写明签名、单位、fallback 与错误矩阵 | DONE |
| P1 | Visual Evidence | 每次 spatial viewer 收尾同时保留明显低频结构与真实资产 local-light capture | DONE |
| P1 | Review | cross-layer guide 增加 GPU differential 单位/共同尺度检查 | DONE |
| P2 | Runtime diagnostics | 后续若再次扩展 geometry 类型，再评估把 raw surface probe 做成常驻 headless diagnostic；本次不扩张 capture ABI | NOT REQUIRED |

## 4. Systematic Expansion

- **Similar Issues**：camera/light basis、world/object transform、ray differential、raster derivative 与 texture-size LOD 都可能携带隐藏尺度；所有“方向归一化后正常、面积/footprint 异常”的问题都应检查共同 scale。
- **Design Improvement**：只允许 renderer 公共层拥有 scene surface construction；source family 与 method backend 从该边界之后才分开。
- **Process Improvement**：部署验收必须区分 lifecycle evidence 与 semantic evidence。`ready`、hash parity、平均颜色和 tone-mapped screenshot 不能替代真实空间字段/外观验证。
- **Knowledge Gap**：ray-cone texture-independent LOD 的单位过去只存在于上游注释，没有进入项目规范；现已补为可执行合同。

## 5. Knowledge Capture

- [x] 新增 `.trellis/spec/viewer/path-surface.md` 并更新 viewer index。
- [x] 更新 `cross-layer-thinking-guide.md` 与 `code-reuse-thinking-guide.md`。
- [x] 更新 `docs/viewer_spec.md` 与 `apps/viewer/README.md`。
- [x] 新增 unit/GPU 回归和真实 capture 证据。
- [x] 本仓库没有 `src/templates/markdown/spec/` 模板树；它是 Trellis consumer 项目，`.trellis/spec/` 即项目规范源，因此无可同步模板。
