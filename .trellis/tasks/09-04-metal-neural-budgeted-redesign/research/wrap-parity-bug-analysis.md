# Bug Analysis：Python wrap bilinear oracle 与 GPU sampler 不一致

## 1. Root Cause Category

- **Category**：B（跨层合同）+ D（测试覆盖缺口）。
- **Specific Cause**：Python oracle把“先对UV取模”等同于wrap address mode，但`grid_sample(..., padding_mode="zeros")`仍会在bilinear footprint跨越0/1时把越界邻居置零。GPU `SamplerState.Wrap`会对每个邻居分别wrap。现有GPU fixture只使用纹理内部`uv=(0.371,0.619)`，因此两个实现看起来一致；最终package的`uv=(0,0)`才提供区分证据。

## 2. Why Fixes Failed

1. 合成fixture通过：它证明FP16权重、SNORM8上传、prepare/evaluate与state pack在内部坐标一致，但没有覆盖address mode边界。
2. 初次最终package validator：先被错误的public/implementation key检查挡住，修正后又因DDS decoder返回只读NumPy view挡在资源上传；两处都是validator自身缺口，不是数值根因。
3. 显式复制DDS mip后：最终Slang数值稳定复现约`0.103`最大绝对差，结合`access=[1,1,0,0,1,0,wrap]`把假设收敛到`uv=0`边界reconstruction，而不是权重、layout或mip选择。

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific Action | Status |
|---|---|---|---|
| P0 | Architecture | CPU oracle显式计算bilinear四邻居并分别应用wrap/clamp | DONE |
| P0 | Test Coverage | unit使用2×2非周期纹理断言`uv=0`的wrap均值与clamp角点不同 | DONE |
| P0 | GPU Coverage | Metal budgeted Python↔Slang fixture固定改用`uv=0` | DONE |
| P0 | Deployment | 最终package validator直接读取manifest、FP16 blob、DDS与sampler执行冻结witness | DONE |
| P1 | Documentation | cross-layer guide与learning合同要求边界witness | DONE |

## 4. Systematic Expansion

- **Similar Issues**：所有用`frac(UV)`模拟wrap、但把reconstruction交给zero padding的CPU/reference路径；environment CDF与radiance lookup使用不同filter/address mode也属于同类。
- **Design Improvement**：address mode必须属于“坐标变换 + reconstruction support”整体合同，不能只作为坐标预处理字段。
- **Process Improvement**：部署parity要同时覆盖内部随机坐标和能区分address mode的边界坐标；最终package自加载是发布门，合成fixture只是前置门。

## 5. Knowledge Capture

- [x] 更新`.trellis/spec/guides/cross-layer-thinking-guide.md`。
- [x] 更新`.trellis/spec/learning/online-training.md`。
- [x] 在`research/protocol-freeze.md`登记identity变化和重跑范围。
- [x] 增加CPU边界unit、GPU边界fixture与最终package validator。
- [x] 项目没有`src/templates/markdown/spec/`或其他spec template副本，因此无可同步模板；`.trellis/spec/`是本仓库唯一规则源。
