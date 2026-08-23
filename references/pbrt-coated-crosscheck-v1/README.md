# ncls.pbrt-coated-crosscheck@1

## 角色

这是 `ncls.layer-stack@1` 的独立交叉验证 reference，不是训练 GT 的默认实现，也不是任意 N 层材质系统。

锁定上游为：

- pbrt-v4 commit `5f7a606806a4ac7b939131ded9d7a30ebd02416e`；
- 上游源码位置 `external/pbrt-v4`；
- 项目 adapter 位置 `tools/reference/pbrt_probe/` 和 `tools/reference/pbrt_compare.py`。

## 当前覆盖

当前 probe 同时实例化 pbrt-v4 `CoatedDiffuseBxDF` 和 `CoatedConductorBxDF`：一个 rough dielectric top、一个 diffuse 或粗糙各向异性 conductor bottom，以及二者之间的均匀 slab。它验证界面透射、彩色基底反射、介质吸收/散射、多次反射和方位方向切片的组合语义。

8192 samples、2 batches、max depth 32 的 smoke suite 中，diffuse-clear、conductor-clear、conductor-absorbing 和 conductor-scattering 四组的总体 mean 相对误差为 0.414%，max 为 2.554%；这是有限样本交叉验证，不把 pbrt 采样值当作无噪声解析真值。不会为了得到 `N=3/4/...` 而递归嵌套 pbrt `LayeredBxDF`。
