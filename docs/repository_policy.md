# 根仓库边界

## 根仓库保存什么

根 Git 仓库只保存由本项目维护、需要审阅和长期复现的内容：

- `src/ncls/` 中的正式 Python package；
- `shaders/ncls/` 中的公共合同、随机游走 reference 和拟合后端；
- `references/` 中的 reference registry、package 说明、轻量 manifest 和项目 adapter；
- `apps/viewer/`、`patches/` 与构建/benchmark 脚本；
- `tests/unit/`、`tests/gpu/`、`tests/integration/`；
- 环境声明、显式配置、中文文档和人工整理的实验结论；
- 体积较小的 JSON 指标与资产清单。

根仓库不维持迁移前的 `schema/`、`datagen/`、固定 K2 packet、泛化 `model/` 或旧 Python lookup viewer。旧数据只通过 `ncls.core.material.legacy_v0` 和 `ncls.data.legacy_v0` 的一次性 reader 转换，不保留第二套 writer。

## 根仓库不保存什么

| 路径或类型 | 原因 |
|---|---|
| `external/` | Falcor、pbrt-v4、OpenPBR、openpbr-bsdf、GLM 和 MaterialX 是固定提交的独立上游克隆 |
| `data/source-materials/` | 原始纹理、测量表、场景和其他源材质大资源；由 `references/` 中的 manifest 锁定 |
| `data/reference-responses/` | reference 查询后生成的响应数据，可由源材质、reference 和采集配置复现 |
| `data/` 其他内容 | 迁移前参考数据、HDRI 和缓存体积大，不进入根 Git |
| `build/` | CMake、viewer 和 pbrt probe 构建产物 |
| `artifacts/` | training run、checkpoint、MethodBundle、capture、benchmark 和临时实验输出 |
| `reports/generated/` | 可重新生成的图像和中间文件 |
| `reports/**/*.npy`、`*.npz`、`*.pt` | 数组、逐样本拟合参数和模型权重属于派生产物 |
| Python/pytest/ruff 缓存 | 与源码无关 |

历史报告中的 Markdown 和轻量 JSON 可以进入 Git，但必须由 `reports/README.md` 标明其接口年代；原始字段名不会成为当前代码合同。

## 上游源码状态

| 上游 | 固定版本 |
|---|---|
| Falcor | tag 8.0，提交 `9dc819c162b2070335c65060436041690b7937f8` |
| pbrt-v4 | 提交 `5f7a606806a4ac7b939131ded9d7a30ebd02416e` |
| ASWF OpenPBR | tag v1.1.1，提交 `f8d6d947dfae4c9b599965a86c22826ea7a8dbfb` |
| Adobe openpbr-bsdf | 提交 `9edf806740d2140846d9bef76e4342fc458e2ef5` |
| GLM | tag 1.0.1，提交 `0af55ccecd98d4e5a8d1fad7de25ba429d60e863` |
| MaterialX | tag v1.39.4，提交 `270b5cf2ae2be24a3b6ef4b0569f1c93038dda1d` |

项目源码不依赖持久修改上游。Windows viewer 通过 `patches/falcor-viewer-overlay.patch` 临时加入 Falcor Samples；`scripts/build_viewer.ps1` 在构建前后验证锁定提交和干净工作树，并在 `finally` 中反向应用补丁。

若未来必须修补上游，先把补丁和应用脚本提交到根仓库，并在 `AGENTS.md` 说明原因。不能只在 `external/` 留下本地修改。

```powershell
git -C external\Falcor status --short
git -C external\pbrt-v4 status --short
git -C external\OpenPBR status --short
git -C external\openpbr-bsdf status --short
git -C external\glm status --short
git -C external\MaterialX status --short
```
