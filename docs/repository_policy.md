# 根仓库边界

## 根仓库保存什么

根 Git 仓库只保存由本项目维护、需要审阅和长期复现的内容：

- `src/ncls/` 中的正式 Python package；
- `shaders/ncls/` 中的公共合同、随机游走 reference、neural material backend 和解析基线；
- `references/` 中的 reference registry、package 说明、轻量 manifest 和项目 adapter；
- `apps/viewer/`、`patches/` 与构建/benchmark 脚本；
- `tests/unit/`、`tests/gpu/`、`tests/integration/`；
- 环境声明、显式配置、中文文档和人工整理的实验结论；
- 作为稳定验收合同的阈值、资产清单和人工整理结论。

根仓库不维持迁移前的 `schema/`、`datagen/`、通用 backend packet、泛化 `model/` 或旧 Python lookup viewer。数据层不保留旧 reader、转换器或第二套 writer；所有 reference response 从锁定输入重新生成。

## 根仓库不保存什么

被 Git 忽略的大文件目录只有以下职责，不能交叉使用：

```text
assets/
  source-materials/
    merl-brdf/v1/                 原始测量表与发布归档
    materialx-polyhaven/v1/       原生 .mtlx、纹理和导入记录
  viewer/
    scenes/studio-v1/             viewer 固定几何
    environments/polyhaven-1k/    viewer/HDRI 原始输入
data/
  reference-responses/            只允许 .h5/.hdf5 ReferenceDataset
artifacts/
  legacy-data/                    迁移前数组与旧实验输出，仅供追溯
  caches/                         可再生成 cache
  ...                             capture、报告、模型和 MethodBundle
external/                         固定提交的第三方源码
```

| 路径或类型 | 原因 |
|---|---|
| `external/` | Falcor、pbrt-v4、OpenPBR、openpbr-bsdf、GLM 和 MaterialX 是固定提交的独立上游克隆 |
| `assets/source-materials/` | 原始材质定义、纹理和测量表；由 `references/` 中的 manifest 锁定 |
| `assets/viewer/` | viewer 固定场景和环境图等运行输入；由版本化 preset/manifest 锁定 |
| `data/reference-responses/` | reference 查询后生成的响应数据，可由源材质、reference 和采集配置复现 |
| `data/` 其他内容 | 不允许；非 HDF5 派生数据进入 `artifacts/`，原始输入进入 `assets/` |
| `build/` | CMake、viewer 和 pbrt probe 构建产物 |
| `artifacts/` | training run、checkpoint、MethodBundle、capture、benchmark、验证报告和临时实验输出 |
| `reports/` | 正确性验证结果、运行摘要和实验报告均可由代码、配置与锁定输入再生，不在根仓库持久化 |
| `*.npy`、`*.npz`、`*.pt` | 数组、逐样本拟合参数和模型权重属于派生产物 |
| Python/pytest/ruff 缓存 | 与源码无关 |

稳定结论写入对应的中文设计文档，验收门槛写入版本化配置；某次运行的 Markdown/JSON 报告统一输出到 `artifacts/`。报告中的原始字段名和某个 backend 的临时状态布局都不是公共接口。

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
