# 根仓库边界

## 根仓库保存什么

根 Git 仓库保存由本项目维护、需要审阅和长期复现的内容：

- `schema/`、`teacher/`、`datagen/`、`closures/`、`baselines/`、`model/`、`viewer/` 和 `scripts/` 中的源码；
- `tests/` 与测试配置；
- Conda、PyTorch 和构建环境声明；
- 研究方案、数据合同、实现说明和人工整理的中文报告；
- 体积较小的 JSON 指标与数据清单，例如 HDRI 清单。

## 根仓库不保存什么

以下内容可由源码重新取得或生成，因此只保留在本机：

| 路径或类型 | 原因 |
|---|---|
| `external/` | Falcor 和 pbrt-v4 是独立上游仓库，由固定提交号复现 |
| `data/` | teacher 数据、HDRI 和缓存体积大，且可按清单重新生成或下载 |
| `build/` | CMake、pbrt 探针等本地构建产物 |
| `reports/generated/` | 可重新生成的图像和中间文件 |
| `reports/**/*.npy`、`*.npz`、`*.pt` | 数组、oracle 参数和模型权重属于派生产物 |
| Python 缓存和测试缓存 | 与源码无关 |

报告中的 Markdown 和 JSON 会进入 Git。这样既能保存实验结论和精确指标，又不会把大体积二进制塞进源码历史。

## 上游源码状态

当前两个上游仓库都是干净工作树，本项目没有修改它们：

| 上游 | 固定版本 |
|---|---|
| Falcor | tag 8.0，提交 `9dc819c162b2070335c65060436041690b7937f8` |
| pbrt-v4 | 提交 `5f7a606806a4ac7b939131ded9d7a30ebd02416e` |

项目自己的 teacher、Falcor compute kernel 和交叉验证代码都位于根仓库，不依赖修改上游。若以后必须修补上游，应把补丁文件和应用脚本放入根仓库，并在 `AGENTS.md` 记录原因；不能只在 `external/` 中留下未说明的本地改动。

可以随时用下面的命令核对：

```powershell
git -C external/Falcor status --short
git -C external/pbrt-v4 status --short
```
