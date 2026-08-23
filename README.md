# NeuralShading

NeuralShading 研究如何把多种保持原生语义的源材质族编译为统一、固定成本、可直接参与实时光照积分的散射实现。源材质可以是纯数学模型、可编辑材质图、程序材质、高分辨率纹理或测量外观；每个材质族保留自己的权威 reference，后续方法按最终散射和呈现结果学习与评测，而不要求 GT 先被分解成某种层参数。

项目已经形成三段可运行闭环：Falcor 数据采集、带 TensorBoard 的 Python 训练/测试、Windows/D3D12 材质查看器。三段只通过带版本的公共合同交换 `MaterialProgram`、`ReferenceDataset` 和 `MethodBundle`；K2 只是一个可替换的历史拟合后端，不是公共接口。当前随机游走 `LayerStackIR` 是第一种已实现的源材质族和 reference，不是所有 GT 的统一内部表示。

当前基线为“精确顶层界面 + 两个 LTC 残差瓣”，方向域 relative-L1 为 median 6.73%、p90 31.20%。粗糙导体基底和深层栈仍有明显长尾，因此表示尚未定稿；接下来的研究重点是降低长尾，再确定结构化网络最终应输出什么。

源材质 reference 已扩展为五个 active package：LayerStack 随机游走、pbrt coated 独立验证、OpenPBR 1.1.1、MERL 测量 BRDF，以及 8 个原生 MaterialX/Poly Haven 4K 纹理材质。它们从 `references/registry.json` 统一发现，但各自保留原始参数、测量表或图/纹理 GT。

## 最短启动路径

创建唯一环境并安装项目：

```powershell
conda env create -f environment.yml
conda run -n neural-shading python -m pip install -r requirements-torch-cu128.txt
conda run -n neural-shading python -m pip install -e .
```

生成小型参考数据并训练 smoke 模型：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data generate-reference `
  --output data\reference-smoke --families 1 --local-states 1 `
  --views 1 --lights 16 --samples-per-replica 16 --max-depth 16

conda run -n neural-shading ncls learn train `
  --dataset data\reference-smoke --run artifacts\runs\smoke `
  --width 8 --steps 2 --batch-size 1 --device cpu
```

导出 realtime `MethodBundle` 并启动 viewer：

```powershell
conda run -n neural-shading ncls bundle export-legacy-ltc-k2 `
  --checkpoint artifacts\runs\smoke\checkpoints\best.pt `
  --run-manifest artifacts\runs\smoke\run_manifest.json `
  --output artifacts\exports\smoke

.\scripts\build_viewer.ps1 -Configuration Release -Run `
  --bundle-root artifacts\exports
```

## 文档入口

- [架构与边界](docs/architecture.md)
- [源材质族与统一表示](docs/material_scope.md)
- [Reference package 固定入口](references/README.md)
- [数据采集](docs/data.md)
- [训练、评测与导出](docs/learning.md)
- [Windows viewer](apps/viewer/README.md)
- [稳定合同](docs/contracts/)
- [分层测试](TESTING.md)
- [仓库边界](docs/repository_policy.md)

Falcor 8.0、pbrt-v4、OpenPBR、openpbr-bsdf、GLM 和 MaterialX 位于被 Git 忽略的 `external/` 独立克隆，固定提交见 `AGENTS.md`。本项目源码不会把未说明修改留在上游工作树。
