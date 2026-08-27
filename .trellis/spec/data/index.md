# 数据层

数据层拥有 source family、reference program/execution、query stream、offline/live batch producer 与 corpus sink。当前产品 family 是 LayerStack、OpenPBR、MERL、MaterialX 与 MDL；pbrt 仅为外部 crosscheck。MDL 的正式执行与 oracle 隔离合同见 [mdl-reference.md](mdl-reference.md)。

开发与质量合同见 `../project/unified-pipeline.md`。新增 source 必须提供 canonical snapshot、typed editor、reference program/package 和 parity；不得增加平行 collector/manifest/reader。
