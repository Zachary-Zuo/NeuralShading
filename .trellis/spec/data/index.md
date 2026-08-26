# 数据层

数据层拥有 source family、reference program/execution、query stream、offline/live batch producer 与 corpus sink。四个产品 family 是 LayerStack、OpenPBR、MERL、MaterialX；pbrt 仅为外部 crosscheck。

开发与质量合同见 `../project/unified-pipeline.md`。新增 source 必须提供 canonical snapshot、typed editor、reference program/package 和 parity；不得增加平行 collector/manifest/reader。
