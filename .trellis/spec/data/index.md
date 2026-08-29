# Source 与 reference 层

这一层拥有 source family、canonical snapshot、reference program、typed runtime/material payload 与公共`ReferenceBackendCapability/Session`。当前正式 family 是 LayerStack、OpenPBR、MERL、MaterialX 与 MDL；pbrt 仅为外部 crosscheck。online训练合同见 [reference-query.md](reference-query.md)，MDL 的正式执行与 oracle 隔离合同见 [mdl-reference.md](mdl-reference.md)。

项目没有离线训练数据层或 corpus sink。新增 source 必须提供 locator loader、canonical snapshot、typed editor、完整 `prepare/evaluate/sample/pdf` reference 与 parity；不得增加 collector、磁盘 batch、reader 或 family-specific producer。
