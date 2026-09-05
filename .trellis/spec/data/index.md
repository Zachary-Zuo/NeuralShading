# Source 与 reference 层

这一层拥有 source family、canonical snapshot、reference program、typed runtime/material payload、公共 `ReferenceBackendCapability/Session`，以及训练共享的 host pipeline、GPU residency、reference scheduler 与 `OnlineDataSession` 合同。当前正式 family 是 LayerStack、OpenPBR、MERL、MaterialX 与 MDL；pbrt 仅为外部 crosscheck。online query 见 [reference-query.md](reference-query.md)，通用训练数据调度见 [online-pipeline.md](online-pipeline.md)，MDL 的正式执行与 oracle 隔离见 [mdl-reference.md](mdl-reference.md)。

项目没有离线训练数据层或 corpus sink。新增 source 必须提供 locator loader、canonical snapshot、typed editor、完整 `prepare/evaluate/sample/pdf` reference 与 parity；新增 method 只通过 Method 数据接口 声明数据需求和 source adapter。不得增加 collector、磁盘 batch、reader、family-specific producer 或 method-specific data loop。
