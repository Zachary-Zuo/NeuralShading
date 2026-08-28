# 实时材质编译

目标运行时是静态有界neural scattering program：`prepare()`获取/过滤latent并复用view-conditioned state，`evaluate(wo,wi)`直接输出线性`f`，需要方向采样时提供匹配的`sample/pdf`。source snapshot、program runtime、material asset与package身份分离。解析closure可作为reference core或proposal，不是目标输出词汇。

所有source reference与neural runtime共享canonical `prepare/evaluate/sample/pdf`合同，但各自保留backend实现和交付身份。source family先从原生locator构造`SourceSnapshot`，reference program再产出typed runtime/material payload；统一dispatcher只调用合同，不按LayerStack、MaterialX、MDL等family分支。

正式训练在GPU上在线生成query。evaluator target就是source `evaluate().f`；sample/PDF接口用于reference transport和proposal正确性验证。NVIDIA matched sampler不模仿source sampler，而是从当前learned proposal取样，用learned evaluator的`luminance(f)·|cosθi|`形成forward-KL目标。

原生MDL沿同一路径接入：项目bridge用锁定MDL SDK编译program，generated target code作为typed Slang source module注入canonical `mdl.slang` backend；argument block、RO data、2D/3D texture和sampler都由通用binder绑定。falcor2只作为隔离validation oracle，不进入正式reference或训练路径。

项目不把query持久化为训练数据，也没有offline replay或旧数据兼容入口。若未来确需固定诊断样本，必须另行定义只用于验证的新artifact合同，不能重新成为训练GT路径。
