"""仅供 slangpy-torch 本机构建：规避 Torch 在中文 Windows 上按 OEM 解码 cl 输出。"""

import torch.utils.cpp_extension

torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
