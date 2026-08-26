from __future__ import annotations

from pathlib import Path


class SlangModuleSession:
    """方法无关的 SlangPy module loader；数学与 callable 身份由 module 自己拥有。"""

    def __init__(self, module_path: Path | str) -> None:
        import slangpy as spy

        self.path = Path(module_path).resolve()
        self.device = spy.create_device(type=spy.DeviceType.cuda)
        self.module = spy.Module.load_from_file(self.device, str(self.path))
