from __future__ import annotations

import importlib

from .contracts import MethodPlugin


_PLUGINS: dict[str, MethodPlugin] | None = None
_PRODUCT_MODULES = ("metal_fused", "nvidia")


def public_method_keys() -> tuple[str, ...]:
    return tuple(plugin.key for plugin in method_plugins())


def method_plugins() -> tuple[MethodPlugin, ...]:
    global _PLUGINS
    if _PLUGINS is None:
        discovered: dict[str, MethodPlugin] = {}
        package = importlib.import_module("ncls.learning.methods")
        for name in _PRODUCT_MODULES:
            module = importlib.import_module(f"{package.__name__}.{name}")
            plugin = getattr(module, "METHOD_PLUGIN", None)
            if not isinstance(plugin, MethodPlugin):
                raise ValueError(f"{module.__name__}.METHOD_PLUGIN must be a MethodPlugin")
            if plugin.key in discovered:
                raise ValueError(f"method plugin {plugin.key!r} is already registered")
            discovered[plugin.key] = plugin
        _PLUGINS = discovered
    return tuple(_PLUGINS[key] for key in sorted(_PLUGINS))


def get_method_plugin(public_key: str) -> MethodPlugin:
    for plugin in method_plugins():
        if plugin.key == public_key:
            return plugin
    raise ValueError(f"unsupported method plugin {public_key!r}")


def reset_method_registry_for_test() -> None:
    global _PLUGINS
    _PLUGINS = None
