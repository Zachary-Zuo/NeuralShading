from __future__ import annotations

import importlib

from ncls.learning.method import Method


_METHODS: dict[str, Method] | None = None
_PRODUCT_MODULES = ("metal", "nvidia")


def method_keys() -> tuple[str, ...]:
    return tuple(plugin.key for plugin in registered_methods())


def registered_methods() -> tuple[Method, ...]:
    global _METHODS
    if _METHODS is None:
        discovered: dict[str, Method] = {}
        package = importlib.import_module("ncls.learning.methods")
        for name in _PRODUCT_MODULES:
            module = importlib.import_module(f"{package.__name__}.{name}.method")
            plugin = getattr(module, "METHOD", None)
            if not isinstance(plugin, Method):
                raise ValueError(f"{module.__name__}.METHOD must be a Method")
            if plugin.key in discovered:
                raise ValueError(f"method {plugin.key!r} is already registered")
            discovered[plugin.key] = plugin
        _METHODS = discovered
    return tuple(_METHODS[key] for key in sorted(_METHODS))


def get_method(public_key: str) -> Method:
    for plugin in registered_methods():
        if plugin.key == public_key:
            return plugin
    raise ValueError(f"unsupported method {public_key!r}")


def reset_method_registry_for_test() -> None:
    global _METHODS
    _METHODS = None
