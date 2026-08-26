from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from ncls.learning.method import MethodDefinition, MethodDescriptor


_DEFINITIONS: dict[str, MethodDefinition] | None = None


def _load_module(module: ModuleType, definitions: dict[str, MethodDefinition]) -> None:
    definition = getattr(module, "METHOD_DEFINITION", None)
    if definition is None:
        return
    if not isinstance(definition, MethodDefinition):
        raise ValueError(f"{module.__name__}.METHOD_DEFINITION must be a MethodDefinition")
    key = definition.descriptor.method_key
    if key in definitions:
        raise ValueError(f"method {key!r} is already registered")
    definitions[key] = definition


def _discover() -> dict[str, MethodDefinition]:
    package = importlib.import_module("ncls.learning.methods")
    result: dict[str, MethodDefinition] = {}
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_") or info.name == "registry":
            continue
        _load_module(importlib.import_module(f"{package.__name__}.{info.name}"), result)
    return result


def method_definitions() -> tuple[MethodDefinition, ...]:
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = _discover()
    return tuple(_DEFINITIONS[name] for name in sorted(_DEFINITIONS))


def method_descriptors() -> tuple[MethodDescriptor, ...]:
    return tuple(definition.descriptor for definition in method_definitions())


def get_method(method_key: str) -> MethodDefinition:
    for definition in method_definitions():
        if definition.descriptor.method_key == method_key:
            return definition
    raise ValueError(f"unsupported method {method_key!r}")


def inject_method_for_test(definition: MethodDefinition) -> None:
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = _discover()
    key = definition.descriptor.method_key
    if key in _DEFINITIONS:
        raise ValueError(f"method {key!r} is already registered")
    _DEFINITIONS[key] = definition


def reset_method_registry_for_test() -> None:
    global _DEFINITIONS
    _DEFINITIONS = None
