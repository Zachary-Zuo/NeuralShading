from __future__ import annotations

from collections.abc import Callable
import importlib
import pkgutil

from .contract import SourceFamilyDefinition, SourceFamilyDescriptor


_FACTORIES: dict[str, Callable[[], SourceFamilyDefinition]] = {}
_DISCOVERED = False


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module("ncls.source_materials.families")
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        definition = getattr(module, "SOURCE_FAMILY_DEFINITION", None)
        if definition is None:
            continue
        if not isinstance(definition, SourceFamilyDefinition):
            raise ValueError(f"{module.__name__}.SOURCE_FAMILY_DEFINITION must be a SourceFamilyDefinition")
        register_source_family(type(definition))
    _DISCOVERED = True


def register_source_family(factory: Callable[[], SourceFamilyDefinition]) -> None:
    definition = factory()
    family_id = definition.descriptor.family_id
    if family_id in _FACTORIES:
        raise ValueError(f"source family {family_id!r} is already registered")
    _FACTORIES[family_id] = factory


def create_source_family(family_id: str) -> SourceFamilyDefinition:
    _discover()
    try:
        return _FACTORIES[family_id]()
    except KeyError as error:
        raise ValueError(f"unsupported source family {family_id!r}") from error


def source_family_descriptors() -> tuple[SourceFamilyDescriptor, ...]:
    _discover()
    return tuple(_FACTORIES[name]().descriptor for name in sorted(_FACTORIES))
