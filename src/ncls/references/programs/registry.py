from __future__ import annotations

import importlib
import pkgutil

from ncls.core.scattering import ReferenceProgramDefinition


def discover_reference_programs() -> tuple[ReferenceProgramDefinition, ...]:
    package = importlib.import_module("ncls.references.programs")
    result = []
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name in {"base", "registry"} or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        definition = getattr(module, "REFERENCE_PROGRAM_DEFINITION", None)
        if definition is not None:
            if not isinstance(definition, ReferenceProgramDefinition):
                raise TypeError(f"{module.__name__}.REFERENCE_PROGRAM_DEFINITION has the wrong type")
            result.append(definition)
    keys = [item.descriptor.program_key for item in result]
    if len(set(keys)) != len(keys):
        raise ValueError("reference program keys must be unique")
    return tuple(sorted(result, key=lambda item: item.descriptor.program_key))


def get_reference_program(program_key: str) -> ReferenceProgramDefinition:
    for definition in discover_reference_programs():
        if definition.descriptor.program_key == program_key:
            return definition
    raise KeyError(f"unknown reference program {program_key!r}")
