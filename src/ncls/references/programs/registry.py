from __future__ import annotations

import importlib
import pkgutil

from ncls.core.source import SourceFamilyDescriptor
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
    source_contracts = [
        (item.descriptor.family_id, item.descriptor.source_contract_version)
        for item in result
    ]
    if len(set(source_contracts)) != len(source_contracts):
        raise ValueError("source contracts must map to one canonical reference program")
    return tuple(sorted(result, key=lambda item: item.descriptor.program_key))


def get_reference_program(program_key: str) -> ReferenceProgramDefinition:
    for definition in discover_reference_programs():
        if definition.descriptor.program_key == program_key:
            return definition
    raise KeyError(f"unknown reference program {program_key!r}")


def get_reference_program_for_source(
    family_id: str,
    source_contract_version: int,
    *,
    source_descriptor: SourceFamilyDescriptor | None = None,
) -> ReferenceProgramDefinition:
    matches = tuple(
        definition
        for definition in discover_reference_programs()
        if definition.descriptor.family_id == family_id
        and definition.descriptor.source_contract_version == source_contract_version
    )
    if len(matches) != 1:
        raise KeyError(
            "source contract has no unique canonical reference program: "
            f"{family_id}@{source_contract_version}"
        )
    definition = matches[0]
    if source_descriptor is not None:
        if (
            source_descriptor.family_id != family_id
            or source_descriptor.source_contract_version != source_contract_version
        ):
            raise ValueError("source descriptor identity disagrees with lookup key")
        expected = f"{definition.descriptor.program_key}@{definition.descriptor.version}"
        if source_descriptor.reference_program_id != expected:
            raise ValueError(
                "source descriptor reference_program_id disagrees with the canonical "
                f"reference program: expected {expected!r}"
            )
    return definition
