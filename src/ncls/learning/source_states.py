from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ncls.core.identity import sha256_json
from ncls.core.source import SourceFamilyDefinition, SourceSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ExpandedSourceStates:
    snapshots: tuple[SourceSnapshot, ...]
    identity: str
    recipe_schema: str
    base_snapshot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.snapshots or not self.base_snapshot_ids or not self.recipe_schema:
            raise ValueError("expanded source-state set is incomplete")
        if len(self.identity) != 64:
            raise ValueError("expanded source-state identity is invalid")


def _project_path(value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source-state recipe paths must be project-relative")
    result = (PROJECT_ROOT / relative).resolve()
    result.relative_to(PROJECT_ROOT)
    return result


def _identity_states(
    family: SourceFamilyDefinition,
    snapshots: Sequence[SourceSnapshot],
    recipe: Mapping[str, Any],
) -> ExpandedSourceStates:
    del family
    if recipe:
        raise ValueError("identity source-state recipe accepts no fields")
    values = tuple(snapshots)
    identity = sha256_json(
        {
            "schema": "ncls.identity-source-states@1",
            "source_snapshot_ids": [snapshot.snapshot_id for snapshot in values],
        }
    )
    return ExpandedSourceStates(
        values,
        identity,
        "ncls.identity-source-states@1",
        tuple(snapshot.snapshot_id for snapshot in values),
    )


def _mdl_metal_states(
    family: SourceFamilyDefinition,
    snapshots: Sequence[SourceSnapshot],
    recipe: Mapping[str, Any],
) -> ExpandedSourceStates:
    from ncls.source_materials.families.mdl import MdlFamilyDefinition
    from ncls.source_materials.mdl import MdlMaterialSource
    from ncls.source_materials.mdl_metal import (
        MdlMetalRegistry,
        MdlMetalStatePool,
        MdlMetalTypedStateRecipe,
    )

    if not isinstance(family, MdlFamilyDefinition):
        raise ValueError("MDL Metal typed-state recipe requires the MDL source family")
    allowed = {
        "registry",
        "recipe_id",
        "split",
        "seed",
        "states_per_export",
        "responsibilities",
        "default_weight",
        "boundary_weight",
    }
    if set(recipe) != allowed:
        raise ValueError(
            f"MDL Metal typed-state recipe fields must be exactly {sorted(allowed)}"
        )
    registry = MdlMetalRegistry.load(_project_path(recipe["registry"]))
    sources = tuple(MdlMaterialSource.from_snapshot(snapshot) for snapshot in snapshots)
    roots = {source.module_root for source in sources}
    if len(roots) != 1:
        raise ValueError("one Metal typed-state pool requires one source pack root")
    selected = tuple(
        registry.resolve_exact_locator(source.module, source.export) for source in sources
    )
    typed_recipe = MdlMetalTypedStateRecipe(
        str(recipe["recipe_id"]),
        str(recipe["split"]),
        int(recipe["seed"]),
        int(recipe["states_per_export"]),
        tuple(str(value) for value in recipe["responsibilities"]),
        float(recipe["default_weight"]),
        float(recipe["boundary_weight"]),
    )
    pool = MdlMetalStatePool.generate(
        registry,
        family,
        typed_recipe,
        module_root=next(iter(roots)),
        exports=selected,
    )
    return ExpandedSourceStates(
        pool.snapshots,
        pool.identity,
        "ncls.mdl-metal-typed-state-recipe@1",
        tuple(snapshot.snapshot_id for snapshot in snapshots),
    )


_RECIPES: dict[
    str,
    Callable[
        [SourceFamilyDefinition, Sequence[SourceSnapshot], Mapping[str, Any]],
        ExpandedSourceStates,
    ],
] = {
    "ncls.identity-source-states@1": _identity_states,
    "ncls.mdl-metal-typed-state-recipe@1": _mdl_metal_states,
}


def expand_source_states(
    family: SourceFamilyDefinition,
    snapshots: Sequence[SourceSnapshot],
    recipe: Mapping[str, Any] | None,
) -> ExpandedSourceStates:
    values = tuple(snapshots)
    if not values:
        raise ValueError("source-state expansion requires base snapshots")
    config = {} if recipe is None else dict(recipe)
    schema = str(config.pop("schema", "ncls.identity-source-states@1"))
    try:
        implementation = _RECIPES[schema]
    except KeyError as error:
        raise ValueError(f"unknown typed source-state recipe {schema!r}") from error
    result = implementation(family, values, config)
    for snapshot in result.snapshots:
        family.validate_snapshot(snapshot)
    return result


__all__ = ["ExpandedSourceStates", "expand_source_states"]
