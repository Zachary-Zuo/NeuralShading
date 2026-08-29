from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ncls.core.identity import sha256_json
from ncls.core.source import SourceEditOperation, SourceEditPatch, SourceSnapshot
from ncls.source_materials.families.mdl import MdlFamilyDefinition


MDL_METAL_REGISTRY_SCHEMA = "ncls.mdl-metal-opaque-registry@1"
MDL_METAL_EXPECTED_COUNTS = {
    "authored_exports": 837,
    "opaque_exports": 692,
    "rejected_cutout_exports": 145,
    "opaque_graphs": 178,
    "opaque_texture_sets": 52,
    "parameter_schemas": 64,
}
PARAMETER_RESPONSIBILITIES = (
    "coordinates",
    "frame",
    "metal-core",
    "finish-microstructure",
    "aging-contamination",
    "coating-composite",
)


def _require_sha256(label: str, value: object) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True)
class MdlMetalExport:
    export_id: str
    exact_locator: Mapping[str, Any]
    graph_id: str
    parameter_schema_id: str
    texture_set_id: str
    recipe_id: str
    metal: str
    finish: str
    parameters: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        _require_sha256("Metal export_id", self.export_id)
        for label, value in (
            ("graph_id", self.graph_id),
            ("parameter_schema_id", self.parameter_schema_id),
            ("texture_set_id", self.texture_set_id),
            ("recipe_id", self.recipe_id),
        ):
            _require_sha256(label, value)
        locator = dict(self.exact_locator)
        if locator.get("kind") != "mdl-export" or not str(locator.get("module", "")).startswith(
            "::vMaterials_2::Metal::"
        ):
            raise ValueError("Metal export locator is not an exact vMaterials 2 Metal locator")
        if not str(locator.get("export", "")).startswith(str(locator["module"]) + "::"):
            raise ValueError("Metal export locator must retain the exact overload signature")
        if not self.metal or not self.finish or not self.parameters:
            raise ValueError("Metal export typed semantics are incomplete")
        for parameter in self.parameters:
            if parameter.get("responsibility") not in PARAMETER_RESPONSIBILITIES:
                raise ValueError("Metal parameter has an unknown semantic responsibility")


class MdlMetalRegistry:
    """严格加载全量 opaque vMaterials 2 Metal registry。"""

    def __init__(self, payload: Mapping[str, Any], *, path: Path | None = None) -> None:
        value = dict(payload)
        if value.get("schema") != MDL_METAL_REGISTRY_SCHEMA:
            raise ValueError("unsupported MDL Metal registry schema")
        identity = _require_sha256("Metal registry identity", value.pop("identity", ""))
        if sha256_json(value) != identity:
            raise ValueError("MDL Metal registry identity mismatch")
        counts = {str(name): int(count) for name, count in value.get("counts", {}).items()}
        if counts != MDL_METAL_EXPECTED_COUNTS:
            raise ValueError(
                f"MDL Metal registry counts changed: expected {MDL_METAL_EXPECTED_COUNTS}, got {counts}"
            )
        tables = value.get("tables")
        if not isinstance(tables, Mapping):
            raise ValueError("MDL Metal registry tables are missing")
        graphs = self._indexed_table(tables, "graphs", counts["opaque_graphs"])
        schemas = self._indexed_table(tables, "parameter_schemas", counts["parameter_schemas"])
        texture_sets = self._indexed_table(
            tables, "texture_sets", counts["opaque_texture_sets"]
        )
        recipes = self._indexed_table(tables, "recipes", None)
        exports = tuple(
            MdlMetalExport(
                str(record["export_id"]),
                record["exact_locator"],
                str(record["graph_id"]),
                str(record["parameter_schema_id"]),
                str(record["texture_set_id"]),
                str(record["recipe_id"]),
                str(record["metal"]),
                str(record["finish"]),
                tuple(record["parameters"]),
            )
            for record in value.get("opaque_exports", ())
        )
        if len(exports) != counts["opaque_exports"]:
            raise ValueError("MDL Metal opaque export table has the wrong size")
        if len({record.export_id for record in exports}) != len(exports):
            raise ValueError("MDL Metal export identities are not unique")
        locator_keys = {
            (record.exact_locator["module"], record.exact_locator["export"])
            for record in exports
        }
        if len(locator_keys) != len(exports):
            raise ValueError("MDL Metal exact locators are not unique")
        for record in exports:
            if record.graph_id not in graphs:
                raise ValueError("MDL Metal export references an unknown graph")
            if record.parameter_schema_id not in schemas:
                raise ValueError("MDL Metal export references an unknown parameter schema")
            if record.texture_set_id not in texture_sets:
                raise ValueError("MDL Metal export references an unknown texture set")
            if record.recipe_id not in recipes:
                raise ValueError("MDL Metal export references an unknown recipe")
        rejected = tuple(value.get("rejected_cutout_exports", ()))
        if len(rejected) != counts["rejected_cutout_exports"]:
            raise ValueError("MDL Metal cutout rejection table has the wrong size")
        if any(
            item.get("reason") != "geometry.cutout_opacity"
            or not item.get("exact_locator")
            for item in rejected
        ):
            raise ValueError("MDL Metal cutout entries must fail closed with an exact locator")
        for texture_set in texture_sets.values():
            slots = tuple(texture_set.get("slots", ()))
            if len(slots) > 9:
                raise ValueError("MDL Metal texture set exceeds the frozen nine-slot bound")
            for slot in slots:
                _require_sha256("Metal texture source_sha256", slot.get("source_sha256"))
                if not slot.get("roles") or not slot.get("channels"):
                    raise ValueError("MDL Metal texture slot has no role/channel contract")
        self.payload = {**value, "identity": identity}
        self.identity = identity
        self.path = None if path is None else path.resolve()
        self.exports = exports
        self.graphs = graphs
        self.parameter_schemas = schemas
        self.texture_sets = texture_sets
        self.recipes = recipes
        self.rejected_cutout_exports = rejected
        self._by_export_id = {record.export_id: record for record in exports}
        self._by_locator = {
            (str(record.exact_locator["module"]), str(record.exact_locator["export"])): record
            for record in exports
        }

    @staticmethod
    def _indexed_table(
        tables: Mapping[str, Any], name: str, expected_count: int | None
    ) -> dict[str, Mapping[str, Any]]:
        values = tuple(tables.get(name, ()))
        if expected_count is not None and len(values) != expected_count:
            raise ValueError(f"MDL Metal {name} table has the wrong size")
        result: dict[str, Mapping[str, Any]] = {}
        for item in values:
            item_id = _require_sha256(f"Metal {name} id", item.get("id"))
            if item_id in result:
                raise ValueError(f"MDL Metal {name} identities are not unique")
            result[item_id] = item
        if not result:
            raise ValueError(f"MDL Metal {name} table is empty")
        return result

    @classmethod
    def load(cls, path: Path) -> "MdlMetalRegistry":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(payload, path=path)

    def export(self, export_id: str) -> MdlMetalExport:
        try:
            return self._by_export_id[export_id]
        except KeyError as error:
            raise ValueError(f"unknown opaque Metal export {export_id!r}") from error

    def resolve_exact_locator(self, module: str, export: str) -> MdlMetalExport:
        try:
            return self._by_locator[(module, export)]
        except KeyError as error:
            raise ValueError("unknown, missing or cutout Metal export") from error

    def selected_exports(
        self,
        *,
        metals: Iterable[str] | None = None,
        finishes: Iterable[str] | None = None,
        recipe_ids: Iterable[str] | None = None,
    ) -> tuple[MdlMetalExport, ...]:
        metal_set = None if metals is None else set(metals)
        finish_set = None if finishes is None else set(finishes)
        recipe_set = None if recipe_ids is None else set(recipe_ids)
        result = tuple(
            record
            for record in self.exports
            if (metal_set is None or record.metal in metal_set)
            and (finish_set is None or record.finish in finish_set)
            and (recipe_set is None or record.recipe_id in recipe_set)
        )
        if not result:
            raise ValueError("Metal registry selection is empty or incompatible")
        return result


@dataclass(frozen=True)
class MdlMetalTypedStateRecipe:
    recipe_id: str
    split: str
    seed: int
    states_per_export: int
    responsibilities: tuple[str, ...] = (
        "metal-core",
        "finish-microstructure",
        "aging-contamination",
        "coating-composite",
    )
    default_weight: float = 0.2
    boundary_weight: float = 0.3

    def __post_init__(self) -> None:
        if not self.recipe_id or self.split not in {"train", "validation"}:
            raise ValueError("Metal typed-state recipe identity/split is invalid")
        if self.seed < 0 or self.states_per_export < 1:
            raise ValueError("Metal typed-state recipe seed/capacity is invalid")
        if not set(self.responsibilities).issubset(PARAMETER_RESPONSIBILITIES):
            raise ValueError("Metal typed-state recipe references an unknown responsibility")
        if (
            self.default_weight < 0.0
            or self.boundary_weight < 0.0
            or self.default_weight + self.boundary_weight > 1.0
        ):
            raise ValueError("Metal typed-state recipe weights are invalid")

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.mdl-metal-typed-state-recipe@1",
                "recipe_id": self.recipe_id,
                "split": self.split,
                "seed": self.seed,
                "states_per_export": self.states_per_export,
                "responsibilities": list(self.responsibilities),
                "default_weight": self.default_weight,
                "boundary_weight": self.boundary_weight,
            }
        )


@dataclass(frozen=True)
class MdlMetalStatePool:
    registry_identity: str
    recipe_identity: str
    split: str
    snapshots: tuple[SourceSnapshot, ...]
    base_export_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256("Metal state-pool registry identity", self.registry_identity)
        _require_sha256("Metal state-pool recipe identity", self.recipe_identity)
        if not self.snapshots or len(self.snapshots) != len(self.base_export_ids):
            raise ValueError("Metal state pool is empty or loses base-export provenance")
        if len({snapshot.snapshot_id for snapshot in self.snapshots}) != len(self.snapshots):
            raise ValueError("Metal state pool contains duplicate compiled states")

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "schema": "ncls.mdl-metal-state-pool@1",
                "registry_identity": self.registry_identity,
                "recipe_identity": self.recipe_identity,
                "split": self.split,
                "states": [
                    {"snapshot_id": snapshot.snapshot_id, "base_export_id": export_id}
                    for snapshot, export_id in zip(self.snapshots, self.base_export_ids)
                ],
            }
        )

    @classmethod
    def generate(
        cls,
        registry: MdlMetalRegistry,
        family: MdlFamilyDefinition,
        recipe: MdlMetalTypedStateRecipe,
        *,
        module_root: Path,
        exports: Sequence[MdlMetalExport] | None = None,
    ) -> "MdlMetalStatePool":
        import torch

        selected = tuple(registry.exports if exports is None else exports)
        snapshots: list[SourceSnapshot] = []
        base_ids: list[str] = []
        seen: set[str] = set()
        for export_index, record in enumerate(selected):
            locator = {
                **record.exact_locator,
                "module_root": str(module_root.resolve()),
            }
            base = family.load_snapshot(locator)
            editable = tuple(
                parameter
                for parameter in record.parameters
                if parameter.get("editable", False)
                and parameter["responsibility"] in recipe.responsibilities
                and _parameter_domain(parameter) is not None
            )
            dimension = max(1, len(editable))
            engine = torch.quasirandom.SobolEngine(
                dimension,
                scramble=True,
                seed=(
                    recipe.seed
                    + int(recipe.identity[:8], 16)
                    + export_index * 104729
                )
                % (2**31 - 1),
            )
            samples = engine.draw(recipe.states_per_export).tolist()
            for state_index, row in enumerate(samples):
                selector = row[0]
                if state_index == 0 or selector < recipe.default_weight or not editable:
                    snapshot = base
                else:
                    operations = []
                    boundary = selector < recipe.default_weight + recipe.boundary_weight
                    for parameter_index, parameter in enumerate(editable):
                        unit = float(row[parameter_index])
                        operations.append(
                            SourceEditOperation(
                                "set",
                                f"/arguments/{parameter['name']}",
                                _sample_parameter(parameter, unit, boundary, state_index),
                            )
                        )
                    snapshot = family.apply_edit(
                        base, SourceEditPatch(base.snapshot_id, tuple(operations))
                    ).snapshot
                if snapshot.snapshot_id not in seen:
                    seen.add(snapshot.snapshot_id)
                    snapshots.append(snapshot)
                    base_ids.append(record.export_id)
        return cls(
            registry.identity,
            recipe.identity,
            recipe.split,
            tuple(snapshots),
            tuple(base_ids),
        )


def _parameter_domain(parameter: Mapping[str, Any]) -> tuple[str, Any, Any] | None:
    kind = str(parameter["type"])
    if kind == "bool":
        return (kind, False, True)
    if kind == "enum" and parameter.get("choices"):
        return (kind, tuple(parameter["choices"]), None)
    minimum = parameter.get("minimum", parameter.get("soft_minimum"))
    maximum = parameter.get("maximum", parameter.get("soft_maximum"))
    if kind in {"float", "double", "int", "color", "float2", "float3", "float4"}:
        if minimum is None or maximum is None or float(maximum) <= float(minimum):
            return None
        return (kind, float(minimum), float(maximum))
    return None


def _sample_parameter(
    parameter: Mapping[str, Any], unit: float, boundary: bool, state_index: int
) -> Any:
    domain = _parameter_domain(parameter)
    if domain is None:
        raise ValueError("cannot sample an unbounded Metal parameter")
    kind, minimum, maximum = domain
    if kind == "bool":
        return bool((state_index + int(unit >= 0.5)) & 1)
    if kind == "enum":
        choices = tuple(minimum)
        return str(choices[min(int(unit * len(choices)), len(choices) - 1)]["name"])
    value = float(minimum if boundary and (state_index & 1) == 0 else maximum) if boundary else (
        float(minimum) + unit * (float(maximum) - float(minimum))
    )
    if kind == "int":
        return int(round(value))
    components = {"color": 3, "float2": 2, "float3": 3, "float4": 4}.get(kind)
    if components is not None:
        return tuple(value for _ in range(components))
    return value


__all__ = [
    "MDL_METAL_EXPECTED_COUNTS",
    "MDL_METAL_REGISTRY_SCHEMA",
    "MdlMetalExport",
    "MdlMetalRegistry",
    "MdlMetalStatePool",
    "MdlMetalTypedStateRecipe",
    "PARAMETER_RESPONSIBILITIES",
]
