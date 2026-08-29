from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ncls.core.identity import sha256_file, sha256_json
from ncls.source_materials.mdl_metal import (
    MDL_METAL_EXPECTED_COUNTS,
    MDL_METAL_REGISTRY_SCHEMA,
    MdlMetalRegistry,
)


INSPECTION_SCHEMA = "ncls.mdl-metal-export-inspection@1"


_COORDINATES = {
    "coordinate_system",
    "horizontal_offset",
    "infinite_tiling",
    "no_uv",
    "object_scaled_bump",
    "projection_type",
    "scale",
    "texture_rotate",
    "texture_scale",
    "texture_translate",
    "uv_space_index",
}
_FRAME_TOKENS = ("roundcorner", "roundcorners", "round_corners", "across_materials")
_AGING_TOKENS = (
    "abrasion",
    "age",
    "cavities",
    "cloud",
    "corrosion",
    "crack",
    "damage",
    "dent",
    "dirt",
    "drops",
    "flow_stains",
    "impurities",
    "imperfection",
    "leak",
    "oxid",
    "patina",
    "pit_",
    "rough_scratches",
    "rust",
    "scratch",
    "smudge",
    "spotsdirt",
    "streak",
    "surface_dirt",
    "surface_impurities",
    "wash",
    "wear",
)
_COATING_TOKENS = (
    "anodization",
    "coating",
    "paint",
    "polish_film",
    "rod_",
    "weave_",
)
_METAL_CORE_TOKENS = (
    "bare_metal",
    "brass_type",
    "copper_tint",
    "coppery",
    "grazing_reflectivity",
    "heat_treatment",
    "metal_color",
    "metal_roughness",
    "metal_tint",
    "metalness",
    "normal_reflectivity",
    "reflection_",
    "steel_",
    "zinc_roughness",
)

_KNOWN_METALS = (
    "stainless-steel",
    "carbon-steel",
    "galvanized-steel",
    "blued-steel",
    "aluminum",
    "brass",
    "bronze",
    "chromium",
    "chrome",
    "copper",
    "gold",
    "iron",
    "mercury",
    "nickel",
    "platinum",
    "silver",
    "titanium",
    "tungsten",
    "zinc",
    "steel",
    "metal",
    "solder",
)


def _responsibility(name: str) -> tuple[str, str]:
    lower = name.lower()
    if lower in _COORDINATES or lower.startswith("cutout_") or lower in {
        "hexagonal_grid",
        "punching_grid_size",
        "shape_select",
        "square_grid_offset",
    }:
        return "coordinates", "exact-name"
    if lower in {"radius", "radius_mm", "roundcorner_radius"} or any(
        token in lower for token in _FRAME_TOKENS
    ):
        return "frame", "frame-keyword"
    if any(token in lower for token in _COATING_TOKENS):
        return "coating-composite", "coating-keyword"
    if any(token in lower for token in _AGING_TOKENS):
        return "aging-contamination", "aging-keyword"
    if any(token in lower for token in _METAL_CORE_TOKENS) or lower in {"color_1", "factor"}:
        return "metal-core", "metal-optical-keyword"
    return "finish-microstructure", "finish-residual-audited"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _metal_finish(module: str) -> tuple[str, str, str]:
    leaf = module.split("::Metal::", 1)[1].replace("::", "_")
    normalized = _slug(leaf)
    special = {
        "aging-copper": ("copper", "aged"),
        "pcb-copper": ("copper", "pcb"),
        "pcb-goldfinger": ("gold", "pcb-goldfinger"),
        "steel-carbon": ("carbon-steel", "base"),
        "steel-galvanized": ("galvanized-steel", "galvanized"),
        "steel-painted": ("steel", "painted"),
        "steel-painted-cracked": ("steel", "painted-cracked"),
        "stainless-steel": ("stainless-steel", "base"),
        "stainless-steel-brushed": ("stainless-steel", "brushed"),
        "stainless-steel-brushed-punched": ("stainless-steel", "brushed-punched"),
        "stainless-steel-milled": ("stainless-steel", "milled"),
        "stainless-steel-punched": ("stainless-steel", "punched"),
        "blued-steel-cold": ("blued-steel", "cold-blued"),
    }
    if normalized in special:
        metal, finish = special[normalized]
    else:
        metal = next(
            (candidate for candidate in _KNOWN_METALS if normalized == candidate or normalized.startswith(candidate + "-")),
            "metal",
        )
        finish = normalized[len(metal) :].strip("-") or "base"
    recipe_name = finish if finish != "base" else "plain"
    return metal, finish, recipe_name


def _role(value: str) -> str:
    token = _slug(value)
    aliases = {
        "diff": "base-color",
        "diffuse": "base-color",
        "albedo": "base-color",
        "col": "base-color",
        "norm": "normal-tangent",
        "normal": "normal-tangent",
        "rough": "roughness",
        "roughness": "roughness",
        "ao": "ambient-occlusion",
        "occ": "ambient-occlusion",
        "height": "height",
        "bump": "height",
        "metallic": "metalness",
        "metalness": "metalness",
    }
    if token in aliases:
        return aliases[token]
    for keyword, role in (
        ("normal", "normal-tangent"),
        ("norm", "normal-tangent"),
        ("rough", "roughness"),
        ("diff", "base-color"),
        ("color", "base-color"),
        ("gradient", "color-lookup"),
        ("height", "height"),
        ("bump", "height"),
        ("opacity", "opacity"),
        ("mask", "mask"),
        ("weight", "mask"),
        ("ao", "ambient-occlusion"),
    ):
        if keyword in token:
            return role
    return token or "source-data"


def _texture_semantics(source_path: str) -> tuple[list[str], Mapping[str, str]]:
    stem = Path(source_path).stem
    packed = re.search(r"(?:^|_)R_(.+?)_G_(.+?)_B_(.+?)(?:_A_(.+))?$", stem, re.IGNORECASE)
    if packed:
        channels = {
            channel: _role(value)
            for channel, value in zip("RGBA", packed.groups())
            if value is not None
        }
    else:
        role = _role(stem)
        channels = {"RGB": role}
    return sorted(set(channels.values())), channels


def _parameter(parameter: Mapping[str, Any]) -> dict[str, Any]:
    responsibility, evidence = _responsibility(str(parameter["name"]))
    result = {
        key: parameter[key]
        for key in (
            "name",
            "type",
            "value",
            "editable",
            "offset",
            "size",
            "choices",
            "minimum",
            "maximum",
            "soft_minimum",
            "soft_maximum",
        )
        if key in parameter
    }
    result.update(
        {
            "responsibility": responsibility,
            "responsibility_evidence": evidence,
            "value_modality": "discrete"
            if parameter["type"] in {"bool", "int", "enum"}
            else "continuous",
        }
    )
    return result


def _texture_set(
    record: Mapping[str, Any],
    module_root: Path,
    artifact_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    artifact = artifact_manifests[str(record["artifact_manifest_sha256"])]
    artifact_textures = {int(item["index"]): item for item in artifact["textures"]}
    slots = []
    for texture in record["textures"]:
        source_path = texture.get("source_path")
        if source_path:
            relative = Path(str(source_path))
            source = (module_root / relative).resolve()
            source.relative_to(module_root)
            provenance_path = relative.as_posix()
            source_sha256 = sha256_file(source)
            roles, channels = _texture_semantics(provenance_path)
            if str(texture["pixel_type"]) in {"Rgba", "Rgba_16", "Color"}:
                channels = {**channels, "A": "auxiliary-alpha"}
                roles = sorted(set((*roles, "auxiliary-alpha")))
            provenance_kind = "authored-file"
        else:
            artifact_texture = artifact_textures[int(texture["index"])]
            data = str(artifact_texture.get("data", ""))
            if not data or data not in artifact["files_sha256"]:
                raise ValueError("MDL provider-owned BSDF table has no payload hash")
            name = str(artifact_texture["name"])
            provenance_path = f"mdl-bsdf-data://{name}"
            source_sha256 = str(artifact["files_sha256"][data])
            roles = ["bsdf-multiple-scattering-lookup"]
            channels = {"R": "bsdf-multiple-scattering-lookup"}
            provenance_kind = "mdl-sdk-static-table"
        is_normal = "normal-tangent" in roles
        slots.append(
            {
                "slot_index": int(texture["index"]),
                "source_path": provenance_path,
                "source_sha256": source_sha256,
                "provenance_kind": provenance_kind,
                "shape": texture["shape"],
                "dimensions": texture["dimensions"],
                "pixel_type": texture["pixel_type"],
                "effective_gamma": float(texture["effective_gamma"]),
                "transfer": "srgb-to-linear"
                if str(texture["gamma"]) == "srgb"
                else "identity-linear",
                "roles": roles,
                "channels": channels,
                "normal_rule": "decode-renormalize-tangent"
                if is_normal
                else "not-normal",
                "mip_rule": "linear-average-renormalize"
                if is_normal
                else "linear-average",
                "filter_rule": "anisotropic-linear",
            }
        )
    return {
        "id": record["texture_set_id"],
        "slots": slots,
        "tile_policy": {
            "core_width": 128,
            "core_height": 128,
            "halo": 8,
            "address_mode": "wrap",
            "mip_source": "decoded-linear-source",
        },
    }


def build_registry(
    inspection: Mapping[str, Any],
    module_root: Path,
    artifact_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root = module_root.resolve()
    if inspection.get("schema") not in {
        INSPECTION_SCHEMA,
        "ncls.task-metal-export-inspection@1",
    }:
        raise ValueError("unsupported MDL Metal inspection schema")
    records = tuple(inspection["records"])
    if len(records) != MDL_METAL_EXPECTED_COUNTS["authored_exports"]:
        raise ValueError("Metal inspection does not contain the frozen 837 authored exports")
    opaque = tuple(
        record for record in records if not record["capability_audit"]["cutout_opacity"]
    )
    cutout = tuple(
        record for record in records if record["capability_audit"]["cutout_opacity"]
    )
    graphs: dict[str, dict[str, Any]] = {}
    schemas: dict[str, dict[str, Any]] = {}
    for inspected_record in records:
        inspected_schema_id = str(inspected_record["parameter_schema_id"])
        schemas.setdefault(
            inspected_schema_id,
            {
                "id": inspected_schema_id,
                "parameters": [
                    {
                        **parameter,
                        "responsibility": _responsibility(str(parameter["name"]))[0],
                        "value_modality": "discrete"
                        if parameter["type"] in {"bool", "int", "enum"}
                        else "continuous",
                    }
                    for parameter in inspected_record["parameter_schema"]
                ],
                "opaque_reachable": False,
            },
        )
    texture_sets: dict[str, dict[str, Any]] = {}
    recipes: dict[str, dict[str, Any]] = {}
    exports = []
    for record in opaque:
        metal, finish, recipe_name = _metal_finish(str(record["module"]))
        recipe_id = sha256_json(
            {"schema": "ncls.mdl-metal-recipe-key@1", "finish": recipe_name}
        )
        recipes.setdefault(
            recipe_id,
            {
                "id": recipe_id,
                "name": recipe_name,
                "compatible_metals": [],
                "compatible_finishes": [],
                "texture_set_ids": [],
                "graph_ids": [],
            },
        )
        for key, value in (
            ("compatible_metals", metal),
            ("compatible_finishes", finish),
            ("texture_set_ids", record["texture_set_id"]),
            ("graph_ids", record["graph_identity_id"]),
        ):
            if value not in recipes[recipe_id][key]:
                recipes[recipe_id][key].append(value)
        graph_id = str(record["graph_identity_id"])
        graphs.setdefault(
            graph_id,
            {
                "id": graph_id,
                "graph_identity": record["graph_identity"],
                "source_module_set_id": record["source_module_set_id"],
                "source_modules": record["source_modules"],
                "capability": record["capability_audit"],
            },
        )
        schema_id = str(record["parameter_schema_id"])
        schemas[schema_id]["opaque_reachable"] = True
        texture_id = str(record["texture_set_id"])
        candidate_texture_set = _texture_set(record, root, artifact_manifests)
        previous_texture_set = texture_sets.setdefault(texture_id, candidate_texture_set)
        if previous_texture_set != candidate_texture_set:
            raise ValueError("one Metal texture-set identity has conflicting slot provenance")
        locator = {
            "kind": "mdl-export",
            "pack_id": "nvidia.vmaterials2",
            "pack_version": "2.4.0",
            "module": record["module"],
            "export": record["exact_export"],
        }
        export_id = sha256_json({"schema": "ncls.mdl-metal-export@1", "locator": locator})
        exports.append(
            {
                "export_id": export_id,
                "exact_locator": locator,
                "export_name": record["export_name"],
                "source_path": record["source_path"],
                "artifact_manifest_sha256": record["artifact_manifest_sha256"],
                "graph_id": graph_id,
                "parameter_schema_id": schema_id,
                "texture_set_id": texture_id,
                "recipe_id": recipe_id,
                "metal": metal,
                "finish": finish,
                "parameters": [_parameter(parameter) for parameter in record["parameters"]],
                "compiled_layout": {
                    "argument_block_bytes": int(record["argument_block_bytes"]),
                    "ro_data_bytes": int(record["ro_data_bytes"]),
                    "texture_slots": int(record["texture_count"]),
                    "generated_code_bytes": int(record["generated_code_bytes"]),
                },
            }
        )
    for recipe in recipes.values():
        for name in (
            "compatible_metals",
            "compatible_finishes",
            "texture_set_ids",
            "graph_ids",
        ):
            recipe[name].sort()
    dependency_paths = sorted(
        {
            str(item["path"])
            for record in records
            for item in record["source_modules"]
        }
        | {
            str(item["source_path"])
            for record in records
            for item in record["textures"]
            if item.get("source_path")
        }
    )
    source_closure = [
        {"path": path, "sha256": sha256_file(root / Path(path))}
        for path in dependency_paths
    ]
    payload = {
        "schema": MDL_METAL_REGISTRY_SCHEMA,
        "source": {
            "pack_id": "nvidia.vmaterials2",
            "pack_version": "2.4.0",
            "mdl_sdk_build": "2025.0.0-387700.1252",
            "inspection_schema": INSPECTION_SCHEMA,
            "bridge_executable_sha256": inspection["bridge_executable_sha256"],
            "source_closure_identity": sha256_json(source_closure),
            "source_closure": source_closure,
        },
        "counts": dict(MDL_METAL_EXPECTED_COUNTS),
        "opaque_exports": sorted(exports, key=lambda item: item["export_id"]),
        "rejected_cutout_exports": sorted(
            (
                {
                    "exact_locator": {
                        "kind": "mdl-export",
                        "pack_id": "nvidia.vmaterials2",
                        "pack_version": "2.4.0",
                        "module": record["module"],
                        "export": record["exact_export"],
                    },
                    "reason": "geometry.cutout_opacity",
                }
                for record in cutout
            ),
            key=lambda item: (item["exact_locator"]["module"], item["exact_locator"]["export"]),
        ),
        "tables": {
            "graphs": sorted(graphs.values(), key=lambda item: item["id"]),
            "parameter_schemas": sorted(schemas.values(), key=lambda item: item["id"]),
            "texture_sets": sorted(texture_sets.values(), key=lambda item: item["id"]),
            "recipes": sorted(recipes.values(), key=lambda item: item["id"]),
        },
    }
    counts = {
        "authored_exports": len(records),
        "opaque_exports": len(opaque),
        "rejected_cutout_exports": len(cutout),
        "opaque_graphs": len(graphs),
        "opaque_texture_sets": len(texture_sets),
        "parameter_schemas": len(schemas),
    }
    if counts != MDL_METAL_EXPECTED_COUNTS:
        raise ValueError(f"Metal registry audit changed: {counts}")
    payload["identity"] = sha256_json(payload)
    MdlMetalRegistry(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从锁定 MDL SDK inspection 生成全量 opaque vMaterials 2 Metal registry。"
    )
    parser.add_argument("--inspection-summary", type=Path)
    parser.add_argument("--module-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "references/mdl-vmaterials2-v1/metal-opaque-v1.json",
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.inspection_summary is None:
        if not arguments.check:
            parser.error("--inspection-summary is required when regenerating the registry")
        registry = MdlMetalRegistry.load(arguments.output)
        root = arguments.module_root.resolve()
        for dependency in registry.payload["source"]["source_closure"]:
            path = root / Path(str(dependency["path"]))
            if sha256_file(path) != dependency["sha256"]:
                raise SystemExit(f"MDL Metal source closure changed: {dependency['path']}")
        return 0
    if arguments.artifact_root is None:
        parser.error("--artifact-root is required with --inspection-summary")
    with arguments.inspection_summary.open("r", encoding="utf-8") as stream:
        inspection = json.load(stream)
    artifact_manifests = {}
    for path in arguments.artifact_root.rglob("manifest.json"):
        identity = sha256_file(path)
        with path.open("r", encoding="utf-8") as stream:
            artifact_manifests[identity] = json.load(stream)
    expected = {str(record["artifact_manifest_sha256"]) for record in inspection["records"]}
    missing = expected - set(artifact_manifests)
    if missing:
        raise ValueError(f"Metal inspection artifact closure is incomplete: {len(missing)} missing")
    payload = build_registry(inspection, arguments.module_root, artifact_manifests)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("tracked MDL Metal registry is stale")
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
