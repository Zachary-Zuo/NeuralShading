"""仅盘点历史 viewer 图像及解释元数据，不复制或修改原产物。"""

import json
from collections import Counter
from pathlib import Path
import subprocess


task_root = Path(__file__).resolve().parents[1]
project_root = Path(__file__).resolve().parents[4]
result = subprocess.run(
    ["rg", "-l", "-F", "ncls.viewer-capture", "artifacts", "--no-ignore", "-g", "*.json"],
    cwd=project_root, capture_output=True, text=True, encoding="utf-8", check=True,
)
captures = []
images = set()
missing = []
for relative in sorted(result.stdout.splitlines()):
    path = project_root / relative
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("format_name") != "ncls.viewer-capture":
        continue
    files = document.get("files", {})
    retained = []
    metadata = [path.relative_to(project_root).as_posix()]
    for role, value in files.items():
        if not isinstance(value, str) or not value:
            continue
        candidate = path.parent / value
        if candidate.suffix.lower() not in {".png", ".exr", ".json", ".csv"}:
            continue
        if not candidate.is_file():
            missing.append({"capture": relative, "role": role, "path": value})
            continue
        try:
            name = candidate.resolve().relative_to(project_root).as_posix()
        except ValueError:
            continue
        if candidate.suffix.lower() in {".png", ".exr"}:
            images.add(name)
            retained.append({"role": role, "path": name})
        else:
            metadata.append(name)
    captures.append({
        "capture": path.relative_to(project_root).as_posix(),
        "historical_format_version": document.get("format_version"),
        "images": retained,
        "metadata": sorted(set(metadata)),
        "source_material": document.get("source_material"),
        "source_family": document.get("source_material_family_id"),
        "resolution": document.get("resolution"),
        "reference_spp": document.get("reference_spp"),
        "slots": [{key: slot.get(key) for key in (
            "slot_index", "mode", "spp", "status", "package_id", "checkpoint_profile_id"
        )} for slot in document.get("slots", [])],
    })
all_images = subprocess.run(
    ["rg", "--files", "--no-ignore", "artifacts", "-g", "*.png", "-g", "*.exr"],
    cwd=project_root, capture_output=True, text=True, encoding="utf-8", check=True,
)
unassociated = sorted(
    path.replace("\\", "/") for path in all_images.stdout.splitlines()
    if path.replace("\\", "/") not in images
)
groups = Counter(path.split("/")[1] for path in images)
inventory = {
    "note": "规划盘点；原文件未复制。历史 capture 格式只作为证据阅读，不是新运行时兼容合同。",
    "capture_count": len(captures),
    "image_count": len(images),
    "image_bytes": sum((project_root / name).stat().st_size for name in images),
    "groups": dict(groups.most_common()),
    "captures": captures,
    "missing_references": missing,
    "unassociated_images": unassociated,
}
output = task_root / "scratch" / "visual-evidence-inventory.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: inventory[key] for key in (
    "capture_count", "image_count", "image_bytes", "groups"
)}, ensure_ascii=False))
print(f"missing_references={len(missing)}; unassociated_images={len(unassociated)}")
