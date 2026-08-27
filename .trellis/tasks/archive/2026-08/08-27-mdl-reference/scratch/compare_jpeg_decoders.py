from pathlib import Path
import json

import numpy as np
from PIL import Image
import slangpy as spy


path = Path(
    "assets/source-materials/mdl-vmaterials2/2.4.0/Materials/"
    "vMaterials_2/Paint/Carpaint/textures/smudges_scratches_A_rough.jpg"
)
sgl = np.array(spy.Bitmap(path), copy=True)
pillow = np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True)
artifacts = []
for manifest_path in Path("build/mdl-reference/cache").glob("*/manifest.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for descriptor in manifest.get("textures", []):
        if (
            descriptor.get("data_origin") == "top_left"
            and str(descriptor.get("path", "")).endswith(path.name)
        ):
            payload = manifest_path.parent / descriptor["data"]
            artifacts.append(
                np.frombuffer(payload.read_bytes(), dtype=np.uint8).reshape(
                    descriptor["height"], descriptor["width"], 3
                )
            )
bridge = artifacts[-1]
vcpkg_manifest_path = Path("artifacts/reference-parity/mdl/diagnostic-vcpkgjpeg-artifact/manifest.json")
vcpkg_manifest = json.loads(vcpkg_manifest_path.read_text(encoding="utf-8"))
vcpkg_descriptor = next(
    item for item in vcpkg_manifest["textures"] if str(item.get("path", "")).endswith(path.name)
)
vcpkg_bridge = np.frombuffer(
    (vcpkg_manifest_path.parent / vcpkg_descriptor["data"]).read_bytes(), dtype=np.uint8
).reshape(vcpkg_descriptor["height"], vcpkg_descriptor["width"], 3)
print(
    "shape",
    sgl.shape,
    pillow.shape,
    "equal",
    np.array_equal(sgl, pillow),
    "max",
    np.abs(sgl.astype(int) - pillow.astype(int)).max(),
    "count",
    np.count_nonzero(sgl != pillow),
)
print(
    "sgl-bridge",
    "equal",
    np.array_equal(sgl, bridge),
    "max",
    np.abs(sgl.astype(int) - bridge.astype(int)).max(),
    "count",
    np.count_nonzero(sgl != bridge),
)
print(
    "sgl-vcpkg-bridge",
    "equal",
    np.array_equal(sgl, vcpkg_bridge),
    "max",
    np.abs(sgl.astype(int) - vcpkg_bridge.astype(int)).max(),
    "count",
    np.count_nonzero(sgl != vcpkg_bridge),
)

uv = np.array([0.2841676056180936, 0.39549965225688416], dtype=np.float32)
size = np.array([4096, 4096], dtype=np.float32)
result = uv * size + np.float32(0.5)
integer = np.floor(result)
fraction = result - integer
fraction = fraction * fraction * fraction * (
    fraction * (fraction * np.float32(6) - np.float32(15)) + np.float32(10)
)
filtered = (integer + fraction - np.float32(0.5)) / size
filtered[1] = np.float32(1) - filtered[1]
pixel = filtered * size - np.float32(0.5)
base = np.floor(pixel).astype(int)
print("sample", uv, "filtered", filtered, "pixel", pixel, "base", base, "frac", pixel - base)
x, y = base
print("sgl", sgl[y : y + 2, x : x + 2, :].tolist())
print("pillow", pillow[y : y + 2, x : x + 2, :].tolist())
print("bridge", bridge[y : y + 2, x : x + 2, :].tolist())
print("vcpkg-bridge", vcpkg_bridge[y : y + 2, x : x + 2, :].tolist())
print(
    "delta",
    (sgl[y : y + 2, x : x + 2, :].astype(int) - pillow[y : y + 2, x : x + 2, :].astype(int)).tolist(),
)
