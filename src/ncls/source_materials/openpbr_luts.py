from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_ENERGY_FILES = (
    "openpbr_ideal_dielectric_energy_complement_data.h",
    "openpbr_ideal_dielectric_avg_energy_complement_data.h",
    "openpbr_ideal_dielectric_reflection_ratio_data.h",
    "openpbr_opaque_dielectric_energy_complement_data.h",
    "openpbr_opaque_dielectric_avg_energy_complement_data.h",
    "openpbr_ideal_metal_energy_complement_data.h",
    "openpbr_ideal_metal_avg_energy_complement_data.h",
)


def _rgba(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape + (4,), dtype=np.float32)
    result[..., 0] = values
    return result


@dataclass(frozen=True)
class OpenPbrLutData:
    ideal_dielectric_energy: np.ndarray
    ideal_dielectric_average: np.ndarray
    ideal_dielectric_ratio: np.ndarray
    opaque_dielectric_energy: np.ndarray
    opaque_dielectric_average: np.ndarray
    ideal_metal_energy: np.ndarray
    ideal_metal_average: np.ndarray
    ltc: np.ndarray


def load_openpbr_luts(openpbr_bsdf_root: str | Path) -> OpenPbrLutData:
    root = Path(openpbr_bsdf_root) / "impl" / "data"
    energy: list[np.ndarray] = []
    expected_counts = (32**3, 32**2, 32**2, 32**3, 32**2, 32**2, 32)
    for filename, expected in zip(_ENERGY_FILES, expected_counts, strict=True):
        text = (root / filename).read_text(encoding="utf-8")
        values = np.asarray([int(item) for item in re.findall(r"(?<![\w.])-?\d+(?![\w.])", text)], dtype=np.float32)
        if values.size != expected:
            raise ValueError(f"OpenPBR LUT {filename} has {values.size} values, expected {expected}")
        energy.append(values / 65535.0)
    ltc_text = (root / "openpbr_ltc_data.h").read_text(encoding="utf-8")
    triples = re.findall(
        r"vec3\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)",
        ltc_text,
    )
    ltc_rgb = np.asarray(triples, dtype=np.float32)
    if ltc_rgb.shape != (32 * 32, 3):
        raise ValueError(f"OpenPBR LTC table has shape {ltc_rgb.shape}")
    ltc = np.zeros((32, 32, 4), dtype=np.float32)
    ltc[..., :3] = ltc_rgb.reshape(32, 32, 3)
    return OpenPbrLutData(
        _rgba(energy[0].reshape(32, 32, 32)),
        _rgba(energy[1].reshape(32, 32)),
        _rgba(energy[2].reshape(32, 32)),
        _rgba(energy[3].reshape(32, 32, 32)),
        _rgba(energy[4].reshape(32, 32)),
        _rgba(energy[5].reshape(32, 32)),
        _rgba(energy[6].reshape(1, 32)),
        ltc,
    )
