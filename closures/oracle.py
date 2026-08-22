from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from schema import BINARY_SIZE as STACK_BINARY_SIZE, unpack_stack

from .packet import ClosurePacket, LtcResidualLobe


def load_oracle_packets(
    dataset_dir: Path,
    archive_path: Path,
    tile_indices: Iterable[int] | None = None,
) -> list[ClosurePacket]:
    """Materialize direct-top + LTC-K2 packets from an oracle archive."""
    with np.load(archive_path) as archive:
        state_indices = np.asarray(archive["state_indices"], dtype=np.uint32)
        amplitude = np.asarray(archive["amplitude"], dtype=np.float32)
        inverse_scale = np.asarray(archive["inverse_scale"], dtype=np.float32)
        shear = np.asarray(archive["shear"], dtype=np.float32)
        angle = np.asarray(archive["angle"], dtype=np.float32)
    if amplitude.shape[1:] != (2, 3) or inverse_scale.shape[1:] != (2, 2):
        raise ValueError("oracle archive is not an LTC-K2 packet archive")

    payload = (dataset_dir / "stacks.bin").read_bytes()
    selected = range(len(state_indices)) if tile_indices is None else tile_indices
    packets: list[ClosurePacket] = []
    for raw_index in selected:
        tile_index = int(raw_index)
        state_index = int(state_indices[tile_index])
        offset = state_index * STACK_BINARY_SIZE
        stack = unpack_stack(payload[offset : offset + STACK_BINARY_SIZE])
        lobes = tuple(
            LtcResidualLobe(
                amplitude=tuple(amplitude[tile_index, lobe_index]),
                inverse_scale=tuple(inverse_scale[tile_index, lobe_index]),
                shear=tuple(shear[tile_index, lobe_index]),
                angle=float(angle[tile_index, lobe_index]),
            )
            for lobe_index in range(2)
        )
        packets.append(ClosurePacket(stack.layers[0], lobes))  # type: ignore[arg-type]
    return packets
