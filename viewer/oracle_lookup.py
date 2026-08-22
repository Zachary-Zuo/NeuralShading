from __future__ import annotations

from pathlib import Path
from typing import Sequence

import falcor
import numpy as np

from closures.packet import BINARY_SIZE, ClosurePacket, pack_packets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHADER_FILE = PROJECT_ROOT / "viewer" / "kernels" / "oracle_lookup.cs.slang"


class FalcorOracleLookup:
    """Evaluate fitted closure packets through the same Slang path used by the viewer."""

    def __init__(self, light_directions: np.ndarray, *, max_packet_batch: int = 1) -> None:
        if max_packet_batch < 1:
            raise ValueError("max_packet_batch must be positive")
        lights = np.asarray(light_directions, dtype=np.float32)
        if lights.ndim != 2 or lights.shape[1] not in {3, 4}:
            raise ValueError("light_directions must have shape [count, 3 or 4]")
        self.light_count = len(lights)
        self.max_packet_batch = max_packet_batch
        self.query_capacity = self.light_count * max_packet_batch
        self.device = falcor.Device(type=falcor.DeviceType.D3D12)
        self.packet_buffer = self._buffer(BINARY_SIZE, max_packet_batch)
        self.view_buffer = self._buffer(16, max_packet_batch)
        self.light_buffer = self._buffer(16, self.light_count)
        self.output_buffer = self._buffer(16, self.query_capacity, writable=True)
        padded_lights = np.zeros((self.light_count, 4), dtype=np.float32)
        padded_lights[:, :3] = lights[:, :3]
        self.light_buffer.from_numpy(padded_lights)
        self.compute = falcor.ComputePass(
            self.device, file=SHADER_FILE, cs_entry="evaluateOraclePackets"
        )
        self.compute.globals.gPackets = self.packet_buffer
        self.compute.globals.gViewDirections = self.view_buffer
        self.compute.globals.gLightDirections = self.light_buffer
        self.compute.globals.gOutput = self.output_buffer
        self.compute.globals.gLightCount = self.light_count

    def _buffer(self, stride: int, element_count: int, *, writable: bool = False):
        flags = falcor.ResourceBindFlags.ShaderResource
        if writable:
            flags |= falcor.ResourceBindFlags.UnorderedAccess
        return self.device.create_structured_buffer(
            struct_size=stride,
            element_count=element_count,
            bind_flags=flags,
        )

    def evaluate(
        self,
        packets: Sequence[ClosurePacket],
        view_directions: np.ndarray,
    ) -> np.ndarray:
        packet_count = len(packets)
        if not 1 <= packet_count <= self.max_packet_batch:
            raise ValueError(f"packet batch must contain 1..{self.max_packet_batch} packets")
        views = np.asarray(view_directions, dtype=np.float32)
        if views.ndim != 2 or views.shape[0] != packet_count or views.shape[1] not in {3, 4}:
            raise ValueError("view_directions must have shape [packet_count, 3 or 4]")

        packet_payload = np.zeros(BINARY_SIZE * self.max_packet_batch, dtype=np.uint8)
        packed = pack_packets(packets)
        packet_payload[: len(packed)] = np.frombuffer(packed, dtype=np.uint8)
        padded_views = np.zeros((self.max_packet_batch, 4), dtype=np.float32)
        padded_views[:packet_count, :3] = views[:, :3]
        self.packet_buffer.from_numpy(packet_payload)
        self.view_buffer.from_numpy(padded_views)
        query_count = packet_count * self.light_count
        self.compute.globals.gQueryCount = query_count
        self.compute.execute(threads_x=query_count)
        return (
            self.output_buffer.to_numpy()
            .view(np.float32)
            .reshape(self.query_capacity, 4)[:query_count, :3]
            .reshape(packet_count, self.light_count, 3)
            .copy()
        )
