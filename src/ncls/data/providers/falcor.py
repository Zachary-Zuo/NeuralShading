from __future__ import annotations

from pathlib import Path

import numpy as np


def import_falcor():
    try:
        import falcor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Falcor reference collection must run through scripts/run_falcor_python.ps1"
        ) from exc
    return falcor


def direction_rows(views: np.ndarray, lights: np.ndarray, surface_count: int) -> tuple[np.ndarray, np.ndarray]:
    view_rows = np.repeat(np.tile(views, (surface_count, 1)), len(lights), axis=0)
    light_rows = np.tile(lights, (surface_count * len(views), 1))
    return np.ascontiguousarray(np.pad(view_rows, ((0, 0), (0, 1)))), np.ascontiguousarray(
        np.pad(light_rows, ((0, 0), (0, 1)))
    )


def structured_buffer(device, falcor, data: np.ndarray, stride: int):
    values = np.ascontiguousarray(data)
    result = device.create_structured_buffer(
        struct_size=stride,
        element_count=len(values),
        bind_flags=falcor.ResourceBindFlags.ShaderResource,
    )
    result.from_numpy(values)
    return result


def output_buffer(device, falcor, count: int):
    flags = falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess
    return device.create_structured_buffer(struct_size=16, element_count=count, bind_flags=flags)


def execute_direction_kernel(compute, device, falcor, views: np.ndarray, lights: np.ndarray, surface_count: int) -> np.ndarray:
    view_rows, light_rows = direction_rows(views, lights, surface_count)
    query_count = len(view_rows)
    view_buffer = structured_buffer(device, falcor, view_rows, 16)
    light_buffer = structured_buffer(device, falcor, light_rows, 16)
    output = output_buffer(device, falcor, query_count)
    compute.globals.gViews = view_buffer
    compute.globals.gLights = light_buffer
    compute.globals.gOutput = output
    compute.globals.gQueryCount = query_count
    compute.execute(threads_x=query_count)
    return (
        output.to_numpy().view(np.float32).reshape(query_count, 4)[:, :3]
        .reshape(surface_count, len(views), len(lights), 3).copy()
    )
