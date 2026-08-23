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
    view_values = np.asarray(views, dtype=np.float32)
    light_values = np.asarray(lights, dtype=np.float32)
    if light_values.ndim == 2:
        light_values = np.broadcast_to(light_values[None, ...], (len(view_values), *light_values.shape))
    if light_values.ndim == 3:
        if light_values.shape[0] != len(view_values):
            raise ValueError("per-view lights must match views")
        light_values = np.broadcast_to(light_values[None, ...], (surface_count, *light_values.shape))
    elif light_values.ndim == 4:
        if light_values.shape[:2] != (surface_count, len(view_values)):
            raise ValueError("per-surface lights must match surfaces and views")
    else:
        raise ValueError("lights must have shape [light, 3], [view, light, 3], or [surface, view, light, 3]")
    direction_count = light_values.shape[-2]
    view_rows = np.tile(np.repeat(view_values, direction_count, axis=0), (surface_count, 1))
    light_rows = light_values.reshape(-1, 3)
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
        .reshape(surface_count, len(views), np.asarray(lights).shape[-2], 3).copy()
    )
