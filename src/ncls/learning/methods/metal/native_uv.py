from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ncls.core.identity import sha256_json


@dataclass(frozen=True)
class UVMapping:
    """原生坐标表达式与运行时数值分开：相同默认值不合并不同可编辑表达式。"""
    expression: str
    affine: tuple[float, ...]
    mode: str = "direct"
    cell_scale: float = 1.0
    lookup_scale: float = 1.0
    hash_offset: int = 0
    uv_space: int = 0

    def __post_init__(self) -> None:
        if len(self.affine) != 6 or not all(math.isfinite(v) for v in self.affine):
            raise ValueError("native UV affine is invalid")
        if self.mode not in {"direct", "nonrepeat"} or self.uv_space != 0:
            raise ValueError("unsupported native UV source or tiling operation")
        if self.cell_scale <= 0 or self.lookup_scale <= 0:
            raise ValueError("native tiling scales must be positive")

    @property
    def identity(self) -> str:
        return sha256_json({"expression": self.expression, "affine": self.affine,
                            "mode": self.mode, "cell_scale": self.cell_scale,
                            "lookup_scale": self.lookup_scale, "hash_offset": self.hash_offset,
                            "uv_space": self.uv_space})

    @property
    def lookup_count(self) -> int:
        return 1 if self.mode == "direct" else 3


@dataclass(frozen=True)
class UVGroup:
    mapping: UVMapping
    slots: tuple[int, ...]


def group_compatible_uv(mappings: Mapping[int, Sequence[UVMapping]]) -> tuple[UVGroup, ...]:
    grouped: dict[str, tuple[UVMapping, list[int]]] = {}
    for slot, accesses in sorted(mappings.items()):
        for mapping in accesses:
            if mapping.identity not in grouped:
                grouped[mapping.identity] = mapping, []
            members = grouped[mapping.identity][1]
            if slot not in members:
                members.append(slot)
    # 按 schema 中 slot 的首次出现排序；一个原图可按不同 UV 表达式被多个组引用。
    return tuple(UVGroup(mapping, tuple(slots)) for mapping, slots in grouped.values())


def _lowbias32(value: torch.Tensor) -> torch.Tensor:
    value = torch.bitwise_and(value, 0xFFFFFFFF)
    value = torch.bitwise_xor(value, value >> 16) * 0x7FEB352D
    value = torch.bitwise_and(value, 0xFFFFFFFF)
    value = torch.bitwise_xor(value, value >> 15) * 0x846CA68B
    value = torch.bitwise_and(value, 0xFFFFFFFF)
    return torch.bitwise_xor(value, value >> 16)


def native_hash22(cells: torch.Tensor) -> torch.Tensor:
    x, y = cells.unbind(dim=-1)
    inner = _lowbias32(y)
    first, second = _lowbias32(x + inner), _lowbias32(x + 32000 + inner)
    # MDL uint2float 分两段转换，避免依赖 int32 signed conversion。
    def convert(value):
        return (value & 0x7FFFFFFF).to(torch.float32) + (value >= 0x80000000).to(torch.float32) * 2147483648.0
    return torch.stack((convert(first), convert(second)), dim=-1) / 4294967296.0


def native_uv_lookups(
    mapping: UVMapping, uv: torch.Tensor, uv_dx: torch.Tensor, uv_dy: torch.Tensor,
    *, return_cells: bool = False,
):
    matrix = uv.new_tensor(mapping.affine).reshape(2, 3)
    coordinate = uv @ matrix[:, :2].T + matrix[:, 2]
    dx, dy = uv_dx @ matrix[:, :2].T, uv_dy @ matrix[:, :2].T
    if mapping.mode == "direct":
        result = coordinate[:, None], dx[:, None], dy[:, None], torch.ones_like(uv[:, :1])
        return (*result, None) if return_cells else result
    tilted = coordinate * mapping.cell_scale
    tilted = torch.stack((tilted[:, 0] - tilted[:, 1] / math.sqrt(3.0),
                          tilted[:, 1] * (2.0 / math.sqrt(3.0))), dim=1)
    cell = torch.floor(tilted).to(torch.int64)
    fraction = tilted - torch.floor(tilted)
    third = 1.0 - fraction[:, 0] - fraction[:, 1]
    lower = third > 0
    offsets_lo = cell.new_tensor(((0, 0), (0, 1), (1, 0)))
    offsets_hi = cell.new_tensor(((1, 1), (1, 0), (0, 1)))
    offsets = torch.where(lower[:, None, None], offsets_lo[None], offsets_hi[None])
    indices = cell[:, None] + offsets + mapping.hash_offset
    samples = coordinate[:, None] * mapping.lookup_scale - native_hash22(indices)
    lo = torch.stack((third, fraction[:, 1], fraction[:, 0]), dim=1)
    hi = torch.stack((-third, 1.0 - fraction[:, 1], 1.0 - fraction[:, 0]), dim=1)
    weights = torch.where(lower[:, None], lo, hi)
    weights = weights / torch.linalg.vector_norm(weights, dim=1, keepdim=True)
    result = (samples, (dx * mapping.lookup_scale)[:, None].expand(-1, 3, -1),
              (dy * mapping.lookup_scale)[:, None].expand(-1, 3, -1), weights)
    return (*result, indices) if return_cells else result


def native_to_texture_coordinates(uv: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor):
    """MDL lower-left UV → encoder 原始图片 top-left；与 reference runtime 文件布局一致。"""
    sign = uv.new_tensor((1., -1.))
    return uv * sign + uv.new_tensor((0., 1.)), dx * sign, dy * sign


def _arguments(text: str, open_index: int) -> tuple[tuple[str, ...], int]:
    depth, quoted, escaped = 1, False, False
    begin, parts = open_index + 1, []
    for index in range(begin, len(text)):
        char = text[index]
        if quoted:
            if char == '"' and not escaped:
                quoted = False
            escaped = char == "\\" and not escaped
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                parts.append(text[begin:index].strip())
                return tuple(parts), index + 1
        elif char == "," and depth == 1:
            parts.append(text[begin:index].strip())
            begin = index + 1
    raise ValueError("unterminated native MDL coordinate expression")


def _number(expression: str, parameters: Mapping[str, Any]) -> np.ndarray:
    expression = re.sub(r"(?<=[\d.])[fFdD]\b", "", expression.strip())
    node = ast.parse(expression, mode="eval").body
    def evaluate(item):
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return np.asarray(item.value, dtype=np.float64)
        if isinstance(item, ast.Name) and item.id in parameters:
            return np.asarray(parameters[item.id], dtype=np.float64)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(item.op, ast.USub) else 1) * evaluate(item.operand)
        if isinstance(item, ast.BinOp):
            left, right = evaluate(item.left), evaluate(item.right)
            if isinstance(item.op, ast.Mult): return left * right
            if isinstance(item.op, ast.Div): return left / right
            if isinstance(item.op, ast.Add): return left + right
            if isinstance(item.op, ast.Sub): return left - right
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in {"float", "float2", "float3"}:
            count = 1 if item.func.id == "float" else int(item.func.id[-1])
            values = np.concatenate([np.atleast_1d(evaluate(arg)) for arg in item.args])
            return np.repeat(values, count) if len(values) == 1 else values
        raise ValueError(f"unsupported native coordinate scalar expression: {expression}")
    value = evaluate(node)
    if not np.isfinite(value).all():
        raise ValueError("native coordinate parameter is non-finite")
    return value


def _coordinate(expression: str, parameters: Mapping[str, Any], *, rotation_degrees: bool) -> tuple[float, ...]:
    expression = expression.strip()
    if "(" not in expression:
        raise ValueError(f"unsupported native coordinate expression: {expression}")
    function = expression[:expression.index("(")].strip().split("::")[-1]
    args, _ = _arguments(expression, expression.index("("))
    if function == "coordinate_source":
        if len(args) >= 2 and int(_number(args[1], parameters)) != 0:
            raise ValueError("only the declared surface UV set 0 is available")
        return (1., 0., 0., 0., 1., 0.)
    if function == "vm_coord_post_scale":
        affine = np.array(_coordinate(args[0], parameters, rotation_degrees=rotation_degrees)).reshape(2, 3)
        scale = np.broadcast_to(_number(args[1], parameters), (2,))
        if np.any(scale == 0): raise ValueError("native UV scale cannot be zero")
        return tuple((affine / scale[:, None]).ravel())
    if function not in {"vmat_transform", "vm_coord"}:
        raise ValueError(f"unsupported native coordinate function {function}")
    if not args[0]:
        return (1., 0., 0., 0., 1., 0.)
    translate = np.broadcast_to(_number(args[0], parameters), (2,))
    angle = float(_number(args[1], parameters))
    if function == "vm_coord" or rotation_degrees: angle = math.radians(angle)
    scale = np.broadcast_to(_number(args[2], parameters), (2,))
    if np.any(scale == 0): raise ValueError("native UV scale cannot be zero")
    uv_arg = 3 if function == "vm_coord" else 4
    if len(args) > uv_arg and int(_number(args[uv_arg], parameters)) != 0:
        raise ValueError("only the declared surface UV set 0 is available")
    c, s = math.cos(angle), math.sin(angle)
    # MDL matrix 构造按 column；native scale*rotate 是逆向旋转后按轴除 scale。
    return (c / scale[0], s / scale[0], float(translate[0]),
            -s / scale[1], c / scale[1], float(translate[1]))


def _tiling_branch_intervals(text: str) -> tuple[tuple[int, int, bool], ...]:
    depth, quoted, escaped = 0, False, False
    pending: list[tuple[int, int, bool, bool]] = []
    result = []
    for index, char in enumerate(text):
        if quoted:
            if char == '"' and not escaped: quoted = False
            escaped = char == "\\" and not escaped
            continue
        if char == '"': quoted = True
        elif char in "([{": depth += 1
        elif char == "?":
            known = re.search(r"\binfinite_tiling\s*$", text[max(0, index - 40):index]) is not None
            pending.append((depth, index + 1, True, known))
        elif char == ":" and text[index - 1:index] != ":" and text[index + 1:index + 2] != ":":
            if pending and pending[-1][0] == depth and pending[-1][2]:
                dep, begin, _, known = pending.pop()
                if known: result.append((begin, index, True))
                pending.append((dep, index + 1, False, known))
        elif char in ")]},;":
            while pending and pending[-1][0] >= depth:
                _, begin, phase, known = pending.pop()
                if known: result.append((begin, index, phase))
            if char in ")]}": depth -= 1
    return tuple(result)


def native_slot_mappings(
    module_file: Path, parameters: Mapping[str, Any], slot_paths: Mapping[int, str],
) -> Mapping[int, tuple[UVMapping, ...]]:
    """读取原生 helper 调用的坐标合同；不反演 BSDF、不处理像素数值、不猜缺失 mapping。"""
    text = module_file.read_text(encoding="utf-8")
    values = dict(parameters)
    # 当前源的共享坐标别名是标量/向量代数，仍由 authored 表达式计算。
    for name, expression in re.findall(r"\bfloat2\s+(\w+)\s*=\s*([^;\n]+);", text):
        if name in {"tex_rescale"}:
            values[name] = _number(expression, values)
    degrees = "float rotation_rad" in text
    infinite = bool(values.get("infinite_tiling", True))
    tiling_branches = _tiling_branch_intervals(text)
    names = {Path(path).name: index for index, path in slot_paths.items()}
    result: dict[int, dict[str, UVMapping]] = {index: {} for index in slot_paths}
    pattern = r"(?<![\w])(?P<name>endless_texture|endless_normal|vm_tex_infinite_normal|vm_tex_infinite|vm_tex_lookup|vm_tex_normal_lookup_2x|vm_tex_normal_lookup|file_texture)\s*\("
    for match in re.finditer(pattern, text):
        name = match.group("name")
        args, _ = _arguments(text, match.end() - 1)
        texture = re.fullmatch(r'texture_2d\s*\(\s*"([^"]+)".*\)', args[0], re.DOTALL)
        if texture is None or Path(texture.group(1)).name not in names:
            continue
        if any(begin <= match.start() < end and phase != infinite for begin, end, phase in tiling_branches):
            continue
        accesses = [(names[Path(texture.group(1)).name], 0)]
        if name == "vm_tex_normal_lookup_2x":
            second = re.fullmatch(r'texture_2d\s*\(\s*"([^"]+)".*\)', args[1], re.DOTALL)
            if second is None or Path(second.group(1)).name not in names:
                raise ValueError("native second normal texture mapping is missing")
            accesses.append((names[Path(second.group(1)).name], 1))
        for slot, texture_position in accesses:
            coordinate_index = {"endless_normal": 4, "file_texture": 4, "vm_tex_normal_lookup_2x": 2 + texture_position}.get(name, 1)
            expression = args[coordinate_index]
            affine = _coordinate(expression, values, rotation_degrees=degrees)
            mode, cell_scale, lookup_scale, offset = "direct", 1., 1., 0
            if name in {"endless_texture", "endless_normal"}:
                index = 5 if name == "endless_normal" else 2
                scale = float(_number(args[index], values))
                patch = float(_number(args[index + 2], values))
                mode, cell_scale, lookup_scale = "nonrepeat", scale, scale / patch
                rule = f"endless({args[index]},{args[index + 2]})"
            elif name in {"vm_tex_infinite", "vm_tex_infinite_normal"}:
                patch = float(_number(args[3], values))
                mode, cell_scale, offset = "nonrepeat", 1. / patch, 935
                rule = f"vm-infinite({args[3]})"
            else:
                rule = "direct"
            identity_expression = re.sub(r"\s+", "", expression) + ":" + rule
            mapping = UVMapping(identity_expression, affine, mode, cell_scale, lookup_scale, offset)
            result[slot][mapping.identity] = mapping
    missing = [slot for slot, accesses in result.items() if not accesses]
    if missing:
        raise ValueError(f"native UV contract has unhandled texture accesses: {module_file.name}, slots={missing}")
    return {slot: tuple(accesses.values()) for slot, accesses in result.items()}
