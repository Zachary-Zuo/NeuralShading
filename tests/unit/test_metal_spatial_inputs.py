import numpy as np
import torch

from ncls.learning.methods.metal.native_assets import MdlMetalNativeAssetCollection


def _decode(values, channels, rect=None, *, transfer="identity-linear", origin="top_left", depth=1):
    collection = MdlMetalNativeAssetCollection.__new__(MdlMetalNativeAssetCollection)
    rect = rect or (0, 0, values.shape[0] // depth, values.shape[1])
    return collection._read_raw_rectangle(values, {"shape": "2d" if depth == 1 else "bsdf_data", "data_origin": origin},
                                          {"channels": channels, "transfer": transfer}, rect, depth)


def test_absolute_ranges_hdr_signed_height_and_external_statistics_are_preserved():
    rough = np.array([[[0.2], [0.3]], [[0.4], [0.5]]], dtype=np.float32)
    torch.testing.assert_close(_decode(rough, {"R": "roughness"}), torch.from_numpy(rough.transpose(2, 0, 1))[None])
    torch.testing.assert_close(_decode(rough + 0.4, {"R": "roughness"}), _decode(rough, {"R": "roughness"}) + 0.4)
    signed = np.array([[[-12.0], [0.0], [34.5]]], dtype=np.float32)
    torch.testing.assert_close(_decode(signed, {"R": "height"}), torch.tensor([[[[-12.0, 0.0, 34.5]]]]))
    large = np.full((8, 8, 1), 1000.0, dtype=np.float32)
    large[2:4, 3:5] = rough
    torch.testing.assert_close(_decode(large, {"R": "roughness"}, (2, 3, 2, 2)), _decode(rough, {"R": "roughness"}))


def test_file_transfer_and_row_origin_have_fixed_non_symmetric_witnesses():
    values = np.array([[[0, 64, 255], [128, 32, 16]], [[255, 16, 8], [8, 4, 2]]], dtype=np.uint8)
    decoded = _decode(values, {"RGB": "base-color"}, transfer="srgb-to-linear")
    linear = values.astype(np.float64) / 255.0
    expected = np.where(linear <= 0.04045, linear / 12.92, ((linear + 0.055) / 1.055) ** 2.4)
    torch.testing.assert_close(decoded, torch.tensor(expected.transpose(2, 0, 1)[None], dtype=torch.float32))
    inverted_file = values[::-1].copy()
    torch.testing.assert_close(_decode(inverted_file, {"RGB": "base-color"}, origin="lower_left", transfer="srgb-to-linear"), decoded)


def test_normal_raw_length_survives_until_native_filter_strength_normalize_order():
    values = np.array([[[0.75, 0.5, 0.6], [0.6, 0.8, 0.9]]], dtype=np.float32)
    decoded = _decode(values, {"RGB": "normal-tangent"})
    torch.testing.assert_close(decoded, torch.tensor(values.transpose(2, 0, 1)[None]))
    # 原生 normal graph 先 lookup，再 strength，再 normalize；逐 texel 单位化改变结果。
    normal = 2 * decoded - 1
    correct = torch.nn.functional.normalize(normal.mean(dim=-1) * torch.tensor([0.3, 0.3, 1.0])[None, :, None], dim=1)
    premature = torch.nn.functional.normalize(torch.nn.functional.normalize(normal, dim=1).mean(dim=-1)
                                               * torch.tensor([0.3, 0.3, 1.0])[None, :, None], dim=1)
    assert not torch.allclose(correct, premature)


def test_native_lookup_depth_preserves_slices_without_surface_row_flip():
    values = np.arange(24, dtype=np.float32).reshape(6, 4, 1)
    decoded = _decode(values, {"R": "bsdf-multiple-scattering-lookup"}, depth=3, origin="lower_left")
    assert decoded.shape == (3, 1, 2, 4)
    torch.testing.assert_close(decoded[:, 0], torch.tensor(values.reshape(3, 2, 4)))
