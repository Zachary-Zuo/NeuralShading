from __future__ import annotations

import numpy as np

from ncls.source_materials.merl import MerlMaterial, merl_indices


def test_merl_material_identity_round_trip() -> None:
    material = MerlMaterial("blue-metallic-paint", "brdfs/blue-metallic-paint.binary")
    assert MerlMaterial.from_json(material.to_json()) == material


def test_merl_parameterization_is_reciprocal_for_isotropic_table() -> None:
    views = np.asarray([[0.0, 0.0, 1.0], [0.3, 0.4, 0.8660254]], dtype=np.float64)
    lights = np.asarray([[0.5, 0.0, 0.8660254], [-0.2, 0.4, 0.89442719]], dtype=np.float64)
    forward, forward_valid = merl_indices(views, lights)
    reverse, reverse_valid = merl_indices(lights, views)
    assert np.array_equal(forward_valid, reverse_valid)
    block = 90 * 180
    forward_theta_h, forward_tail = np.divmod(forward, block)
    reverse_theta_h, reverse_tail = np.divmod(reverse, block)
    forward_theta_d, forward_phi_d = np.divmod(forward_tail, 180)
    reverse_theta_d, reverse_phi_d = np.divmod(reverse_tail, 180)
    assert np.array_equal(forward_theta_h, reverse_theta_h)
    assert np.array_equal(forward_theta_d, reverse_theta_d)
    phi_distance = np.abs(forward_phi_d - reverse_phi_d)
    assert np.all(np.minimum(phi_distance, 180 - phi_distance) <= 1)
