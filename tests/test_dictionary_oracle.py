import numpy as np
import torch

from baselines.dictionary_fit import SharedSgDictionary


def test_shared_dictionary_shapes_and_positive_basis() -> None:
    target = torch.ones((3, 8, 3), dtype=torch.float32)
    dictionary = SharedSgDictionary(target, atom_count=4, seed=5)
    lights = torch.nn.functional.normalize(torch.rand((8, 3)), dim=1)
    basis = dictionary.basis(lights)
    prediction = dictionary.prediction(basis, torch.tensor([0, 2]))
    assert basis.shape == (4, 8)
    assert prediction.shape == (2, 8, 3)
    assert torch.all(torch.isfinite(basis))
    assert torch.all(basis > 0.0)
    assert torch.all(prediction > 0.0)


def test_reflection_frame_dictionary_depends_on_view_and_stays_finite() -> None:
    target = torch.ones((2, 8, 3), dtype=torch.float32)
    dictionary = SharedSgDictionary(target, atom_count=6, seed=7, reflection_frame=True)
    lights = torch.nn.functional.normalize(torch.rand((8, 3)), dim=1)
    views = torch.tensor([[0.0, 0.0, 1.0], [0.7, 0.0, np.sqrt(0.51)]], dtype=torch.float32)
    basis = dictionary.basis(lights, views)
    assert basis.shape == (2, 6, 8)
    assert torch.all(torch.isfinite(basis))
    assert not torch.allclose(basis[0], basis[1])
