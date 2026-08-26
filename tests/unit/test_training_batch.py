import pytest
import torch

from ncls.data.training_batch import TrainingBatch


def _batch(device="cpu"):
    tensors = {
        "source_index": torch.zeros(2, dtype=torch.int64, device=device),
        "wo": torch.tensor([[0., 0., 1.], [0., 0., 1.]], device=device),
        "wi": torch.tensor([[[0., 0., 1.]], [[0., 0., 1.]]], device=device),
        "target": torch.ones((2, 1, 3), device=device),
        "solid_angle_weight": torch.ones((2, 1), device=device),
        "reference_pdf": torch.ones((2, 1), device=device),
        "sample_count": torch.ones((2, 1), dtype=torch.int64, device=device),
        "rng_seed": torch.zeros((2, 1), dtype=torch.int64, device=device),
        "query_role": torch.zeros(2, dtype=torch.int64, device=device),
    }
    return TrainingBatch("family", ("a" * 64, "b" * 64), "linear-f", tensors, {"source": "test"})


def test_training_batch_enforces_one_device_and_shapes():
    batch = _batch()
    assert batch.batch_size == 2 and batch.device.type == "cpu"
    bad = dict(batch.tensors)
    bad["target"] = torch.ones((2, 2, 3))
    with pytest.raises(ValueError, match="target"):
        TrainingBatch("family", batch.source_state_ids, "linear-f", bad, {})
