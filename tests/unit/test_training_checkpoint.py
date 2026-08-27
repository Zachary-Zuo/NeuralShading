import torch

from ncls.core.identity import sha256_json
from ncls.learning.training.checkpoint import TrainingCheckpoint, load_checkpoint, save_checkpoint
from tests.fixtures.method_definition import METHOD_DEFINITION


def test_checkpoint_v2_roundtrip_and_tensor_schema(tmp_path):
    descriptor = METHOD_DEFINITION.descriptor
    config = {"name": "fixture"}
    checkpoint = TrainingCheckpoint(
        descriptor.method_key, descriptor.descriptor_sha256, descriptor.implementation_sha256,
        config, sha256_json(config), "offline:test", tuple(x.to_dict() for x in descriptor.supported_sources),
        ("a" * 64,), 7, "bootstrap", {"metric": 1.0},
        {"fixture.scale": torch.ones(3), "fixture.bias": torch.zeros(3)},
    )
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(path, checkpoint)
    assert len(digest) == 64
    restored = load_checkpoint(path, descriptor=descriptor)
    assert restored.format_version == 2 and restored.step == 7
