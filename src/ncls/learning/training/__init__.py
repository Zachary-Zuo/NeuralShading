from .config import TrainingConfig
from .sampler_config import SamplerTrainingConfig


def train(*args, **kwargs):
    from .runner import train as run

    return run(*args, **kwargs)


def train_sampler(*args, **kwargs):
    from .sampler_runner import train_sampler as run

    return run(*args, **kwargs)

__all__ = ["SamplerTrainingConfig", "TrainingConfig", "train", "train_sampler"]
