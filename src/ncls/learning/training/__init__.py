from .config import TrainingConfig


def train(*args, **kwargs):
    from .runner import train as run

    return run(*args, **kwargs)

__all__ = ["TrainingConfig", "train"]
