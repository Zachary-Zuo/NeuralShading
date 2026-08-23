from .metrics import response_loss


def evaluate_checkpoint(*args, **kwargs):
    from .evaluator import evaluate_checkpoint as run

    return run(*args, **kwargs)


def evaluate_model(*args, **kwargs):
    from .evaluator import evaluate_model as run

    return run(*args, **kwargs)

__all__ = ["evaluate_checkpoint", "evaluate_model", "response_loss"]
