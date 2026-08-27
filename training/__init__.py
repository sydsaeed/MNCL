from .checkpoint import CheckpointState, load_checkpoint, save_checkpoint

__all__ = [
    "CTREvaluationResult",
    "CheckpointState",
    "EpochLossStats",
    "evaluate_checkpoint",
    "evaluate_ctr",
    "evaluate_one_epoch",
    "load_checkpoint",
    "move_graphs_to_device",
    "predict_ctr",
    "resolve_device",
    "save_checkpoint",
    "train_epochs",
    "train_one_epoch",
]


def __getattr__(name):
    if name in {
        "EpochLossStats",
        "evaluate_one_epoch",
        "move_graphs_to_device",
        "resolve_device",
        "train_epochs",
        "train_one_epoch",
    }:
        from . import trainer
        return getattr(trainer, name)

    if name in {
        "CTREvaluationResult",
        "evaluate_checkpoint",
        "evaluate_ctr",
        "predict_ctr",
    }:
        from . import evaluator
        return getattr(evaluator, name)

    raise AttributeError(name)
