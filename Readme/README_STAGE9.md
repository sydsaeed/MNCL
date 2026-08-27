# Stage 9 - Checkpoint and Resume

This stage adds persistent training state while keeping `main.ipynb` as the controller.

## Files

- `training/checkpoint.py`: save/load model, optimizer, histories, config, epoch, best loss, and RNG state.
- `training/trainer.py`: automatically saves `best` and `latest` checkpoints after each epoch.
- `config.py`: checkpoint directory and path helpers.
- `main.ipynb`: optional resume cell and checkpoint-aware train cell.

## Checkpoint files

For `last-fm` the defaults are:

- `checkpoints/last-fm_best.pt`: overwritten only when `test_loss` becomes smaller.
- `checkpoints/last-fm_latest.pt`: overwritten every epoch and intended for continuation.

The best checkpoint stores the exact model/optimizer state at the lowest observed `test_loss` up to that epoch.
The latest checkpoint stores the most recently completed epoch.

## Saved state

Each checkpoint contains:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `best_test_loss`
- `train_history`
- `test_history`
- serialized config values
- Python, NumPy, PyTorch, and CUDA RNG state when CUDA is available

## Resume

Initialize the model and optimizer first, then run the optional resume cell. By default it loads the latest checkpoint:

```python
resume_path = config.latest_checkpoint_path()
```

To continue from the best model instead:

```python
resume_path = config.best_checkpoint_path()
```

After loading, `total_train_loss` and `total_test_loss` are restored. The train cell uses their length as `start_epoch`, so executing it continues epoch numbering instead of restarting it.

## Train block behavior

`train_epochs` still returns exactly two arrays:

```python
train_loss, test_loss
```

The notebook then appends them to:

```python
total_train_loss
total_test_loss
```

Checkpoint saving happens inside `training/trainer.py`, not inside the notebook.
