# Final run checklist

## 1. Environment

From the project root:

```bash
python verify_environment.py --dataset last-fm
```

The result should be `READY`.

## 2. End-to-end smoke test

```bash
python smoke_check.py
```

This must print `Smoke test passed` before a real training run.

## 3. Notebook order

Run `main.ipynb` from the project root in this order:

1. Imports
2. Config / seed / device
3. Data load, split, and graph build
4. Model initialization
5. Optional checkpoint resume
6. Train block
7. Loss plot
8. CTR evaluation

Only rerun the **Train block** when you want more epochs. Do not rerun model initialization unless you intentionally want a fresh model.

## 4. Checkpoints

- `checkpoints/last-fm_latest.pt`: continue the most recent run.
- `checkpoints/last-fm_best.pt`: model with the minimum `test_loss` seen so far.

## 5. Training history

After each train block:

```python
len(total_train_loss) == len(total_test_loss)
```

Both arrays should grow by exactly the requested number of epochs.

## 6. Evaluation

The evaluation block loads `last-fm_best.pt` into a separate model and reports:

- AUC
- F1
- Precision
- Recall

## 7. First real-run recommendation

Start with a small number of epochs, inspect train/test curves, and only then increase training duration. Keep the first run on Last.FM before moving to MovieLens or Book-Crossing.
