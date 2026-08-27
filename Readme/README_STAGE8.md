# Stage 8 - BPR Sampling and Training Loop

This stage adds the modular training pipeline requested for `main.ipynb`.

## Added files

- `datasets/bpr_dataset.py`: builds same-user `(user, positive_item, negative_item)` triplets from the preprocessed 0/1 ratings.
- `training/trainer.py`: device helpers, one-epoch train/evaluation functions, and `train_epochs` with tqdm.
- `training/__init__.py`: public training imports.

## BPR pairing

The input ratings already contain positive (`label=1`) and negative (`label=0`) samples. For each user, positive and negative rows are independently shuffled and paired. The pairing is rebuilt with an epoch-dependent seed during training so the association between a user's positive and negative examples can change between epochs.

The Last.FM split currently gives equal positive/negative counts per user, so Stage 8 produces:

- Train triplets: 16,919
- Test triplets: 4,254

If another dataset has unequal per-user counts, the smaller side is cycled so all available examples from the larger side can participate.

## Training behavior

`main.ipynb` keeps model initialization and training in separate cells.

The initialization cell creates:

```python
model
loss_fn
optimizer
total_train_loss
total_test_loss
```

The train cell only calls `train_epochs(...)`. Running it again reuses the same model and optimizer. It does not initialize MNCL again.

`start_epoch=len(total_train_loss)` keeps tqdm epoch numbering continuous across repeated executions.

Each call returns exactly:

```python
train_loss, test_loss
```

and the notebook appends them to:

```python
total_train_loss
total_test_loss
```

The next cell plots both histories together.

## Test loss definition

For now `test_loss` evaluates the same complete MNCL objective used by training on BPR triplets built from `test_ratings`. Evaluation disables noise, structural dropout, and message dropout, so repeated evaluation with unchanged parameters is deterministic.

The graph encoders still use only the training interaction graph constructed in Stage 1; test interactions are never added to message passing.

## Contrastive batches

The full graph is encoded for each optimization step, but contrastive matrices are formed only from unique users/items in the current BPR batch. This limits the quadratic contrastive memory cost while preserving cross-view negatives inside the batch.

## Not included yet

Checkpoint save/load and best-`test_loss` persistence are intentionally left for Stage 9. Stage 8 establishes the training state that Stage 9 will serialize, including model parameters, optimizer state, epoch count, and loss histories.
