# Stage 5 - Multi-Negative Contrastive Loss

This stage implements Eqs. 15-17 of MNCL.

## Added files

- `losses/contrastive.py`
- `losses/__init__.py`
- `tests/test_contrastive.py`

## Loss terms

The module computes three terms:

- `L_G_user`: user-item view vs user-item-entity view for users
- `L_G_item`: user-item view vs user-item-entity view for items
- `L_KG_item`: user-item view vs item-entity view for items

The final contrastive objective is:

`L_CL = L_G_user + L_G_item + L_KG_item`

For every anchor node, the denominator contains:

1. the positive pair from the second view,
2. negatives from the original view weighted by `omega`,
3. negatives from the second view.

Cosine similarity is used. The implementation normalizes embeddings first and then applies the temperature-scaled dot product.

## Numerical stability

The paper writes the loss using exponentials. The code uses `torch.logsumexp`, which is algebraically equivalent but avoids numerical overflow.

## Temperature

The supplied paper defines the temperature `tau` but does not report its chosen value in Table 3 or the surrounding experimental settings. To keep the project runnable, `config.py` currently uses:

```python
tau = 0.2
```

This is an implementation default, not a value claimed by the paper. It should be treated as a tunable hyperparameter.

## Batch behavior

The loss accepts `[B, D]` embeddings. Negatives are all other nodes in the provided batch. Passing all users/items reproduces the full-matrix interpretation; passing sampled subsets gives a lower-memory batch approximation.
