# Stage 7 - BPR and complete MNCL objective

This stage implements Eqs. 21-22 from the supplied MNCL paper.

## Added

- `losses/bpr.py`
- `losses/regularization.py`
- `losses/mncl_objective.py`
- tests for BPR, L2 regularization, and the complete objective

## Eq. 21 - BPR

For paired positive and negative item scores:

```text
L_BPR = -log sigmoid(score_pos - score_neg)
```

The implementation uses the equivalent stable expression:

```text
softplus(score_neg - score_pos)
```

The paper writes a sum over samples. The code supports `sum`, `mean`, and `none`; the project default is `mean` for minibatch training.

## Eq. 22 - complete objective

```text
L_MNCL = L_BPR + alpha * L_CL + lambda * ||Theta||_2^2
```

`MNCLLoss` returns all four values:

```text
total_loss
bpr_loss
contrastive_loss
l2_loss
```

## L2 coefficient

The paper reports a search grid for L2 regularization but does not state the selected Last.FM value in the supplied text. The project therefore uses `1e-5` only as an implementation default and keeps it configurable as `config.l2_lambda`.

## Contrastive scope

By default, the loss can use all user/item view embeddings. Optional user/item ID tensors can restrict contrastive learning to nodes from the current minibatch. This makes the later trainer able to trade exact all-node negatives for lower memory use without changing the loss API.
