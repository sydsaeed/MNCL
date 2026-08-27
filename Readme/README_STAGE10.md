# Stage 10 - CTR Evaluation

This stage adds deterministic CTR evaluation for MNCL.

## New modules

- `utils/metrics.py`
  - ROC-AUC without an extra scikit-learn dependency
  - F1, precision, and recall
- `training/evaluator.py`
  - deterministic prediction over every test interaction
  - sigmoid conversion from raw dot-product scores to probabilities
  - best-checkpoint evaluation

## Evaluation flow

1. Load the best checkpoint into a separate evaluation model.
2. Run all three graph views with noise/dropout disabled.
3. Compute fused user/item embeddings once.
4. Score every row of `test_ratings` in batches.
5. Apply sigmoid to raw scores.
6. Compute global AUC and F1 on the complete test split.

`f1_threshold=0.5` is configurable in `config.py`.

## Paper vs released repository

The paper describes CTR evaluation with AUC and F1 and defines F1 from precision and recall. The released repository applies sigmoid to dot-product scores and uses a `0.5` threshold for F1. This stage follows that scoring convention while computing AUC/F1 globally over the entire test split.

The released repository averages per-batch AUC/F1 values. This project intentionally uses global metrics because they represent the complete test interaction set and do not depend on evaluation batch boundaries.

## Notebook behavior

The evaluation block creates a separate `evaluation_model`, loads `*_best.pt`, and leaves the training model/optimizer untouched. This means you can evaluate the best checkpoint and still continue training from the current/latest state.
