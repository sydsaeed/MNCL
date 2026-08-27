# Stage 2 - Noise-Enhanced LightGCN

This stage implements the user-item view encoder of MNCL.

## Files

- `models/noise_lightgcn.py`
- `models/__init__.py`
- `tests/test_noise_lightgcn.py`

## Implemented equations

- Symmetric LightGCN normalization: Eqs. (1)-(3)
- Layer aggregation: Eqs. (4)-(5)
- Noise enhancement: Eq. (6)
- Noise-enhanced propagation: Eq. (7)

The final representation is the mean of the initial embedding and all `L` propagated embeddings, following Eqs. (4)-(5), where each layer weight is `1 / (L + 1)`.

## Noise

The paper defines `gamma` as a normalized random matrix but does not specify the random distribution or normalization axis. This implementation uses uniform random values and row-wise L2 normalization:

`gamma = normalize(rand_like(x), p=2, dim=1)`

Noise is enabled automatically in training mode and disabled automatically in evaluation mode. It can be overridden with the `add_noise` argument.

## Design

The encoder does not own user/item embedding tables. It receives them as input so the final `MNCL` model can control and share base embeddings across views.

## Last.FM defaults

- `num_layers = 4`
- `beta = 1.5`

## Tests

The module was tested for:

- output shapes
- finite outputs
- deterministic evaluation
- stochastic training noise
- beta=0 behavior
- gradient flow

All five unit tests pass.
