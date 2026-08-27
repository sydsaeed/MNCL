# Stage 4 — Path-Aware GNN

This stage adds the user-item-entity view of MNCL.

## Files

- `models/path_gnn.py`: Path-Aware encoder.
- `models/path_augmentation.py`: structural sampling for the corrupted view.
- `datasets/graph_builder.py`: exposes local user-item and KG indices.
- `tests/test_path_gnn.py`: PyG runtime tests.
- `tests/test_path_augmentation.py`: structural sampling tests.

## Output

The encoder returns:

- `e_m_user`: global user representation.
- `e_m_item`: global item representation.

Both have shape `[count, embedding_dim]`.

## Path aggregation

Each hop performs two synchronized updates:

1. Users aggregate interacted item embeddings with mean aggregation.
2. KG nodes aggregate relation-aware neighbor messages.

For a KG edge, the message is `neighbor * relation`. Attention uses the concatenated node/relation representations described by Eq. 14 and is normalized over neighbors of the target node.

Layer outputs are L2-normalized and accumulated with residual summation.

## Corrupted view

The paper states that the user-item-entity view uses structural corruption. Its exact drop probabilities are not reported. The official repository exposes `context_hops=2`, `node_dropout_rate=0.5`, and `mess_dropout_rate=0.1` in `utils/parser.py`.

The official `node_dropout` implementation samples KG edges and interaction entries rather than physically deleting node tensors. This project follows that operational behavior and names it `structural_keep_ratio` to avoid ambiguity.

Defaults:

```python
path_hops = 2
path_structural_keep_ratio = 0.5
path_message_dropout_rate = 0.1
```

Structural sampling and message dropout are enabled automatically in `train()` mode and disabled in `eval()` mode. They can also be overridden in `forward()`.

## Graph IDs

`PathAwareGNN` receives local ID spaces:

- users: `0 .. M-1`
- KG nodes: `0 .. N+K-1`
- items inside KG: `0 .. N-1`
- relations: `0 .. L-1`

No item mapping is required.

## Source notes

Paper: Section 4.3, Eqs. 12-14.

Official repository inspected for implementation ambiguities:

- `model/Mymodel.py`
- `utils/parser.py`

The paper's Eq. 13 has inconsistent index notation around `u` and `j`. For the KG attention, this implementation uses head-neighbor relation attention, matching the official code's operational use of head, tail, and relation while retaining the paper's concatenation-based attention score.
