# Stage 3 - Relation-Aware Item-Entity View

This stage implements the item-entity view of MNCL.

## Files

- `models/relation_gnn.py`
- `models/item_lightgcn.py`
- `tests/test_relation_gnn.py`
- `tests/test_item_lightgcn.py`
- `config.py`

## Relation-aware propagation

For each directed KG edge, the message is:

`neighbor_embedding * relation_embedding`

Messages are averaged at the destination node. For the current data pipeline, pass the bidirectional KG edges from `graphs.kg.bidirectional_edge_index` and `graphs.kg.bidirectional_edge_type`.

The relation IDs are used directly from the dataset (`0 ... L-1`).

## Semantic LightGCN

The paper defines a normalized semantic adjacency matrix `S` but does not specify how it is constructed. The public MNCL implementation builds a top-k cosine-similarity graph. This project follows that missing construction detail with `semantic_topk=10`.

The semantic graph is built over all KG nodes. The final item output is the first `N` rows because item IDs occupy `0 ... N-1` in the shared KG ID space.

The graph is built in chunks to avoid materializing the entire similarity matrix at once.

## Final output

The main output is:

`e_g_item: [N, embedding_dim]`

No noise enhancement is used in this view.

## Paper-first choices

- `K=2` remains controlled by `config.K`.
- Relation aggregation uses mean messages as written in Eq. 9.
- The semantic LightGCN sums the initial representation and the `K` propagated representations.
- Extra dropout and other implementation-specific settings from the public repository are not added at this stage because they are not stated in the paper.
