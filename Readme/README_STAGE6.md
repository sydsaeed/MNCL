# Stage 6 - Main MNCL model and prediction

This stage connects the three MNCL views into one model.

## Added

- `models/mncl.py`
- shared trainable user, entity/item, and relation embeddings
- Xavier initialization
- `encode_views()` for the three encoders
- Eq. 18-19 view fusion
- Eq. 20 dot-product prediction
- tests for fusion and pair scoring

## Shared base embeddings

Items and KG entities stay in the same ID space. The first `N` rows of the entity table are item embeddings.

```text
user_embeddings   [M, D]
entity_embeddings [N + K, D]
relation_embeddings [L, D]
```

The relation embedding table is shared by the relation-aware and path-aware encoders. The older encoders still support internal relation embeddings when used alone.

## Paper-first fusion

The default fusion follows Eqs. 18-19 in the supplied paper:

```text
user = e_s_user || e_s_user || e_m_user
item = e_s_item || e_m_item || e_g_item
```

Both final representations therefore have dimension `3 * D`.

## Prediction

For a pair `(u, i)`:

```text
score(u, i) = user_final[u]^T item_final[i]
```

No sigmoid is applied inside the model. This keeps the raw score available for the later BPR objective.

## Note on the official repository

The current official implementation concatenates its corresponding blocks in a different order from the printed Eqs. 18-19. This project keeps the paper equations as the default implementation so the choice remains explicit and reproducible.
