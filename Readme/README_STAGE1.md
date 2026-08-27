# MNCL - Stage 1: Data Pipeline

This stage implements dataset loading, validation, user-wise stratified train/test splitting, and the three graph containers required by MNCL.

## Current structure

```text
mncl_stage1/
├── config.py
├── data/
│   └── last-fm/
│       ├── ratings_final.npy
│       └── kg_final.npy
├── datasets/
│   ├── __init__.py
│   ├── loader.py
│   ├── splitter.py
│   └── graph_builder.py
├── utils/
│   ├── __init__.py
│   └── seed.py
└── requirements.txt
```

## Quick usage

```python
from config import MNCLConfig
from datasets import build_graphs, load_dataset, split_ratings
from utils import set_seed

config = MNCLConfig(dataset="last-fm")
set_seed(config.seed)

data = load_dataset(config)
split = split_ratings(
    data.ratings,
    test_ratio=config.test_ratio,
    seed=config.seed,
)
graphs = build_graphs(data, split)

print(data.info)
print(split.train_ratings.shape)
print(split.test_ratings.shape)
print(graphs.user_item)
print(graphs.kg)
print(graphs.user_item_entity)
```

## Split note

The paper does not specify the train/test split ratio in the supplied text. The current `test_ratio=0.2` is therefore an implementation choice and is configurable in `MNCLConfig`.
