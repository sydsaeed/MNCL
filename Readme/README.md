# MNCL — modular PyTorch/PyG implementation

Final integrated project for the supplied MNCL paper.

## Project structure

```text
MNCL/
├── main.ipynb
├── config.py
├── verify_environment.py
├── smoke_check.py
├── IMPLEMENTATION_NOTES.md
├── FINAL_CHECKLIST.md
├── data/
│   ├── movie/
│   ├── last-fm/
│   └── book/
├── datasets/
├── models/
├── losses/
├── training/
├── utils/
├── tests/
└── checkpoints/
```

Each dataset directory should contain:

```text
ratings_final.npy
kg_final.npy
```

## Main modules

- `datasets/loader.py`: load and validate NPY files.
- `datasets/splitter.py`: reproducible train/test split.
- `datasets/graph_builder.py`: build the three MNCL graph views.
- `models/noise_lightgcn.py`: noise-enhanced LightGCN.
- `models/relation_gnn.py`: item-entity relation-aware view.
- `models/path_gnn.py`: user-item-entity path-aware view.
- `losses/contrastive.py`: multi-negative contrastive loss.
- `losses/mncl_objective.py`: BPR + contrastive + L2 objective.
- `training/trainer.py`: tqdm training loop and loss histories.
- `training/checkpoint.py`: best/latest checkpoint and resume.
- `training/evaluator.py`: AUC/F1 CTR evaluation.

## Last.FM paper defaults

```text
embedding_dim = 64
batch_size = 4096
alpha = 0.1
K = 2
L = 4
beta = 1.5
omega = 0.8
```

`tau=0.6` is taken from the public author implementation because the supplied paper does not report a value for `tau`.

## Verification

Run:

```bash
python verify_environment.py --dataset last-fm
python smoke_check.py
pytest -q
```

Then use `main.ipynb` for training, resume, plotting, and evaluation.

See `IMPLEMENTATION_NOTES.md` before comparing results with the paper or author repository.
