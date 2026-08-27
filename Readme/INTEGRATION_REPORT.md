# Integration report

## Scope

Final audit of the modular MNCL implementation for the supplied Last.FM data and MNCL paper.

## Static verification

- All Python source files compile successfully.
- `main.ipynb` is valid notebook JSON.
- The notebook contains no training/model helper function definitions; it remains an orchestration notebook.
- Model initialization and the train block remain separate, so rerunning the train block does not reinitialize the model.
- Best/latest checkpoint paths remain separate.
- Train block still returns `train_loss` and `test_loss`, which are appended to `total_train_loss` and `total_test_loss`.

## Unit-test result in the build environment

```text
41 passed, 4 skipped
```

The skipped tests require `torch_geometric`.

## Data verification

```text
ratings_final.npy: (42346, 3)
kg_final.npy:      (15518, 3)
```

Both files are readable and have the expected three-column structure.

## Environment status in the build environment

```text
torch:           available
torch_geometric: unavailable
numpy:           available
tqdm:            available
matplotlib:      available
```

The build environment has no network access, so PyG could not be installed here. A full end-to-end smoke test is included as `smoke_check.py` and should be run on the target training machine.

## Final source-choice audit

The final implementation remains paper-first. See `IMPLEMENTATION_NOTES.md` for differences between the paper and the public author repository.
