__all__ = [
    "BPRTripletDataset",
    "BPRTriplets",
    "build_bpr_dataloader",
    "build_bpr_triplets",
    "GraphBundle",
    "build_graphs",
    "DatasetBundle",
    "DatasetInfo",
    "load_dataset",
    "RatingSplit",
    "split_ratings",
]


def __getattr__(name):
    if name in {
        "BPRTripletDataset",
        "BPRTriplets",
        "build_bpr_dataloader",
        "build_bpr_triplets",
    }:
        from . import bpr_dataset
        return getattr(bpr_dataset, name)
    if name in {"GraphBundle", "build_graphs"}:
        from . import graph_builder
        return getattr(graph_builder, name)
    if name in {"DatasetBundle", "DatasetInfo", "load_dataset"}:
        from . import loader
        return getattr(loader, name)
    if name in {"RatingSplit", "split_ratings"}:
        from . import splitter
        return getattr(splitter, name)
    raise AttributeError(name)
