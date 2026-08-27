__all__ = [
    "ItemEntityLightGCN",
    "MNCL",
    "MNCLOutput",
    "MNCLViewEmbeddings",
    "NoiseEnhancedLightGCN",
    "PathAwareGNN",
    "RelationAwareGNN",
    "fuse_view_embeddings",
    "pair_scores",
]


def __getattr__(name):
    if name == "ItemEntityLightGCN":
        from .item_lightgcn import ItemEntityLightGCN
        return ItemEntityLightGCN
    if name in {
        "MNCL",
        "MNCLOutput",
        "MNCLViewEmbeddings",
        "fuse_view_embeddings",
        "pair_scores",
    }:
        from .mncl import (
            MNCL,
            MNCLOutput,
            MNCLViewEmbeddings,
            fuse_view_embeddings,
            pair_scores,
        )
        return {
            "MNCL": MNCL,
            "MNCLOutput": MNCLOutput,
            "MNCLViewEmbeddings": MNCLViewEmbeddings,
            "fuse_view_embeddings": fuse_view_embeddings,
            "pair_scores": pair_scores,
        }[name]
    if name == "NoiseEnhancedLightGCN":
        from .noise_lightgcn import NoiseEnhancedLightGCN
        return NoiseEnhancedLightGCN
    if name == "PathAwareGNN":
        from .path_gnn import PathAwareGNN
        return PathAwareGNN
    if name == "RelationAwareGNN":
        from .relation_gnn import RelationAwareGNN
        return RelationAwareGNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
