from __future__ import annotations

import numpy as np
import torch

from config import MNCLConfig
from datasets import DatasetBundle, DatasetInfo, build_graphs, split_ratings
from datasets.bpr_dataset import build_bpr_dataloader
from losses import MNCLLoss
from models import MNCL
from training import move_graphs_to_device, resolve_device
from utils import set_seed


def build_tiny_data() -> DatasetBundle:
    """Build a small dataset for an end-to-end smoke test."""
    ratings = np.array(
        [
            [0, 0, 1], [0, 1, 1], [0, 2, 0], [0, 3, 0],
            [1, 1, 1], [1, 2, 1], [1, 0, 0], [1, 3, 0],
            [2, 2, 1], [2, 3, 1], [2, 0, 0], [2, 1, 0],
        ],
        dtype=np.int64,
    )
    kg = np.array(
        [
            [0, 0, 4],
            [1, 0, 4],
            [2, 1, 5],
            [3, 1, 5],
            [4, 1, 5],
        ],
        dtype=np.int64,
    )
    info = DatasetInfo(M=3, N=4, K=2, L=2)
    return DatasetBundle(ratings=ratings, kg=kg, info=info)


def main() -> None:
    config = MNCLConfig()
    config.embedding_dim = 8
    config.batch_size = 4
    config.K = 1
    config.L = 1
    config.beta = 0.1
    config.semantic_topk = 2
    config.semantic_chunk_size = 8
    config.path_hops = 1
    config.path_structural_keep_ratio = 1.0
    config.path_message_dropout_rate = 0.0
    config.l2_lambda = 1e-6

    set_seed(config.seed)
    device = resolve_device(config.device)

    data = build_tiny_data()
    split = split_ratings(data.ratings, test_ratio=0.5, seed=config.seed)
    graphs = build_graphs(data, split)
    graphs = move_graphs_to_device(graphs, device)

    model = MNCL(data.info, config).to(device)
    loss_fn = MNCLLoss(
        alpha=config.alpha,
        l2_lambda=config.l2_lambda,
        temperature=config.tau,
        omega=config.omega,
        bpr_reduction=config.bpr_reduction,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loader = build_bpr_dataloader(
        split.train_ratings,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    user_ids, positive_item_ids, negative_item_ids = next(iter(loader))
    user_ids = user_ids.to(device)
    positive_item_ids = positive_item_ids.to(device)
    negative_item_ids = negative_item_ids.to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(graphs)

    positive_scores = (
        output.user_embeddings[user_ids]
        * output.item_embeddings[positive_item_ids]
    ).sum(dim=-1)
    negative_scores = (
        output.user_embeddings[user_ids]
        * output.item_embeddings[negative_item_ids]
    ).sum(dim=-1)

    unique_users = torch.unique(user_ids)
    unique_items = torch.unique(
        torch.cat([positive_item_ids, negative_item_ids], dim=0)
    )
    loss_output = loss_fn(
        model=model,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        views=output.views,
        contrastive_user_ids=unique_users,
        contrastive_item_ids=unique_items,
    )

    if not torch.isfinite(loss_output.total_loss):
        raise RuntimeError("Training loss is not finite.")

    loss_output.total_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_output = model(
            graphs,
            add_noise=False,
            structural_dropout=False,
            message_dropout=False,
        )

    tensors = (
        eval_output.user_embeddings,
        eval_output.item_embeddings,
        eval_output.views.e_s_user,
        eval_output.views.e_s_item,
        eval_output.views.e_m_user,
        eval_output.views.e_m_item,
        eval_output.views.e_g_item,
    )
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise RuntimeError("Non-finite values found in model outputs.")

    print("Smoke test passed")
    print("device:", device)
    print("loss:", float(loss_output.total_loss.detach()))
    print("user embedding:", tuple(eval_output.user_embeddings.shape))
    print("item embedding:", tuple(eval_output.item_embeddings.shape))


if __name__ == "__main__":
    main()
