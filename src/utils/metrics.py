import math
import torch

def hit_rate_at_k(
    rank: int,
    k: int
) -> float:
    return float(rank <= k)

def ndcg_at_k(
    rank: int,
    k: int
) -> float:
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def get_positive_rank(
    scores: torch.Tensor,
    positive_index: int = 0
) -> int:
    
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional tensor")
    if not 0 <= positive_index < scores.numel():
        raise ValueError("positive_index is outside the candidate range")

    rank_indices = torch.argsort(
        scores,
        descending=True
    )
    
    positive = (
        rank_indices == positive_index
    ).nonzero(
        as_tuple=True
    )[0].item()
    
    return positive + 1

def calculate_ranking_metrics(
    scores: torch.Tensor,
    k: int,
):
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, candidates]")
    if scores.size(0) == 0 or scores.size(1) == 0:
        raise ValueError("scores must contain at least one sample and candidate")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    positive_scores = scores[:, 0].unsqueeze(1)

    ranks = (
        scores > positive_scores
    ).sum(dim=1) + 1

    hits = (
        ranks <= k
    ).float()

    ndcg = torch.where(
        ranks <= k,
        1.0 / torch.log2(
            ranks.float() + 1.0
        ),
        torch.zeros_like(
            ranks,
            dtype=torch.float,
        ),
    )

    return {
        "hit_rate": hits.mean().item(),
        "ndcg": ndcg.mean().item(),
        "ranks": ranks,
    }
