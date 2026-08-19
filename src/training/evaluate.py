import torch
from tqdm.auto import tqdm

from src.utils.metrics import calculate_ranking_metrics


def evaluate_ranking(
    model,
    dataloader,
    device,
    k=10,
):
    model.eval()

    total_hits = 0.0
    total_ndcg = 0.0
    total_samples = 0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            unit="batch",
            leave=False,
        )

        for batch in progress_bar:

            input_vectors = batch[
                "input_vectors"
            ].to(device)

            candidate_vectors = batch[
                "candidate_vectors"
            ].to(device)

            scores = model.score_candidates(
                input_vectors=input_vectors,
                candidate_vectors=candidate_vectors,
            )

            metrics = calculate_ranking_metrics(
                scores=scores,
                k=k,
            )

            batch_size = input_vectors.size(0)
            total_hits += (
                metrics["hit_rate"] * batch_size
            )
            total_ndcg += (
                metrics["ndcg"] * batch_size
            )
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataloader")

    return {
        f"hit_rate@{k}": (
            total_hits / total_samples
        ),
        f"ndcg@{k}": (
            total_ndcg / total_samples
        ),
    }
