import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.training.evaluate import evaluate_ranking


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_norm=5.0,
):
    model.train()
    total_loss = torch.zeros((), device=device)
    progress_bar = tqdm(
        dataloader,
        desc="Training",
        unit="batch",
        leave=False,
    )
    for batch_index, batch in enumerate(progress_bar, start=1):
        input_vectors = batch["input_vectors"].to(
            device,
            non_blocking=True,
        )
        positive_vectors = batch["positive_vectors"].to(
            device,
            non_blocking=True,
        )
        negative_vectors = batch["negative_vectors"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(
            input_vectors=input_vectors,
            positive_vectors=positive_vectors,
            negative_vectors=negative_vectors,
        )

        positive_logits = outputs["positive_logits"]
        negative_logits = outputs["negative_logits"]

        loss = criterion(
            positive_logits=positive_logits,
            negative_logits=negative_logits,
            positive_vectors=positive_vectors,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
        )

        optimizer.step()

        total_loss += loss.detach()
        if batch_index % 20 == 0:
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                refresh=False,
            )

    if len(dataloader) == 0:
        raise ValueError("Cannot train with an empty dataloader")
    average_loss = total_loss.item() / len(dataloader)

    return average_loss


def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    k=10,
    patience=3,
    min_delta=1e-4,
    run_dir="outputs",
    model_config=None,
    max_norm=5.0,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    best_ndcg = float("-inf")
    best_epoch = 0
    bad_epochs = 0

    history = {
        "train_loss": [],
        "hit_rate": [],
        "ndcg": [],
    }

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_norm=max_norm,
        )

        metrics = evaluate_ranking(
            model=model,
            dataloader=validation_loader,
            device=device,
            k=k,
        )

        hit_rate = metrics[f"hit_rate@{k}"]
        ndcg = metrics[f"ndcg@{k}"]

        history["train_loss"].append(train_loss)
        history["hit_rate"].append(hit_rate)
        history["ndcg"].append(ndcg)

        logger.info(
            "Epoch %d/%d - Loss: %.4f - HR@%d: %.4f - NDCG@%d: %.4f",
            epoch + 1, num_epochs, train_loss, k, hit_rate, k, ndcg,
        )

        if ndcg > best_ndcg + min_delta:
            best_ndcg = ndcg
            best_epoch = epoch + 1
            bad_epochs = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_ndcg": best_ndcg,
                    "k": k,
                    "model_config": model_config,
                },
                checkpoint_path,
            )

            logger.info("Saved new best model at epoch %d.", best_epoch)
        else:
            bad_epochs += 1
            logger.info(
                "No improvement. Patience: %d/%d", bad_epochs, patience
            )

        if bad_epochs >= patience:
            logger.info("Early stopping at epoch %d.", epoch + 1)
            break

    return {
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
        "model_path": str(checkpoint_path),
        "history": history,
    }
