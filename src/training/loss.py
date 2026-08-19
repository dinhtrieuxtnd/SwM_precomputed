import torch
from torch import nn

class SASRecLoss(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
        )
        
    def forward(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        positive_vectors: torch.Tensor,
    ) -> torch.Tensor:
        
        positive_labels = torch.ones_like(positive_logits)
        negative_labels = torch.zeros_like(negative_logits)
        
        positive_loss = self.loss_fn(
            input=positive_logits,
            target=positive_labels,
        )
        
        negative_loss = self.loss_fn(
            input=negative_logits,
            target=negative_labels,
        )
        
        if positive_logits.shape != negative_logits.shape:
            raise ValueError("positive_logits and negative_logits must have equal shape")
        if positive_vectors.ndim != positive_logits.ndim + 1:
            raise ValueError("positive_vectors must have shape [B, L, T]")
        if positive_vectors.shape[:-1] != positive_logits.shape:
            raise ValueError("Token batch/sequence dimensions must match logits")

        valid_mask = positive_vectors.ne(
            0
        ).any(dim=-1).to(positive_loss.dtype)
        
        loss = (positive_loss + negative_loss) * valid_mask
        loss = loss.sum() / valid_mask.sum().clamp_min(1.0)
        
        return loss
