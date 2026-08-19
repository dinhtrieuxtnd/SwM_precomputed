import math
import torch
from torch import device, nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        scores = scores / math.sqrt(d_k)

        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask == 0,
                -1e9,
            )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = torch.matmul(
            attention_weights,
            value,
        )

        return output, attention_weights
    
def create_causal_mask(
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            device=device,
            dtype=torch.bool,
        )
    )

    return mask

def create_attention_mask(
    item_ids: torch.Tensor,
    padding_id: int = 0,
) -> torch.Tensor:
    _, sequence_length = item_ids.shape

    causal_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=item_ids.device,
        )
    )

    key_padding_mask = (
        item_ids != padding_id
    ).unsqueeze(1)

    combined_mask = (
        causal_mask.unsqueeze(0)
        & key_padding_mask
    )

    return combined_mask