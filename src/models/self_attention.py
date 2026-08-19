import torch
import torch.nn as nn

from src.models.attention import ScaledDotProductAttention

class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.attention = ScaledDotProductAttention(
            dropout=dropout
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        return output, attention_weights