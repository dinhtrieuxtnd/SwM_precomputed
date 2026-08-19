import torch
from torch import nn

from src.models.attention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.attention = ScaledDotProductAttention(
            dropout=dropout
        )

    def split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def combine_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2)

        return x.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask is not None and attention_mask.ndim != 4:
            raise ValueError(
                "attention_mask must have shape [B,L,L] or [B,H,L,L]")

        context, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        context = self.combine_heads(context)

        output = self.output_projection(context)

        return output, attention_weights
