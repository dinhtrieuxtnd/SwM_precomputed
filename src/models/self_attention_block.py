import torch
from torch import nn

from src.models.multi_head_attention import MultiHeadSelfAttention
from src.models.feed_forward import (
    PointWiseFeedForward,
)


class SelfAttentionBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            embedding_dim,
        )

        self.self_attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attention_dropout = nn.Dropout(
            dropout,
        )

        self.feed_forward_norm = nn.LayerNorm(
            embedding_dim,
        )

        self.feed_forward = PointWiseFeedForward(
            embedding_dim=embedding_dim,
        )

        self.feed_forward_dropout = nn.Dropout(
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attention_input = self.attention_norm(x)

        attention_output, attention_weights = (
            self.self_attention(
                x=attention_input,
                attention_mask=attention_mask,
            )
        )

        x = x + self.attention_dropout(
            attention_output
        )

        feed_forward_input = (
            self.feed_forward_norm(x)
        )

        feed_forward_output = self.feed_forward(
            feed_forward_input
        )

        x = x + self.feed_forward_dropout(
            feed_forward_output
        )

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)

        return x, attention_weights
