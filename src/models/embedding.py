import torch
from torch import nn


class SASRecEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        max_sequence_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        
    def forward(
        self,
        input_vectors: torch.Tensor
    ) -> torch.Tensor:
        x = input_vectors + self.position_embedding(
            torch.arange(input_vectors.size(1), device=input_vectors.device)
        ).unsqueeze(0)
        x = self.dropout(x)
        return x
