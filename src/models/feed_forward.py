import torch
import torch.nn as nn

class PointWiseFeedForward(nn.Module):
    
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()
        self.linear1 = nn.Linear(
            embedding_dim,
            embedding_dim * 4,
        )
        self.activation = nn.ReLU()
        
        self.linear2 = nn.Linear(
            embedding_dim * 4,
            embedding_dim,
        )
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x