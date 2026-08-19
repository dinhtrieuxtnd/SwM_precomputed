import torch
from torch import nn

from src.models.embedding import SASRecEmbedding
from src.models.self_attention_block import SelfAttentionBlock


def sequence_padding_mask(
    input_vetors: torch.Tensor
) -> torch.Tensor:
    if input_vetors.ndim != 3:
        raise ValueError("input_vetors must have shape [B, L, T]")
    return input_vetors.ne(0).any(dim=-1)


def create_attention_mask(
    input_vectors: torch.Tensor
) -> torch.Tensor:
    padding_mask = sequence_padding_mask(
        input_vectors
    )
    sequence_length = input_vectors.size(1)
    causal_mask = torch.tril(torch.ones(
        sequence_length, sequence_length,
        dtype=torch.bool, device=input_vectors.device,
    ))
    return causal_mask.unsqueeze(0) & padding_mask.unsqueeze(1)


class SASRec(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = SASRecEmbedding(
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)

    def _run_blocks(
        self,
        input_vector: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        
        padding_mask = sequence_padding_mask(
            input_vector)
        attention_mask = create_attention_mask(
            input_vector)
        
        x = self.embedding(
            input_vector,
        )
        
        all_attention_weights = []
        
        for block in self.blocks:
            x, attention_weights = block(
                x=x,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )
            
            all_attention_weights.append(attention_weights)
            
        x = self.final_norm(x) * padding_mask.unsqueeze(-1)
        
        return x, all_attention_weights

    def score_candidates(
        self,
        input_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor
    ) -> torch.Tensor:
        if candidate_vectors.ndim != 3:
            raise ValueError(
                "candidate_vectors must have shape [B, C, T]"
            )
        sequence_output, _ = self._run_blocks(input_vectors)
        padding_mask = sequence_padding_mask(input_vectors)
        positions = torch.arange(
            input_vectors.size(1),
            device=input_vectors.device)
        last_indices = (positions
                        .unsqueeze(0)
                        .masked_fill(~padding_mask, -1)
                        .max(1).values)
        if torch.any(last_indices < 0):
            raise ValueError(
                "Every input sequence must contain a non-padding article"
            )
        last_hidden = sequence_output[
            torch.arange(
                input_vectors.size(0),
                device=input_vectors.device
            ),
            last_indices
        ]
        return (
            candidate_vectors * last_hidden.unsqueeze(1)
        ).sum(dim=-1)

    def forward(
        self,
        input_vectors: torch.Tensor,
        positive_vectors: torch.Tensor,
        negative_vectors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sequence_output, attention_weights = self._run_blocks(
            input_vectors
        )
        return {
            "positive_logits": (
                sequence_output * positive_vectors
                ).sum(dim=-1),
            "negative_logits": (
                sequence_output * negative_vectors
                ).sum(dim=-1),
            "sequence_output": sequence_output,
            "attention_weights": attention_weights,
        }
