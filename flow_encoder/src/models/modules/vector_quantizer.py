from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self, n_embeddings: int, embedding_dim: int, commitment_cost: float
    ):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._n_embeddings = n_embeddings

        self._embedding = nn.Embedding(self._n_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._n_embeddings, 1 / self._n_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # convert inputs from BCTHW -> BTHWC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._n_embeddings,
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )

        # Loss
        e_latent_loss = F.mse_loss(quantized, inputs)
        q_latent_loss = F.mse_loss(quantized, inputs)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        # convert quantized from BTHWC -> BCTHW
        return (
            loss,
            quantized.permute(0, 4, 1, 2, 3).contiguous(),
            perplexity,
            encodings,
        )
