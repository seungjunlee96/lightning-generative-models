from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    """Vector Quantizer module as described in VQ-VAE.

    Args:
        embedding_dim: Dimensionality of the tensors in the quantized space.
        num_embeddings: Number of vectors in the quantized space.
        commitment_cost: Controls the weighting of the loss terms.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self._init_embedding()

    def forward(self, latents: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        quantized_latents, encodings = self._quantize(latents)
        vq_loss = self._calculate_vq_loss(latents, quantized_latents)
        quantized_latents = self._straight_through_estimator(latents, quantized_latents)
        perplexity = self._calculate_perplexity(encodings)
        return quantized_latents, vq_loss, perplexity

    def _init_embedding(self):
        """Initialize embedding."""
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings,
            1 / self.num_embeddings,
        )

    def _quantize(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize the latents by replacing each with its closest code."""
        # Flatten latents for easier computation of distances to the codebook
        B, D, H, W = latents.shape
        flat_latents = latents.permute(0, 2, 3, 1).reshape(B * H * W, D)

        # Compute distances from each latent to each entry in the codebook
        # This is to determine which code in the codebook is closest to each latent
        distances = (
            (flat_latents**2).sum(dim=1, keepdim=True)
            + (self.embedding.weight**2).sum(dim=1)
            - 2 * flat_latents @ self.embedding.weight.T
        )

        # Find the closest code for each latent
        encoding_indices = distances.argmin(dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized_latents = (
            self.embedding(encoding_indices)
            .reshape(B, H, W, D)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return quantized_latents, encodings

    def _calculate_vq_loss(self, latents: Tensor, quantized_latents: Tensor) -> Tensor:
        """
        The VQ loss ensures that the quantized latents are close to the original latents
        and penalizes large deviations. This helps the model to learn meaningful codes.
        """
        e_latent_loss = F.mse_loss(quantized_latents, latents.detach())
        q_latent_loss = F.mse_loss(quantized_latents.detach(), latents)
        return e_latent_loss + self.commitment_cost * q_latent_loss

    def _calculate_perplexity(
        self, encodings: Tensor, epsilon: float = 1e-10
    ) -> Tensor:
        """
        Perplexity provides a measure of how well the model uses its codebook.
        Higher perplexity means the model uses more of the codebook, which is desirable.
        """
        avg_probs = torch.mean(encodings, dim=0)
        return torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + epsilon)))

    @staticmethod
    def _straight_through_estimator(latents, quantized_latents):
        """Enables gradient propagation through the discrete latent space via the straight-through estimator."""
        return latents + (quantized_latents - latents).detach()


class VectorQuantizerEMA(VectorQuantizer):
    """Vector Quantizer with Exponential Moving Average (EMA) as described in VQ-VAE-2.
        EMA stabilizes the embeddings' learning, which can otherwise be very noisy,
        leading to unstable training and poor convergence.

    Args:
        embedding_dim: Dimensionality of the tensors in the quantized space.
        num_embeddings: Number of vectors in the quantized space.
        commitment_cost: Controls the weighting of the loss terms.
        decay: Decay factor for EMA.
        epsilon: Small value to avoid division by zero.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__(
            num_embeddings, embedding_dim, commitment_cost
        )

        # Buffers for EMA: these maintain the average of the assigned latents and the average size
        # of each cluster, aiding in the stable learning of the codebook.
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_embedding", self.embedding.weight.data.clone())
        self.decay = decay
        self.epsilon = epsilon

    @torch.no_grad()
    def _ema_update(self, encodings: Tensor, latents: Tensor):
        """Update the codebook embeddings using Exponential Moving Averages."""
        B, D, H, W = latents.shape
        flat_latents = latents.permute(0, 2, 3, 1).reshape(B * H * W, D)
        self._ema_cluster_size.mul_(self.decay).add_(
            encodings.sum(0),
            alpha=1 - self.decay,
        )
        n = self._ema_cluster_size.sum()
        cluster_weights = (
            (self._ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        dw = torch.matmul(encodings.T, flat_latents)
        self._ema_embedding.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        self.embedding.weight.data.copy_(
            self._ema_embedding / cluster_weights.unsqueeze(1)
        )

    def _quantize(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantizes the latents and performs EMA updates if the module is in training mode."""
        # Flatten latents for easier computation of distances to the codebook
        B, D, H, W = latents.shape
        flat_latents = latents.permute(0, 2, 3, 1).reshape(B * H * W, D)

        # Compute distances from each latent to each entry in the codebook
        # This is to determine which code in the codebook is closest to each latent
        distances = (
            (flat_latents**2).sum(dim=1, keepdim=True)
            + (self.embedding.weight**2).sum(dim=1)
            - 2 * flat_latents @ self.embedding.weight.T
        )

        # Find the closest code for each latent
        encoding_indices = distances.argmin(dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Update the codebook embeddings using Exponential Moving Averages if training.
        if self.training:
            self._ema_update(encodings, latents)

        # Quantize
        quantized_latents = (
            self.embedding(encoding_indices)
            .reshape(B, H, W, D)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return quantized_latents, encodings
