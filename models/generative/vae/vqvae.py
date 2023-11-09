import os
from typing import Dict, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch.optim import Adam
from torchvision.utils import make_grid

from models.modules.residual import ResidualStack
from models.modules.vector_quantizer import VectorQuantizer, VectorQuantizerEMA


class Encoder(nn.Module):
    """
    Encoder module for VQ-VAE.

    This module is responsible for compressing the input image into a lower-dimensional
    latent space. The output of the encoder is then quantized to produce discrete latent
    representations.
    """

    def __init__(
        self,
        img_channels: int,
        embedding_dim: int,
        hidden_dim: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim // 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            ResidualStack(
                hidden_dim,
                hidden_dim,
                num_residual_layers,
                num_residual_hiddens,
            ),
            nn.Conv2d(hidden_dim, embedding_dim, 1),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class Decoder(nn.Module):
    """
    Decoder module for VQ-VAE.

    This module decodes quantized latents back into images.
    """

    def __init__(
        self,
        img_channels: int,
        embedding_dim: int,
        hidden_dim: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, 1, 1),
            ResidualStack(
                hidden_dim, hidden_dim, num_residual_layers, num_residual_hiddens
            ),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class VQVAE(pl.LightningModule):
    """
    Vector-Quantized Variational Autoencoder (VQ-VAE) model.

    VQ-VAE is a generative model that learns a discrete representation of the data.
    This implementation uses convolutional layers for encoding and decoding and a vector
    quantization layer in the middle.

    Reference:
        [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
        [2] VQ-VAE https://arxiv.org/abs/1711.00937
        [3] https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
        [4] https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        hidden_dim: int = 256,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 256,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
        ckpt_path: str = "",
        loss_weights: Dict = {
            "recon_loss": 1.0,
            "vq_loss": 1.0,
        },
    ) -> None:
        super(VQVAE, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            img_channels=img_channels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self.decoder = Decoder(
            img_channels=img_channels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        if use_ema:
            self.vector_quantizer = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                epsilon=epsilon,
            )

        else:
            self.vector_quantizer = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
            )

        if os.path.exists(ckpt_path):
            self.load_from_checkpoint(ckpt_path)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        latents = self.encoder(x)
        quantized_latents, vq_loss, perplexity = self.vector_quantizer(latents)
        return self.decoder(quantized_latents), vq_loss, perplexity

    def _common_step(self, batch, batch_idx: int, split: str):
        x, _ = batch

        # Forward
        x_hat, vq_loss, perplexity = self(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss_dict = {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }
        loss = sum(
            [self.hparams.loss_weights[k] * loss for k, loss in loss_dict.items()]
        )

        # Logging
        self.log_dict(
            {
                f"{split}_loss": loss,
                f"{split}_recon_loss": recon_loss,
                f"{split}_vq_loss": vq_loss,
                f"{split}_perplexity": perplexity,
            },
            on_step=True if self.training else False,
            on_epoch=False if self.training else True,
        )
        if batch_idx == 0 and not self.training:
            self._log_images(torch.cat([x, x_hat], dim=0), "Reconstruction")
            self._log_images(self.random_sample(x), "Random Sample")
            self._log_embedding()
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def _log_images(self, images: Tensor, fig_name: str):
        # Normalize the images to the range [0, 255] for visualization
        images = ((images + 1.0) / 2.0) * 255.0
        images = images.clamp(0, 255).byte().detach().cpu()

        # Create a grid of images and log it
        fig = make_grid(images)
        self.logger.experiment.log(
            {fig_name: [wandb.Image(fig)]}, step=self.global_step
        )

    @torch.no_grad()
    def random_sample(self, x):
        """
        Generate and log images from the embeddings using the decoder.

        Args:
            num_samples (int): Number of samples to generate.
        """
        latents = self.encoder(x)
        B, D, H, W = latents.shape

        # Randomly select embedding indices
        encoding_indices = torch.randint(
            0, self.hparams.num_embeddings, (B * H * W,), device=self.device
        )

        # Map indices to their corresponding embeddings
        quantized_latents = (
            self.vector_quantizer.embedding(encoding_indices)
            .reshape(B, H, W, D)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return self.decoder(quantized_latents)

    @torch.no_grad()
    def _log_embedding(self):
        embedding = self.vector_quantizer.embedding.weight.cpu()
        data = pd.DataFrame(
            {f"dim_{i}": embedding[:, i] for i in range(embedding.size(1))}
        )
        self.logger.experiment.log(
            {"embedding": wandb.Table(data=data)},
            step=self.global_step,
        )
