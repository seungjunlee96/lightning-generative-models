from typing import List

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits

from models.generative.gan.dcgan import DCGAN


class R1GAN(DCGAN):
    """
    Implements R1GAN, an extension of the DCGAN architecture with R1 regularization
    for improved training stability and convergence. R1 regularization applies a
    gradient penalty on the real data distribution to stabilize the discriminator
    updates.

    Reference:
      Mescheder, Geiger, and Nowozin, "Which Training Methods for GANs do actually Converge?",
      https://arxiv.org/pdf/1801.04406.pdf

    Parameters:
    - img_channels (int): Number of channels in the images.
    - img_size (int): Size of each image dimension.
    - latent_dim (int): Dimensionality of the latent space.
    - lr (float): Learning rate for the optimizer.
    - b1 (float): Beta1 parameter for the Adam optimizer.
    - b2 (float): Beta2 parameter for the Adam optimizer.
    - weight_decay (float): Weight decay for the optimizer.
    - r1_penalty (float, optional): Weight of the R1 regularization term. Default: 10.0.
    - calculate_metrics (bool, optional): Flag to enable or disable metric calculation.

    The class inherits from DCGAN, adding the R1 regularization term to the
    discriminator's loss during training.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
        lr: float,
        b1: float,
        b2: float,
        weight_decay: float,
        r1_penalty: float = 10.0,
        calculate_metrics: bool = False,
        metrics: List[str] = [],
    ) -> None:
        super().__init__(
            img_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
            lr=lr,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            calculate_metrics=calculate_metrics,
            metrics=metrics,
        )
        self.save_hyperparameters()

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        # Calculate standard discriminator losses
        logits_real = self.D(x)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))
        logits_fake = self.D(x_hat.detach())
        d_loss_fake = bce_with_logits(logits_fake, torch.zeros_like(logits_fake))

        # Average the real and fake loss
        d_loss = (d_loss_real + d_loss_fake) / 2

        # R1 Penalty calculation needs to enable gradient computation explicitly
        with torch.enable_grad():
            x.requires_grad_(True)
            logits_real_for_grad = self.D(x)
            grad_real = torch.autograd.grad(outputs=logits_real_for_grad.sum(), inputs=x, create_graph=True)[0]
            r1_penalty = 0.5 * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

            # Disable gradient computation for x to prevent impacting subsequent operations
            x.requires_grad_(False)

        # Add the R1 penalty to the discriminator loss
        d_loss += self.hparams.r1_penalty * r1_penalty

        # Compile loss components into a dictionary
        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "r1_penalty": r1_penalty,
            "logits_real": logits_real.mean(),
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict
