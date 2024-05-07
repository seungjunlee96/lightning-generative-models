from typing import List

import torch
from torch import Tensor

from models.generative.gan.dcgan import DCGAN


class LSGAN(DCGAN):
    """
    Least Squares Generative Adversarial Network (LSGAN):

    Traditional GANs utilize the binary cross-entropy loss, which can lead to the
    vanishing gradients problem during training. LSGANs address this issue by
    using a least squares (or quadratic) loss for the discriminator, which not
    only ensures more stable training dynamics but also tends to produce higher
    quality generated images.

    The least squares loss pushes generated samples toward the decision boundary
    of the discriminator. In this way, LSGANs penalize samples that are far from
    the decision boundaries more heavily than samples that are close, compelling
    the generator to produce samples that lie closer to the true data distribution.

    Reference: "Least Squares Generative Adversarial Networks" by Xudong Mao et al.
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        latent_dim: int = 100,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
        calculate_metrics: bool = False,
        metrics: List[str] = [],
        summary: bool = True,
    ) -> None:
        super(LSGAN, self).__init__(
            img_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
            lr=lr,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            calculate_metrics=calculate_metrics,
            metrics=metrics,
            summary=summary,
        )

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """
        Calculate the discriminator's loss based on the least squares GAN objective.

        The discriminator aims to assign high scores to real samples and low scores
        to generated samples. The quadratic loss ensures more penalty for samples that
        are far off from these targets.
        """
        # Real images
        logits_real = self.D(x)
        d_loss_real = 0.5 * torch.mean((logits_real - 1) ** 2)

        # Generated images
        logits_fake = self.D(x_hat.detach())
        d_loss_fake = 0.5 * torch.mean(logits_fake**2)

        # Combine the losses
        d_loss = d_loss_real + d_loss_fake

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "logits_real": logits_real.mean(),
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict

    def _calculate_g_loss(self, x_hat: Tensor) -> Tensor:
        """
        Calculate the generator's loss based on the least squares GAN objective.

        The generator aims to produce samples that the discriminator believes to be real,
        i.e., the discriminator assigns them high scores. The quadratic loss ensures that
        the generator is penalized more when it produces samples that are far from the
        discriminator's decision boundary.
        """
        logits_fake = self.D(x_hat)
        g_loss = 0.5 * torch.mean((logits_fake - 1) ** 2)

        loss_dict = {
            "g_loss": g_loss,
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict
