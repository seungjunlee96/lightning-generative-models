from typing import List, Tuple

import torch
from torch import Tensor
from torch.optim import Adam, RMSprop

from models.generative.gan.dcgan import DCGAN


class WGAN(DCGAN):
    """
    Wasserstein Generative Adversarial Network (WGAN).

    WGAN is an alternative to traditional GAN training. It provides more stable learning,
    avoids mode collapse, and offers meaningful learning curves useful for debugging and hyperparameter tuning.
    This implementation allows for two methods to enforce the 1-Lipschitz constraint:
    gradient penalty (gp) or weight clipping (clip).
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        latent_dim: int = 100,
        lr: float = 0.00005,
        weight_decay: float = 0,
        b1: float = 0.5,
        b2: float = 0.9,
        n_critic: int = 5,
        clip_value: float = 0.01,
        grad_penalty: float = 10,
        constraint_method: str = "gp",
        calculate_metrics: bool = False,
        metrics: List[str] = [],
        summary: bool = True,
    ) -> None:
        super(WGAN, self).__init__(
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
        assert constraint_method in [
            "gp",
            "clip",
        ], "Either gradient penalty (gp) or weight clipping (clip) to enforce 1-Lipschitz constraint."
        self.clip_value = clip_value
        self.grad_penalty = grad_penalty
        self.constraint_method = constraint_method
        self.save_hyperparameters()

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        x, _ = batch
        x_hat = self.G.random_sample(x.size(0))
        d_optim, g_optim = self.optimizers()

        # Train Discriminator
        if (self.global_step + 1) % (self.hparams.n_critic + 1) != 0:
            loss_dict = self._calculate_d_loss(x, x_hat)
            d_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["d_loss"])
            d_optim.step()

        # Train Generator
        else:
            loss_dict = self._calculate_g_loss(x_hat)
            g_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["g_loss"])
            g_optim.step()

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        d_loss_real = self.D(x).mean()
        d_loss_fake = self.D(x_hat.detach()).mean()
        d_loss = d_loss_fake - d_loss_real

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
        }

        if self.training:
            if self.hparams.constraint_method == "gp":
                gradient_penalty = self._calculate_gradient_penalty(x, x_hat)
                d_loss += gradient_penalty
                loss_dict["gradient_penalty"] = gradient_penalty

            elif self.hparams.constraint_method == "clip":
                self._weight_clipping()

            else:
                raise ValueError(
                    f"{self.hparams.constraint_method} not expected, "
                    "constraint method must be either 'gp' for gradient penalty "
                    "or 'clip' for weight clipping."
                )
        return loss_dict

    def _calculate_g_loss(self, x_hat: Tensor) -> Tensor:
        g_loss = -self.D(x_hat).mean()
        loss_dict = {"g_loss": g_loss}
        return loss_dict

    def _calculate_gradient_penalty(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """
        Calculates the gradient penalty for WGAN-GP.

        The gradient penalty ensures the discriminator's gradients have a norm close to 1,
        enforcing the Lipschitz constraint. This results in a more stable training process and
        mitigates the issue of mode collapse.

        Args:
            x (Tensor): A batch of real images from the dataset.
            x_hat (Tensor): A batch of images produced by the generator.

        Returns:
            Tensor: The computed gradient penalty.
        """

        # Generate random tensor for interpolation
        alpha = torch.rand(x.size(0), 1, 1, 1, device=self.device)

        # Create interpolated samples by blending real and generated images
        interpolated_images = alpha * x + (1 - alpha) * x_hat
        interpolated_images.requires_grad_(True)

        # Compute the discriminator's scores for interpolated samples
        scores_on_interpolated = self.D(interpolated_images)

        # Calculate gradients of the scores with respect to the interpolated images
        gradients = torch.autograd.grad(
            outputs=scores_on_interpolated,
            inputs=interpolated_images,
            grad_outputs=torch.ones_like(scores_on_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute the gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        grad_penalty = ((gradient_norm - 1) ** 2).mean()

        return grad_penalty * self.hparams.grad_penalty

    def _weight_clipping(self):
        """
        Clip the discriminator's weights for stable training,
        which ensures the discriminator's gradients are bounded.
        A crude way to enforce the 1-Lipschitz constraint on the critic.
        """
        for param in self.D.parameters():
            param.data.clamp_(
                -self.hparams.clip_value,
                self.hparams.clip_value,
            )

    def configure_optimizers(self):
        if self.hparams.constraint_method == "clip":
            # Empirically the authors recommended RMSProp optimizer on the critic,
            # rather than a momentum based optimizer such as Adam which could cause instability in the model training.
            d_optim = RMSprop(
                self.D.parameters(),
                lr=self.hparams.lr,
            )
            g_optim = RMSprop(
                self.G.parameters(),
                lr=self.hparams.lr,
            )

        elif self.hparams.constraint_method == "gp":
            d_optim = Adam(
                self.D.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.b1, self.hparams.b2),
                weight_decay=self.hparams.weight_decay,
            )
            g_optim = Adam(
                self.G.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.b1, self.hparams.b2),
                weight_decay=self.hparams.weight_decay,
            )

        return [d_optim, g_optim], []
