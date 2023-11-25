from typing import List

import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule
from torchinfo import summary

from models.generative.autoencoder.diffusion_unet import DiffusionUNet
from models.scheduler.ddpm import DDPMScheduler
from utils.visualization import make_grid


class DDPM(LightningModule):
    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 32,
        diffusion_timesteps: int = 1000,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
    ):
        """
        Denoising Diffusion Probabilistic Model (DDPM) for image generation.

        Args:
            img_channels (int): Number of channels in the input images.
            img_size (int): Size (height/width) of the input images.
            diffusion_timesteps (int): Number of timesteps in the diffusion process.
            lr (float): Learning rate for the optimizer.
            beta_start (float): Starting beta value for the diffusion process.
            beta_end (float): Ending beta value for the diffusion process.
            num_epochs (int): Number of training epochs.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffusionUNet(img_size=img_size, in_channels=img_channels)
        self.diffusion_scheduler = DDPMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            num_timesteps=diffusion_timesteps,
        )

        self.summary()
        self.fixed_noise = torch.randn([16, img_channels, img_size, img_size])

    def forward(self, noisy_images, timesteps):
        return self.model(noisy_images, timesteps)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (Tensor): Input batch of images.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: The training loss.
        """
        x, _ = batch
        batch_size = x.shape[0]

        timesteps = torch.randint(
            0, self.hparams.diffusion_timesteps, (batch_size,), device=self.device
        ).long()
        noisy_images, noise = self.diffusion_scheduler.forward_diffusion(x, timesteps)
        noise_pred = self(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        x, _ = batch
        batch_size = x.shape[0]

        timesteps = torch.randint(
            0, self.hparams.diffusion_timesteps, (batch_size,), device=self.device
        ).long()

        noisy_images, noise = self.diffusion_scheduler.forward_diffusion(x, timesteps)
        noise_pred = self(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss, prog_bar=True)

        # Log sampled images if it's the first batch of the epoch
        if batch_idx == 0:
            # Sample a few images for visualization
            sample_images = self.diffusion_scheduler.sampling(
                self.model,
                self.fixed_noise.to(self.device),
                save_all_steps=False,
            )
            sample_images = ((sample_images + 1.0) / 2.0) * 255.0
            sample_images = sample_images.clamp(0, 255).byte().detach().cpu()
            fig = make_grid(sample_images)
            self.logger.experiment.log(
                {"Random Generation": [wandb.Image(fig)]}, step=self.global_step
            )

        return loss

    def configure_optimizers(self):
        """
        Configures the model's optimizers and learning rate schedulers.

        Returns:
            list: List containing the optimizer and the learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=100,
                eta_min=1e-9,
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "learning_rate",
        }
        return [optimizer], [lr_scheduler]

    def summary(
        self,
        col_names: List[str] = [
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds",
            "trainable",
        ],
    ):
        x = torch.randn(
            [
                1,
                self.hparams.img_channels,
                self.hparams.img_size,
                self.hparams.img_size,
            ],
            device=self.device,
        )

        timesteps = torch.randint(
            0,
            self.hparams.diffusion_timesteps,
            (1,),
            device=self.device,
        ).long()

        summary(
            self.model,
            input_data=(x, timesteps),
            col_names=col_names,
        )
