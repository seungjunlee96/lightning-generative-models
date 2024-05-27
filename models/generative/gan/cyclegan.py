from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 9,
    ):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            self._initial_block(in_channels=in_channels, out_channels=64),
            *self._downsampling_blocks(in_channels=64, num_blocks=2),
            *self._residual_blocks(in_channels=256, num_blocks=num_res_blocks),
            *self._upsampling_blocks(in_channels=256, num_blocks=2),
            self._output_block(in_channels=64, out_channels=out_channels)
        )

    def _initial_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def _downsampling_blocks(
        self,
        in_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            blocks.append(nn.InstanceNorm2d(in_channels * 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels *= 2
        return blocks

    def _residual_blocks(self, in_channels: int, num_blocks: int):
        return [ResidualBlock(in_channels) for _ in range(num_blocks)]

    def _upsampling_blocks(
        self,
        in_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels // 2,
                    kernel_size,
                    stride,
                    padding,
                    output_padding=1,
                )
            )
            blocks.append(nn.InstanceNorm2d(in_channels // 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels //= 2
        return blocks

    def _output_block(
        self,
        in_channels: int,
        out_channels: int,
    ):
        return nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._discriminator_block(in_channels, 64, stride=2),
            self._discriminator_block(64, 128, stride=2),
            self._discriminator_block(128, 256, stride=2),
            self._discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def _discriminator_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1,
    ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class CycleGAN(pl.LightningModule):
    """
    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
    - Official code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lambda_identity: float = 0.5,
        lambda_cycle: float = 10.0,
        lr: float = 0.0002,
    ):
        super(CycleGAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.G_AB = Generator(in_channels, out_channels)
        self.G_BA = Generator(out_channels, in_channels)
        self.D_A = Discriminator(in_channels)
        self.D_B = Discriminator(out_channels)

    def forward(self, real_A, real_B):
        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)
        identity_A = self.G_BA(fake_B)
        identity_B = self.G_AB(fake_A)
        return fake_A, fake_B, identity_A, identity_B

    def _calculate_g_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        fake_B: torch.Tensor,
        identity_A: torch.Tensor,
        identity_B: torch.Tensor,
    ) -> torch.Tensor:
        # Adversarial loss
        logits_fake_A = self.D_A(fake_A)
        logits_fake_B = self.D_B(fake_B)

        adv_loss = (
            bce_with_logits(logits_fake_A, torch.ones_like(logits_fake_A))
            + bce_with_logits(logits_fake_B, torch.ones_like(logits_fake_B))
        )

        # Identity loss
        identity_loss = (
            F.l1_loss(fake_B, real_A)
            + F.l1_loss(fake_A, real_B)
        )

        # Cycle consistency loss
        cycle_loss = (
            F.l1_loss(identity_A, real_A)
            + F.l1_loss(identity_B, real_B)
        )

        # Generator loss
        g_loss = (
            adv_loss
            + identity_loss * self.hparams.lambda_identity
            + cycle_loss * self.hparams.lambda_cycle
        )

        loss_dict = {
            "adv_loss": adv_loss,
            "identity_loss": identity_loss,
            "cycle_loss": cycle_loss,
            "g_loss": g_loss,
        }

        return loss_dict

    def _calculate_d_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        fake_B: torch.Tensor,
    ) -> torch.Tensor:
        # Discriminator A
        logits_real_A = self.D_A(real_A)
        d_loss_real_A = bce_with_logits(logits_real_A, torch.ones_like(logits_real_A))

        logits_fake_A = self.D_A(fake_A.detach())
        d_loss_fake_A = bce_with_logits(logits_fake_A, torch.zeros_like(logits_fake_A))

        d_loss_A = (d_loss_real_A + d_loss_fake_A) / 2

        # Discriminator B
        logits_real_B = self.D_B(real_B)
        d_loss_real_B = bce_with_logits(logits_real_B, torch.ones_like(logits_real_B))

        logits_fake_B = self.D_B(fake_B.detach())
        d_loss_fake_B = bce_with_logits(logits_fake_B, torch.zeros_like(logits_fake_B))

        d_loss_B = (d_loss_real_B + d_loss_fake_B) / 2

        # Discriminator loss
        d_loss = d_loss_A + d_loss_B

        loss_dict = {
            "d_loss_real_A": d_loss_real_A,
            "d_loss_fake_A": d_loss_fake_A,
            "d_loss_A": d_loss_A,
            "d_loss_real_B": d_loss_real_B,
            "d_loss_fake_B": d_loss_fake_B,
            "d_loss_B": d_loss_B,
            "d_loss": d_loss,
        }

        return loss_dict

    def _common_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        real_A, real_B = batch
        fake_A, fake_B, identity_A, identity_B = self(real_A, real_B)
        d_optim, g_optim = self.optimizers()

        # Train Discriminator
        loss_dict = self._calculate_d_loss(
            real_A=real_A,
            real_B=real_B,
            fake_A=fake_A,
            fake_B=fake_B,
        )
        if self.training:
            d_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["d_loss"])
            d_optim.step()

        # Train Generator
        loss_dict.update(
            self._calculate_g_loss(
                real_A=real_A,
                real_B=real_B,
                fake_A=fake_A,
                fake_B=fake_B,
                identity_A=identity_A,
                identity_B=identity_B,
            )
        )
        if self.training:
            g_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["g_loss"])
            g_optim.step()

        loss_dict = {f"{mode}_{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        return loss_dict

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self._common_step(batch, "val")

    def configure_optimizers(self):
        d_optim = Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
        )
        g_optim = Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
        )
        return (d_optim, g_optim), []
