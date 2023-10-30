from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_res_blocks=9):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            self._initial_block(input_channels, 64),
            *self._downsampling_blocks(64, 2),
            *self._residual_blocks(256, num_res_blocks),
            *self._upsampling_blocks(256, 2),
            self._output_block(64, output_channels)
        )

    def _initial_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_channels, 7),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def _downsampling_blocks(self, in_channels, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Conv2d(in_channels, in_channels * 2, 3, stride=2, padding=1)
            )
            blocks.append(nn.InstanceNorm2d(in_channels * 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels *= 2
        return blocks

    def _residual_blocks(self, in_channels, num_blocks):
        return [ResidualBlock(in_channels) for _ in range(num_blocks)]

    def _upsampling_blocks(self, in_channels, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels // 2,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            blocks.append(nn.InstanceNorm2d(in_channels // 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels //= 2
        return blocks

    def _output_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(in_channels, out_channels, 7), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._discriminator_block(input_channels, 64, stride=2),
            self._discriminator_block(64, 128, stride=2),
            self._discriminator_block(128, 256, stride=2),
            self._discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def _discriminator_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lr: float = 0.0002,
    ):
        super(CycleGAN, self).__init__()

        # Models
        self.G = Generator(input_channels, output_channels)
        self.F = Generator(output_channels, input_channels)
        self.D_A = Discriminator(input_channels)
        self.D_B = Discriminator(output_channels)

        # Hyperparameters
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lr = lr

        # Loss criteria
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def forward(self, real_A, real_B):
        fake_B = self.G(real_A)
        fake_A = self.F(real_B)
        identity_A = self.F(real_A)
        identity_B = self.G(real_B)
        return fake_A, fake_B, identity_A, identity_B

    def adversarial_loss(self, preds: torch.Tensor, target_real: bool) -> torch.Tensor:
        target = torch.ones_like(preds) if target_real else torch.zeros_like(preds)
        return self.criterion_GAN(preds, target)

    def _generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ) -> torch.Tensor:
        fake = generator(real_A)
        identity = generator(generator(real_B))

        # Adversarial loss
        loss_GAN = self.adversarial_loss(discriminator(fake), True)

        # Cycle consistency loss
        loss_cycle = self.criterion_cycle(generator(fake), real_A) * self.lambda_cycle

        # Identity loss
        loss_identity = self.criterion_identity(identity, real_A) * self.lambda_identity

        return loss_GAN + loss_cycle + loss_identity

    def _discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor, discriminator: nn.Module
    ) -> torch.Tensor:
        loss_real = self.adversarial_loss(discriminator(real), True)
        loss_fake = self.adversarial_loss(discriminator(fake.detach()), False)
        return (loss_real + loss_fake) / 2

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        real_A, real_B = batch

        # Generator G
        if optimizer_idx == "opt_G":
            loss = self._generator_loss(real_A, real_B, self.G, self.D_B)
            self.log("train_loss_G", loss)
            return loss

        # Generator F
        if optimizer_idx == "opt_F":
            loss = self._generator_loss(real_B, real_A, self.F, self.D_A)
            self.log("train_loss_F", loss)
            return loss

        fake_A = self.F(real_B)
        fake_B = self.G(real_A)

        # Discriminator D_A
        if optimizer_idx == "opt_D_A":
            loss = self._discriminator_loss(real_A, fake_A, self.D_A)
            self.log("train_loss_D_A", loss)
            return loss

        # Discriminator D_B
        if optimizer_idx == "opt_D_B":
            loss = self._discriminator_loss(real_B, fake_B, self.D_B)
            self.log("train_loss_D_B", loss)
            return loss

    def configure_optimizers(self):
        optimizers = {
            "opt_G": torch.optim.Adam(
                self.G.parameters(), lr=self.lr, betas=(0.5, 0.999)
            ),
            "opt_F": torch.optim.Adam(
                self.F.parameters(), lr=self.lr, betas=(0.5, 0.999)
            ),
            "opt_D_A": torch.optim.Adam(
                self.D_A.parameters(), lr=self.lr, betas=(0.5, 0.999)
            ),
            "opt_D_B": torch.optim.Adam(
                self.D_B.parameters(), lr=self.lr, betas=(0.5, 0.999)
            ),
        }
        return optimizers.values(), []
