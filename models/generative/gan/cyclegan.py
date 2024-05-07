from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn


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
            *self._residual_blocks(in_channels=256, nun_blocks=num_res_blocks),
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

    def _residual_blocks(self, in_channels, num_blocks):
        return [ResidualBlock(in_channels) for _ in range(num_blocks)]

    def _upsampling_blocks(self, in_channels, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
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
    https://junyanz.github.io/CycleGAN/
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lambda_cycle: float = 10.0,
        lr: float = 0.0002,
    ):
        super(CycleGAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Models
        self.G_AB = Generator(in_channels, out_channels)
        self.G_BA = Generator(out_channels, in_channels)
        self.D_A = Discriminator(in_channels)
        self.D_B = Discriminator(out_channels)

        # Hyperparameters
        self.lambda_cycle = lambda_cycle
        self.lr = lr

        # Loss criteria
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()

    def forward(self, real_A, real_B):
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        identity_A = self.G_AB(fake_A)
        identity_B = self.G_BA(fake_B)
        return fake_A, fake_B, identity_A, identity_B

    def adversarial_loss(
        self,
        preds: torch.Tensor,
        target_real: bool,
    ) -> torch.Tensor:
        target = (
            torch.ones_like(preds) if target_real
            else torch.zeros_like(preds)
        )
        return self.criterion_GAN(preds, target)

    def _generator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        identity: torch.Tensor,
        G: nn.Module,
        D: nn.Module,
        name: str,
    ) -> torch.Tensor:
        # Adversarial loss
        adv_loss = self.adversarial_loss(D(fake), True)

        # Cycle consistency loss
        cycle_loss = self.criterion_cycle(identity, real) * self.lambda_cycle

        loss_dict = {
            f"adv_loss_{name}": adv_loss,
            f"cycle_loss_{name}": cycle_loss,
            f"g_loss_{name}": adv_loss + cycle_loss,
        }
        return loss_dict

    def _discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        D: nn.Module,
        name: str,
    ) -> torch.Tensor:
        d_loss_real = self.adversarial_loss(D(real), True)
        d_loss_fake = self.adversarial_loss(D(fake.detach()), False)
        d_loss = (d_loss_real + d_loss_fake) / 2

        loss_dict = {
            f"d_loss_real_{name}": d_loss_real,
            f"d_loss_fake_{name}": d_loss_fake,
            f"d_loss_{name}": d_loss,
        }
        return loss_dict

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        real_A, real_B = batch
        fake_A, fake_B, identity_A, identity_B = self(real_A, real_B)

        # Generator AB
        loss_dict = self._generator_loss(
            real=real_A,
            fake=fake_A,
            identity=identity_A,
            D=D_A,
        )
        d_optim.zero_grad(set_to_none=True)
        self.manual_backward(loss_dict["G"])
        d_optim.step()

        # Generator BA
        if batch_idx % 4 == 1:
            loss_dict = self._generator_loss(
                real=real_B,
                fake=fake_B,
                identity=identity_B,
                D=D_B,
            )
            return loss_dict["G"]

        # Discriminator D_A
        if batch_idx % 4 == 2:
            loss_dict = self._discriminator_loss(
                real=real_A,
                fake=fake_A,
                D=D_A,
            )
            return loss_dict["D"]

        # Discriminator D_B
        if batch_idx % 4 == 3:
            loss_dict = self._discriminator_loss(
                real=real_B,
                fake=fake_B,
                D=D_B,
            )
            return loss_dict["D"]

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

    def configure_optimizers(self):
        optimizers = {
            "G": torch.optim.Adam(
                self.G_AB.parameters(),
                lr=self.lr,
                betas=(0.5, 0.999),
            ),
            "F": torch.optim.Adam(
                self.G_BA.parameters(),
                lr=self.lr,
                betas=(0.5, 0.999),
            ),
            "D_A": torch.optim.Adam(
                self.D_A.parameters(),
                lr=self.lr,
                betas=(0.5, 0.999),
            ),
            "D_B": torch.optim.Adam(
                self.D_B.parameters(),
                lr=self.lr,
                betas=(0.5, 0.999),
            ),
        }
        return optimizers.values(), []
