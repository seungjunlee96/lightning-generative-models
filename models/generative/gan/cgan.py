from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1.0)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class Generator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 7 * 7 * 256)
        self.deconv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view(x.size(0), 256, 7, 7)
        x = F.leaky_relu(self.deconv1(x), 0.2)
        x = torch.sigmoid(self.deconv2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channels + num_classes, 64, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        c = c.view(-1, c.size(1), 1, 1)
        c = c.expand(x.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConditionalGAN(pl.LightningModule):
    def __init__(
        self, num_channels: int = 1, num_classes: int = 10, latent_dim: int = 128
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.generator = Generator(latent_dim=latent_dim, num_classes=num_classes)
        self.discriminator = Discriminator(
            num_channels=num_channels, num_classes=num_classes
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        return self.generator(z, c)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, c = batch
        c = F.one_hot(c, num_classes=self.num_classes).float()
        d_optimizer, g_optimizer = self.optimizers()

        # Train Discriminator
        if self.global_step % 2 == 0:
            loss_dict = self._calculate_d_loss(x, c)
            d_optimizer.zero_grad()
            self.manual_backward(loss_dict["d_loss"])
            d_optimizer.step()

        # Train Generator
        else:
            loss_dict = self._calculate_g_loss(x, c)
            g_optimizer.zero_grad()
            self.manual_backward(loss_dict["g_loss"])
            g_optimizer.step()

        self.log_dict(loss_dict, on_step=True, prog_bar=True)
        return loss_dict

    def configure_optimizers(self):
        d_optim = Adam(self.discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
        g_optim = Adam(self.generator.parameters(), lr=0.0003, betas=(0.5, 0.999))

        return [d_optim, g_optim]

    def _calculate_d_loss(self, x: Tensor, c: Tensor) -> Tensor:
        logits_real = self.discriminator(x, c)
        d_loss_real = bce_with_logits(logits_real, torch.ones_like(logits_real))

        # Generate fake images
        z = torch.randn(x.size(0), self.latent_dim).to(self.device)
        x_hat = self.generator(z, c)
        logits_fake = self.discriminator(x_hat.detach(), c)
        d_loss_fake = bce_with_logits(logits_fake, torch.zeros_like(logits_fake))

        d_loss = d_loss_real + d_loss_fake

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "logits_real": logits_real.mean(),
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict

    def _calculate_g_loss(self, x: Tensor, c: Tensor) -> Tensor:
        # Generate fake images
        z = torch.randn(x.size(0), self.latent_dim).to(self.device)
        x_hat = self.generator(z, c)

        logits_fake = self.discriminator(x_hat, c)
        g_loss = bce_with_logits(logits_fake, torch.ones_like(logits_fake))

        loss_dict = {
            "g_loss": g_loss,
            "logits_fake": logits_fake.mean(),
        }
        return loss_dict
