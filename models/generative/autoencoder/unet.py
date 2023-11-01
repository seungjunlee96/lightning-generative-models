from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim


class DoubleConvolution(nn.Module):
    """
    Double 3x3 convolution with ReLU activation. Optionally includes a MaxPooling layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_maxpool: bool = False,
    ) -> None:
        super(DoubleConvolution, self).__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2) if use_maxpool else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Deconvolution(nn.Module):
    """
    Deconvolution followed by a double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Deconvolution, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.double_conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x: Tensor, skip_connection: Tensor) -> Tensor:
        x = self.deconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.double_conv(x)


class Encoder(nn.Module):
    """
    Encoder part of the U-Net architecture.
    """

    def __init__(self, in_channels: int, features_list: List[int]) -> None:
        super(Encoder, self).__init__()

        layers = [DoubleConvolution(in_channels, features_list[0], use_maxpool=False)]
        for in_features, out_features in zip(features_list[:-1], features_list[1:]):
            layers += [DoubleConvolution(in_features, out_features, use_maxpool=True)]
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    """
    Decoder part of the U-Net architecture.
    """

    def __init__(self, features_list: List[int], out_channels: int) -> None:
        super(Decoder, self).__init__()

        features_list = list(reversed(features_list))
        self.blocks = nn.ModuleList(
            [
                Deconvolution(in_features, out_features)
                for in_features, out_features in zip(
                    features_list[:-1], features_list[1:]
                )
            ]
        )
        self.head = nn.Conv2d(features_list[0], out_channels, kernel_size=1)

    def forward(self, encoder_features: List[Tensor]) -> Tensor:
        # Extract the last feature tensor for initial processing in the decoder
        x = encoder_features.pop()

        # Use the remaining encoder features for skip connections in reverse order
        for block, skip_connection in zip(self.blocks, encoder_features[::-1]):
            x = block(x, skip_connection)

        return self.head(x)


class UNet(pl.LightningModule):
    """
    U-Net architecture using PyTorch Lightning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features_list: List[int],
        lr: float = 1e-3,
    ) -> None:
        super(UNet, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(in_channels, features_list)
        self.decoder = Decoder(features_list, out_channels)
        self.loss = nn.MSELoss()  # Adjust this as needed for your task

    def forward(self, x: Tensor) -> Tensor:
        encoder_features = self.encoder(x)
        return self.decoder(encoder_features)

    def _common_step(self, batch: Tuple[Tensor, Tensor], split: str):
        assert split in ["train", "val"]
        x, _ = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)

        self.log(
            f"{split}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        return self._common_step(batch, "val")

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer
