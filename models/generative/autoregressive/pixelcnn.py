from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        # Create a mask tensor of the same shape as the convolutional weights
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # Mask out future pixel values for each pixel.
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the mask to the weights before the convolution operation
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class GatedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GatedBlock, self).__init__()
        # Masked convolution to ensure no information flow from future pixels
        self.conv = MaskedConv2d("B", in_channels, 2 * out_channels, 7, 1, 3)
        # Skip connection if the number of channels changes
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.skip:
            x = self.skip(x)
        # Split the output into two parts for the gated activation
        tanh, sigmoid = torch.chunk(out, 2, dim=1)
        # Gated activation
        return x + torch.tanh(tanh) * torch.sigmoid(sigmoid)


class PixelCNN(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        num_layers: int = 5,
        learning_rate: float = 0.001,
    ):
        super(PixelCNN, self).__init__()
        self.learning_rate = learning_rate

        # Initial masked convolution ('A' type)
        self.input_conv = MaskedConv2d("A", input_channels, hidden_channels, 7, 1, 3)

        # Sequence of gated blocks with masked convolutions ('B' type)
        self.gated_blocks = nn.ModuleList(
            [GatedBlock(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        # Final convolution to produce the output
        self.output_conv = nn.Conv2d(hidden_channels, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for block in self.gated_blocks:
            x = block(x)
        return self.output_conv(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        # Compute the cross-entropy loss between predictions and targets
        loss = nn.CrossEntropyLoss()(output, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def sample(self, num_samples: int) -> torch.Tensor:
        samples = torch.zeros(
            num_samples, self.input_channels, self.img_size, self.img_size
        ).to(self.device)
        with torch.no_grad():
            for i in range(self.img_size):
                for j in range(self.img_size):
                    out = self(samples)
                    # Sample a pixel value based on the predicted probabilities
                    probs = F.softmax(out[:, :, i, j], dim=-1)
                    samples[:, :, i, j] = torch.multinomial(probs, 1).float()
        return samples
