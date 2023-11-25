from typing import List, Tuple

import torch
import torch.nn as nn

from models.modules.unet_layers import (
    AttentionDownBlock,
    AttentionUpBlock,
    ConvDownBlock,
    ConvUpBlock,
    SinusoidalPositionalEmbedding,
)


class DiffusionUNet(nn.Module):
    """
    Implements the U-Net architecture used in the Denoising Diffusion Probabilistic Models (DDPM) paper.
    This version includes the following modifications:
    1. Group normalization replaces weight normalization.
    2. Different resolutions are used based on the image size (e.g., four feature map resolutions for 32x32 images).
    3. Incorporates two convolutional residual blocks per resolution level and self-attention blocks at the 16x16 resolution.
    4. Adds Transformer sinusoidal position embedding for diffusion time 't' in each residual block.

    Args:
        img_size (int): Size of the input image, assumed to be square.
        in_channels (int): Number of channels in the input image.
    """

    def __init__(self, img_size: int = 32, in_channels: int = 3):
        """
        Initializes the DiffusionUNet model.

        Args:
            img_size (int, optional): Size of the input image, assumed to be square. Default is 32.
            in_channels (int, optional): Number of channels in the input image. Default is 3.
        """
        super().__init__()

        self.initial_conv = nn.Conv2d(
            in_channels, 128, kernel_size=3, stride=1, padding="same"
        )
        self.positional_encoding = self._build_positional_encoding_module()

        self.downsample_blocks = self._build_downsample_blocks()
        self.bottleneck = self._build_bottleneck_block()
        self.upsample_blocks = self._build_upsample_blocks()
        self.output_conv = self._build_output_conv_layer()

    def _build_positional_encoding_module(self) -> nn.Sequential:
        """Builds the positional encoding module."""
        return nn.Sequential(
            SinusoidalPositionalEmbedding(dimension=128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def _build_downsample_blocks(self) -> nn.ModuleList:
        """Builds downsample blocks for the U-Net architecture."""
        return nn.ModuleList(
            [
                ConvDownBlock(
                    128, 128, num_layers=2, num_groups=32, time_emb_channels=512
                ),
                ConvDownBlock(
                    128, 128, num_layers=2, num_groups=32, time_emb_channels=512
                ),
                ConvDownBlock(
                    128, 256, num_layers=2, num_groups=32, time_emb_channels=512
                ),
                AttentionDownBlock(
                    256,
                    256,
                    num_layers=2,
                    num_att_heads=4,
                    num_groups=32,
                    time_emb_channels=512,
                ),
                ConvDownBlock(
                    256, 512, num_layers=2, num_groups=32, time_emb_channels=512
                ),
            ]
        )

    def _build_bottleneck_block(self) -> AttentionDownBlock:
        """Builds the bottleneck block with attention mechanism."""
        return AttentionDownBlock(
            512,
            512,
            num_layers=2,
            num_att_heads=4,
            num_groups=32,
            time_emb_channels=512,
            downsample=False,
        )

    def _build_upsample_blocks(self) -> nn.ModuleList:
        """Builds upsample blocks for the U-Net architecture."""
        return nn.ModuleList(
            [
                ConvUpBlock(
                    1024, 512, num_layers=2, num_groups=32, time_emb_channels=512
                ),
                AttentionUpBlock(
                    768,
                    256,
                    num_layers=2,
                    num_att_heads=4,
                    num_groups=32,
                    time_emb_channels=512,
                ),
                ConvUpBlock(
                    512, 256, num_layers=2, num_groups=32, time_emb_channels=512
                ),
                ConvUpBlock(
                    384, 128, num_layers=2, num_groups=32, time_emb_channels=512
                ),
                ConvUpBlock(
                    256, 128, num_layers=2, num_groups=32, time_emb_channels=512
                ),
            ]
        )

    def _build_output_conv_layer(self) -> nn.Sequential:
        """Builds the final output convolution layer."""
        return nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
        )

    def forward(self, input_tensor: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the U-Net model.

        Args:
            input_tensor (torch.Tensor): The input image tensor.
            time (torch.Tensor): The diffusion time tensor.

        Returns:
            torch.Tensor: The output tensor of the model.
        """
        time_encoded = self.positional_encoding(time)
        initial_x = self.initial_conv(input_tensor)

        x, skip_connections = self._apply_downsample_blocks(initial_x, time_encoded)
        x = self.bottleneck(x, time_encoded)
        x = self._apply_upsample_blocks(x, skip_connections, time_encoded)

        return self._apply_output_conv_layer(x, initial_x)

    def _apply_downsample_blocks(
        self, x: torch.Tensor, time_encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Applies downsample blocks and records skip connections.

        Args:
            x (torch.Tensor): Input tensor.
            time_encoded (torch.Tensor): Time-encoded tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output tensor and a list of skip connection tensors.
        """
        skip_connections = [x]
        for block in self.downsample_blocks:
            x = block(x, time_encoded)
            skip_connections.append(x)
        return x, skip_connections[::-1]

    def _apply_upsample_blocks(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor],
        time_encoded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies upsample blocks using skip connections from the downsample path.

        Args:
            x (torch.Tensor): Input tensor.
            skip_connections (List[torch.Tensor]): List of skip connection tensors.
            time_encoded (torch.Tensor): Time-encoded tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for block, skip in zip(self.upsample_blocks, skip_connections):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded)
        return x

    def _apply_output_conv_layer(
        self, x: torch.Tensor, initial_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the final output convolution layer.

        Args:
            x (torch.Tensor): Input tensor.
            initial_x (torch.Tensor): Initial input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = torch.cat([x, initial_x], dim=1)
        return self.output_conv(x)
