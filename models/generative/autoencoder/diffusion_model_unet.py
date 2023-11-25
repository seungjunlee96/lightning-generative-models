# Code from https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/unet.py
import torch
import torch.nn as nn

from models.modules.unet_layers import (
    AttentionDownBlock,
    AttentionUpBlock,
    ConvDownBlock,
    ConvUpBlock,
    TransformerPositionalEmbedding,
)


class UNet(nn.Module):
    """
    Model architecture as described in the DDPM paper, Appendix, section B
    """

    def __init__(self, img_size=32, in_channels=3):
        super().__init__()
        # 1. We replaced weight normalization with group normalization
        # 2. Our 32x32 models use four feature map resolutions (32x32 to 4x4), and our 256x256 models use six
        # 3. Two convolutional residual blocks per resolution level and self-attention blocks at the 16x16 resolution
        # between the convolutional blocks [https://arxiv.org/pdf/1712.09763.pdf]
        # 4. Diffusion time t is specified by adding the Transformer sinusoidal position embedding into
        # each residual block [https://arxiv.org/pdf/1706.03762.pdf]

        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4),
        )

        self.downsample_blocks = nn.ModuleList(
            [
                ConvDownBlock(
                    in_channels=128,
                    out_channels=128,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                ConvDownBlock(
                    in_channels=128,
                    out_channels=128,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                ConvDownBlock(
                    in_channels=128,
                    out_channels=256,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                AttentionDownBlock(
                    in_channels=256,
                    out_channels=256,
                    num_layers=2,
                    num_att_heads=4,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                ConvDownBlock(
                    in_channels=256,
                    out_channels=512,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
            ]
        )

        self.bottleneck = AttentionDownBlock(
            in_channels=512,
            out_channels=512,
            num_layers=2,
            num_att_heads=4,
            num_groups=32,
            time_emb_channels=128 * 4,
            downsample=False,
        )  # 16x16x256 -> 16x16x256

        self.upsample_blocks = nn.ModuleList(
            [
                ConvUpBlock(
                    in_channels=512 + 512,
                    out_channels=512,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                AttentionUpBlock(
                    in_channels=512 + 256,
                    out_channels=256,
                    num_layers=2,
                    num_att_heads=4,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                ConvUpBlock(
                    in_channels=256 + 256,
                    out_channels=256,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                ConvUpBlock(
                    in_channels=256 + 128,
                    out_channels=128,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
                ConvUpBlock(
                    in_channels=128 + 128,
                    out_channels=128,
                    num_layers=2,
                    num_groups=32,
                    time_emb_channels=128 * 4,
                ),
            ]
        )

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=256, num_groups=32),
            nn.SiLU(),
            nn.Conv2d(256, 3, 3, padding=1),
        )

    def forward(self, input_tensor, time):
        time_encoded = self.positional_encoding(time)

        initial_x = self.initial_conv(input_tensor)

        # Downsample path
        skip_connections = [initial_x]
        x = initial_x
        for block in self.downsample_blocks:
            x = block(x, time_encoded)
            skip_connections.append(x)
        skip_connections = skip_connections[
            ::-1
        ]  # Reverse for correct skip connections

        # Bottleneck
        x = self.bottleneck(x, time_encoded)

        # Upsample path
        for block, skip_connection in zip(self.upsample_blocks, skip_connections):
            x = torch.cat([x, skip_connection], dim=1)
            x = block(x, time_encoded)

        # Final output convolution
        # Ensure the tensor has the right number of channels before final convolution
        x = torch.cat([x, initial_x], dim=1)
        out = self.output_conv(x)

        return out
