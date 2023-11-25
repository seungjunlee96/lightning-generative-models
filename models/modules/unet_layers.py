import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Implements sinusoidal positional embeddings as described in the "Attention Is All You Need" paper.
    These embeddings provide a way to encode sequential information for models that do not inherently process sequences.

    Args:
        dimension (int): Dimension of the embedding.
        max_timesteps (int): Maximum number of timesteps to encode.
    """

    def __init__(self, dimension: int, max_timesteps: int = 1000):
        """
        Initializes the SinusoidalPositionalEmbedding module.

        Args:
            dimension (int): Dimension of the embedding.
            max_timesteps (int, optional): Maximum number of timesteps to encode. Default is 1000.
        """
        super().__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"

        self.dimension = dimension
        self.pe_matrix = self._generate_positional_matrix(max_timesteps, dimension)

    @staticmethod
    def _generate_positional_matrix(max_timesteps: int, dimension: int) -> torch.Tensor:
        """Generates the positional embedding matrix."""
        even_indices = torch.arange(0, dimension, 2)
        log_term = torch.log(torch.tensor(10000.0)) / dimension
        div_term = torch.exp(even_indices * -log_term)

        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        pe_matrix = torch.zeros(max_timesteps, dimension)
        pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

        return pe_matrix

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Computes positional embeddings for the given timesteps.

        Args:
            timestep (torch.Tensor): Tensor representing timesteps.

        Returns:
            torch.Tensor: The positional embeddings for the input timesteps.
        """
        return self.pe_matrix.to(timestep.device)[timestep]


class ConvBlock(nn.Module):
    """
    Basic convolutional block comprising a convolution layer, group normalization, and SiLU activation.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        groups (int): Number of groups for group normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        """
        Initializes the ConvBlock module.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            groups (int, optional): Number of groups for group normalization. Default is 32.
        """
        super().__init__()
        groups = min(groups, out_channels) if out_channels % groups != 0 else groups
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, padding: int):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv(input_tensor)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float = 2.0):
        super(UpsampleBlock, self).__init__()

        self.scale = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # align_corners=True for potential convertibility to ONNX
        x = F.interpolate(
            input_tensor, scale_factor=self.scale, mode="bilinear", align_corners=True
        )
        x = self.conv(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_emb_channels: int,
        num_groups: int,
        downsample: bool = True,
    ):
        """
        Convolutional Down Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_layers (int): Number of ResNet layers.
            time_emb_channels (int): Number of time embedding channels.
            num_groups (int): Number of groups for group normalization.
            downsample (bool): Whether to downsample the output.

        """
        super(ConvDownBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_channels=time_emb_channels,
                num_groups=num_groups,
            )
            resnet_blocks.append(resnet_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        self.downsample = (
            DownsampleBlock(
                in_channels=out_channels, out_channels=out_channels, stride=2, padding=1
            )
            if downsample
            else None
        )

    def forward(
        self, input_tensor: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.downsample:
            x = self.downsample(x)
        return x


class ConvUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_emb_channels: int,
        num_groups: int,
        upsample: bool = True,
    ):
        """
        Convolutional Up Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_layers (int): Number of ResNet layers.
            time_emb_channels (int): Number of time embedding channels.
            num_groups (int): Number of groups for group normalization.
            upsample (bool): Whether to upsample the output.

        """
        super(ConvUpBlock, self).__init__()
        resnet_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_channels=time_emb_channels,
                num_groups=num_groups,
            )
            resnet_blocks.append(resnet_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        self.upsample = (
            UpsampleBlock(in_channels=out_channels, out_channels=out_channels)
            if upsample
            else None
        )

    def forward(
        self, input_tensor: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        x = input_tensor
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x, time_embedding)
        if self.upsample:
            x = self.upsample(x)
        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-attention blocks are applied at the 16x16 resolution in the original DDPM paper.
    Implementation is based on "Attention Is All You Need" paper, Vaswani et al., 2015
    (https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
        self,
        num_heads: int,
        in_channels: int,
        num_groups: int = 32,
        embedding_dim: int = 256,
    ):
        """
        Self-Attention Block.

        Args:
            num_heads (int): Number of attention heads.
            in_channels (int): Number of input channels.
            num_groups (int): Number of groups for group normalization.
            embedding_dim (int): Dimension of the embedding.

        """
        super(SelfAttentionBlock, self).__init__()
        # For each of heads use d_k = d_v = d_model / num_heads
        self.num_heads = num_heads
        self.d_model = embedding_dim
        self.d_keys = embedding_dim // num_heads
        self.d_values = embedding_dim // num_heads

        self.query_projection = nn.Linear(in_channels, embedding_dim)
        self.key_projection = nn.Linear(in_channels, embedding_dim)
        self.value_projection = nn.Linear(in_channels, embedding_dim)

        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.GroupNorm(num_channels=embedding_dim, num_groups=num_groups)

    def split_features_for_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        # We receive Q, K and V at shape [batch, h*w, embedding_dim].
        # This method splits embedding_dim into 'num_heads' features so that
        # each channel becomes of size embedding_dim / num_heads.
        # Output shape becomes [batch, num_heads, h*w, embedding_dim/num_heads],
        # where 'embedding_dim/num_heads' is equal to d_k = d_k = d_v = sizes for
        # K, Q and V respectively, according to paper.
        batch, hw, emb_dim = tensor.shape
        channels_per_head = emb_dim // self.num_heads
        heads_splitted_tensor = torch.split(
            tensor, split_size_or_sections=channels_per_head, dim=-1
        )
        heads_splitted_tensor = torch.stack(heads_splitted_tensor, 1)
        return heads_splitted_tensor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor
        batch, features, h, w = x.shape
        # Do reshape and transpose input tensor since we want to process depth feature maps, not spatial maps
        x = x.view(batch, features, h * w).transpose(1, 2)

        # Get linear projections of K, Q and V according to Fig. 2 in the original Transformer paper
        queries = self.query_projection(x)  # [b, in_channels, embedding_dim]
        keys = self.key_projection(x)  # [b, in_channels, embedding_dim]
        values = self.value_projection(x)  # [b, in_channels, embedding_dim]

        # Split Q, K, V between attention heads to process them simultaneously
        queries = self.split_features_for_heads(queries)
        keys = self.split_features_for_heads(keys)
        values = self.split_features_for_heads(values)

        # Perform Scaled Dot-Product Attention (eq. 1 in the Transformer paper).
        # Each SDPA block yields tensor of size d_v = embedding_dim/num_heads.
        scale = self.d_keys**-0.5
        attention_scores = torch.softmax(
            torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1
        )
        attention_scores = torch.matmul(attention_scores, values)

        # Permute computed attention scores such that
        # [batch, num_heads, h*w, embedding_dim] --> [batch, h*w, num_heads, d_v]
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

        # Concatenate scores per head into one tensor so that
        # [batch, h*w, num_heads, d_v] --> [batch, h*w, num_heads*d_v]
        concatenated_heads_attention_scores = attention_scores.view(
            batch, h * w, self.d_model
        )

        # Perform linear projection and reshape tensor such that
        # [batch, h*w, d_model] --> [batch, d_model, h*w] -> [batch, d_model, h, w]
        linear_projection = self.final_projection(concatenated_heads_attention_scores)
        linear_projection = linear_projection.transpose(-1, -2).reshape(
            batch, self.d_model, h, w
        )

        # Residual connection + norm
        x = self.norm(linear_projection + input_tensor)
        return x


class AttentionDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_emb_channels: int,
        num_groups: int,
        num_att_heads: int,
        downsample: bool = True,
    ):
        """
        Attention Down Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_layers (int): Number of ResNet layers.
            time_emb_channels (int): Number of time embedding channels.
            num_groups (int): Number of groups for group normalization.
            num_att_heads (int): Number of attention heads.
            downsample (bool): Whether to downsample the output.

        """
        super(AttentionDownBlock, self).__init__()

        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_channels=time_emb_channels,
                num_groups=num_groups,
            )
            attention_block = SelfAttentionBlock(
                in_channels=out_channels,
                embedding_dim=out_channels,
                num_heads=num_att_heads,
                num_groups=num_groups,
            )

            resnet_blocks.append(resnet_block)
            attention_blocks.append(attention_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.downsample = (
            DownsampleBlock(
                in_channels=out_channels, out_channels=out_channels, stride=2, padding=1
            )
            if downsample
            else None
        )

    def forward(
        self, input_tensor: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        x = input_tensor
        for resnet_block, attention_block in zip(
            self.resnet_blocks, self.attention_blocks
        ):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class AttentionUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_emb_channels: int,
        num_groups: int,
        num_att_heads: int,
        upsample: bool = True,
    ):
        """
        Attention Up Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_layers (int): Number of ResNet layers.
            time_emb_channels (int): Number of time embedding channels.
            num_groups (int): Number of groups for group normalization.
            num_att_heads (int): Number of attention heads.
            upsample (bool): Whether to upsample the output.

        """
        super(AttentionUpBlock, self).__init__()

        resnet_blocks = []
        attention_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_channels=time_emb_channels,
                num_groups=num_groups,
            )
            attention_block = SelfAttentionBlock(
                in_channels=out_channels,
                embedding_dim=out_channels,
                num_heads=num_att_heads,
                num_groups=num_groups,
            )

            resnet_blocks.append(resnet_block)
            attention_blocks.append(attention_block)

        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.upsample = (
            UpsampleBlock(in_channels=out_channels, out_channels=out_channels)
            if upsample
            else None
        )

    def forward(
        self, input_tensor: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        x = input_tensor
        for resnet_block, attention_block in zip(
            self.resnet_blocks, self.attention_blocks
        ):
            x = resnet_block(x, time_embedding)
            x = attention_block(x)
        if self.upsample:
            x = self.upsample(x)
        return x


class ResNetBlock(nn.Module):
    """
    In the original DDPM paper Wide ResNet was used
    (https://arxiv.org/pdf/1605.07146.pdf).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_channels: int = None,
        num_groups: int = 8,
    ):
        """
        ResNet Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_emb_channels (int): Number of time embedding channels.
            num_groups (int): Number of groups for group normalization.

        """
        super(ResNetBlock, self).__init__()
        self.time_embedding_projectile = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_channels, out_channels))
            if time_emb_channels
            else None
        )

        self.block1 = ConvBlock(in_channels, out_channels, groups=num_groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=num_groups)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        input_tensor = x
        h = self.block1(x)
        # According to authors implementations, they inject timestep embedding into the network
        # using MLP after the first conv block in all the ResNet blocks
        time_emb = self.time_embedding_projectile(time_embedding)
        time_emb = time_emb[:, :, None, None]
        x = time_emb + h

        x = self.block2(x)
        return x + self.residual_conv(input_tensor)
