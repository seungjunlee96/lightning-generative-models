from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from models.generative.gan.dcgan import DCGAN, initialize_weights


class Discriminator(nn.Module):
    def __init__(
        self,
        img_channels: int,
    ) -> None:
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(img_channels, 64, 4, 2, 1, use_bn=False),
            self._block(64, 128, 4, 2, 1, use_bn=False),
            self._block(128, 256, 4, 2, 1, use_bn=False),
            self._block(256, 512, 4, 2, 1, use_bn=False),
            self._block(512, 1, 4, 1, 0, final_layer=True),
        )
        self.model = initialize_weights(self.model)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_bn: bool = False,
        final_layer: bool = False,
    ) -> nn.Sequential:
        """
        Returns a block for the discriminator containing
        - a strided convolution with spectral normalization,
        - and a LeakyReLU activation for all layers except the output layer.
        """
        layers = [
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=True,
                )
            )
        ]

        if not final_layer:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).view(x.size(0), -1)


class SNGAN(DCGAN):
    """
    Spectral Normalization Generative Adversarial Network (SNGAN).

    This class implements an SNGAN, which applies spectral normalization to the
    discriminator to stabilize the training of the GAN.
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        latent_dim: int = 100,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
        ckpt_path: str = "",
    ) -> None:
        super(SNGAN, self).__init__(
            img_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
            lr=lr,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            ckpt_path=ckpt_path,
        )
        self.discriminator = Discriminator(img_channels=img_channels)

    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        d_loss_real = self.discriminator(x).mean()
        d_loss_fake = self.discriminator(x_hat.detach()).mean()

        d_loss = d_loss_fake - d_loss_real

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
        }

        return loss_dict

    def _calculate_g_loss(self, x_hat: Tensor) -> Tensor:
        g_loss = -self.discriminator(x_hat).mean()

        loss_dict = {"g_loss": g_loss}
        return loss_dict
