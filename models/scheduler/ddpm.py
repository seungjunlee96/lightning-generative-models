import torch
from tqdm import tqdm


class DDPMScheduler:
    """
    Implements a Denoising Diffusion Probabilistic Model (DDPM) pipeline as described in
    the paper 'Denoising Diffusion Probabilistic Models' (https://arxiv.org/pdf/2006.11239.pdf).

    This class provides the functionality for both forward diffusion (adding noise to images)
    and reverse diffusion (generating images from noise), utilizing a learned model.

    Attributes:
        betas (torch.Tensor): Beta values for each timestep, controlling noise addition.
        alphas (torch.Tensor): Alpha values derived from betas, representing noise scale.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas, used in diffusion calculations.
        num_timesteps (int): Number of timesteps used in the diffusion process.

    Args:
        beta_start (float): Starting value of beta for the linear schedule.
        beta_end (float): Ending value of beta for the linear schedule.
        num_timesteps (int): Total number of timesteps for the diffusion process.
    """

    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
        num_timesteps: float = 1000,
    ):
        super().__init__()

        # Initialize betas and alphas
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_timesteps = num_timesteps

    def forward_diffusion(self, images, timesteps):
        """
        Performs the forward diffusion process on a batch of images.

        Args:
            images (torch.Tensor): A batch of images to be diffused.
            timesteps (torch.Tensor): Timesteps at which the diffusion process is evaluated.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the diffused images and the
            applied Gaussian noise.
        """
        device = images.device
        gaussian_noise = torch.randn_like(images)
        alpha_hat = self.alphas_cumprod.to(device)[timesteps]

        # Ensure alpha_hat is correctly broadcasted to match the dimensions of images
        while len(alpha_hat.shape) < len(images.shape):
            alpha_hat = alpha_hat.unsqueeze(-1)

        sqrt_alpha_hat = torch.sqrt(alpha_hat)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

        diffused_images = (
            sqrt_alpha_hat * images + sqrt_one_minus_alpha_hat * gaussian_noise
        )

        return diffused_images, gaussian_noise

    @torch.no_grad()
    def sampling(self, model, initial_noise, save_all_steps=False):
        """
        Generates images by sampling from the model using the reverse diffusion process.

        Args:
            model (torch.nn.Module): The neural network model to be used for image generation.
            initial_noise (torch.Tensor): The initial noise tensor to start the reverse diffusion.
            save_all_steps (bool): If True, saves images from all timesteps. Otherwise, only
                                   the final image is returned.

        Returns:
            list[torch.Tensor] or torch.Tensor: A list of images from each timestep if
            `save_all_steps` is True, otherwise a single image tensor.
        """
        device = initial_noise.device
        image = initial_noise.to(device)
        images = []

        for timestep in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
            ts = torch.full((image.size(0),), timestep, dtype=torch.long, device=device)
            predicted_noise = model(image, ts)

            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_cumprod[timestep].to(device)
            beta_t = self.betas[timestep].to(device)
            alpha_hat_prev = (
                self.alphas_cumprod[timestep - 1].to(device)
                if timestep > 0
                else torch.tensor(1.0).to(device)
            )

            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance = (
                torch.sqrt(beta_t_hat) * torch.randn_like(image) if timestep > 0 else 0
            )

            image = (
                torch.pow(alpha_t, -0.5)
                * (image - beta_t / torch.sqrt(1 - alpha_hat_prev) * predicted_noise)
                + variance
            )

            if save_all_steps:
                images.append(image.cpu())

        return images if save_all_steps else image
