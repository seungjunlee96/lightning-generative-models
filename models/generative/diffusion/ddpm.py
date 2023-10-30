import copy
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam


def cosine_beta_schedule(timesteps):
    ts = np.linspace(0, 1, timesteps)
    betas = 0.5 * (1 + np.cos(np.pi * ts))
    return betas


# 3. Extract utility
def extract(tensor, t, shape):
    return tensor[t].expand(shape)


# 4. Default utility
def default(x, default_val):
    return x if x is not None else default_val()


# 5. A simple Unet (just a placeholder; a proper Unet implementation is needed)
class Unet(nn.Module):
    def __init__(self, dim, channels):
        super().__init__()
        self.dim = dim
        self.channels = channels
        # This is just a placeholder, you need a proper Unet implementation here
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x, t):
        return self.network(x)


class EMA:
    def __init__(self, decay):
        self.decay = decay

    def update_model_average(self, averaged_model, model):
        # Update parameters
        params1 = averaged_model.named_parameters()
        params2 = model.named_parameters()
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(
                    self.decay * param1.data
                    + (1 - self.decay) * dict_params2[name1].data
                )

        # Update buffers
        buffers1 = averaged_model.named_buffers()
        buffers2 = model.named_buffers()
        dict_buffers2 = dict(buffers2)
        for name1, buf1 in buffers1:
            if name1 in dict_buffers2:
                dict_buffers2[name1].data.copy_(buf1.data)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=3,
        timesteps=1000,
        loss_type="l1",
        betas=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        """Sample from diffusion distribution q(x_t|x_{t-1})"""
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        mean, _, _ = self.q_mean_variance(x_start, t)
        return (
            mean + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)
        if self.loss_type == "l1":
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, x, *args, **kwargs):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *x.shape,
            x.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)


class DDPM(pl.LightningModule):
    def __init__(
        self,
        image_size,
        channels=3,
        timesteps=1000,
        loss_type="l1",
        betas=None,
        train_lr=2e-5,
        ema_decay=0.995,
    ):
        super(DDPM, self).__init__()

        # Define the DDPM model
        self.model = GaussianDiffusion(
            denoise_fn=Unet(dim=image_size, channels=channels),
            image_size=image_size,
            channels=channels,
            timesteps=timesteps,
            loss_type=loss_type,
            betas=betas,
        )
        self.gradient_accumulate_every = (
            1  # Number of batches before taking an optimizer step
        )
        self.update_ema_every = 10  # Number of steps before updating the EMA

        self.ema_decay = ema_decay
        self.ema = EMA(self.ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        self.train_lr = train_lr

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.ema_model(batch)
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.ema_model(batch)
            self.log(
                "test_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.train_lr)
        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # Reference from old PyTorch (pre 1.1.0) for backward compatibility
        if (batch_idx + 1) % self.gradient_accumulate_every == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update the EMA model
        if self.trainer.global_step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.model)


# # Assume you have train_dataloader, val_dataloader, and test_dataloader already defined

# # Instantiate the model
# model = DDPM(image_size=128)

# # Define the PyTorch Lightning trainer
# trainer = pl.Trainer(max_epochs=1)  # or any other configuration you'd like

# # Train the model
# trainer.fit(model, train_dataloader, val_dataloader)
# trainer.test(model, test_dataloader)
