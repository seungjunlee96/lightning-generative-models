# ⚡️Lightning Generative Models

Implementation of deep generative models using **[PyTorch Lightning](https://lightning.ai/)** to train and **[Weights & Biases (W&B)](https://wandb.ai/site)** to log experiments!

## 🌟 Models

- GANs
    - [GAN](models/generative/gan/gan.py)
    - [DCGAN](models/generative/gan/dcgan.py)
    - [LSGAN](models/generative/gan/lsgan.py)
    - [WGAN](models/generative/gan/wgan.py)
    - (Upcoming) [CycleGAN](models/generative/gan/cyclegan.py)

- VAE
    - [VAE](models/generative/vae/vae.py)
    - [VQVAE](models/generative/vae/vqvae.py)

- Autoregressive models
    - (Upcoming) PixelCNN

- Normalizing flows
    - (Upcoming) NICE (Non-linear Independent Components Estimation)
    - (Upcoming) RealNVP
    - (Upcoming) Glow

- Diffusion models
    - (Upcoming) DDPM
    - (Upcoming) DDIM

## 📊 Evaluation Metrics
Assess the quality and performance of generative models using:

1. Inception Score (IS)
2. Fréchet Inception Distance (FID)
3. Precision and Recall Distribution (PRD)
4. Density and Coverage metrics
5. Reconstruction-FID

## 🔧 Installation

**macOS M1 Support**: This repository has been tested and is compatible with macOS on the M1 chip with GPU acceleration. Ensure that you have the necessary dependencies and versions installed to leverage the M1's GPU capabilities.

```bash
# Clone the repository
$ git clone https://github.com/seungjunlee96/lightning-generative-models.git
$ cd lightning-generative-models

# Create conda environment
$ conda create -n lightning-generative-models python=3.11
$ conda activate lightning-generative-models
$ pip install -r environments/requirements.txt

# For development
$ pre-commit install
```


## 🚀 Usage
### (1) Download data
Automatic-download supported, However, Some datasets (e.g., LSUN) require manual download:
```bash
$ git clone https://github.com/fyu/lsun.git ./data/dataset/LSUN
$ cd lsun
$ python3 download.py
$ for file in *.zip; do unzip "$file"; done
$ rm *.zip
```

### (2) Run `train.py`
Run `train.py` with `config` parser to train from, for example:

```bash
# Train GAN
python train.py --config configs/gan/gan.json

# Train DCGAN
python train.py --config configs/gan/dcgan.json

# Train WGAN with gradient penalty
python train.py --config configs/gan/wgan_gp.json

# Train VAE
python train.py --config configs/vae/vae.json

# Train VQVAE
python train.py --config configs/vae/vqvae.json

# Train VQVAE with EMA (Exponential Moving Average)
python train.py --config configs/vae/vqvae_ema.json

# ... and many more
```

## 🤝 Contributing
All contributions are appreciated! Open an issue for discussions or submit a pull request directly.

## 📩 Contact
For queries or feedback, email me at lsjj096@gmail.com.

## 📚 References
- https://github.com/nocotan/pytorch-lightning-gans
- https://nn.labml.ai/
