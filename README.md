#  ⚡️Lightning Generative Models
Harness the capabilities of **[PyTorch Lightning](https://lightning.ai/)** and **[Weights & Biases (W&B)](https://wandb.ai/site)** to implement and train a variety of deep generative models, while seamlessly logging your experiments.

![generative_models](assets/generative_models.png)

*Figure modified from https://lilianweng.github.io/*

## Wandb Experiment Logging
**Weights & Biases** provides machine learning developers tools to track their experiments, visualize model training and metrics, and optimize machine learning models.
The figure below is an example of the experiments logging interface on the Wandb platform.

![Wandb Experiments](assets/wandb_experiments.png)
Visit the [Wandb Experiment Page](https://wandb.ai/i_am_seungjun/Lightning%2520generative%2520models?workspace%253Duser-i_am_seungjun) for more details.

## 🔧 Installation
I have tested on **(1) the macOS M1** and **(2) Ubuntu NVIDIA Titan X** with GPU acceleration.

```bash
# Clone the repository
git clone https://github.com/seungjunlee96/lightning-generative-models.git
cd lightning-generative-models

# Set up a conda environment
conda create -n lightning-generative-models python=3.11 -y
conda activate lightning-generative-models
pip install -r environments/requirements.txt

# For contributors
pre-commit install
```

## 🌟 Models

| Category             | Model      | Status | Paper Link                                                                                           |
|----------------------|------------|--------|------------------------------------------------------------------------------------------------------------------|
| GANs                 | GAN        | ✅     | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)                                            |
|                      | CGAN       | ✅     | [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)                                    |
|                      | DCGAN      | ✅     | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) |
|                      | LSGAN      | ✅     | [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)                             |
|                      | WGAN       | ✅     | [Wasserstein GAN](https://arxiv.org/abs/1701.07875)                                                           |
|                      | WGAN-GP    | ✅     | [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)                                      |
|                      | CycleGAN   | -      | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/) |
| VAEs                 | VAE        | ✅     | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)                                            |
|                      | VQVAE      | ✅     | [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)                                   |
| Autoregressive Models| PixelCNN   | -      | [Conditional Image Generation with PixelCNN Decoders](https://ar5iv.org/abs/1606.05328)                     |
| Normalizing Flows    | NICE       | -      |                                                                                                                |
|                      | RealNVP    | -      |                                                                                                                |
|                      | Glow       | -      |                                                                                                                |
| Diffusion Models     | DDPM       | ✅     | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)                                |
|                      | DDIM       | -      |                                                                                                                |


## 🚀 Usage
### (1) Download data
While automatic dataset downloading is supported, ome datasets (e.g., LSUN) require manual downloading:

#### Download CycleGAN dataset
```bash
# Example: monet2photo dataset
# Available Options: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos
chmod +x ./data/download_cyclegan_dataset.sh
./data/download_cyclegan_dataset.sh monet2photo
```

#### Download Pix2Pix dataset
```bash
chmod +x ./data/download_pix2pix_dataset.sh
./data/download_pix2pix_dataset.sh
```

#### LSUN dataset
```bash
chmod +x ./data/download_lsun_dataset.sh

# Default to download only bedroom dataset
./data/download_lsun_dataset.sh

# If you wish to download all LSUN dataset
./data/download_lsun_dataset.sh all
```

### (2) Run `train.py`
Use the `config` parser with `train.py`, for example:

```bash
# Train GAN with MNIST
python train.py --config configs/gan/gan.json --experiment_name gan

# Train LSGAN
python train.py --config configs/gan/lsgan.json --experiment_name lsgan

# Train CGAN
python train.py --config configs/gan/cgan.json --experiment_name cgan

# Train DCGAN
python train.py --config configs/gan/dcgan.json --experiment_name dcgan

# Train WGAN with weight clipping
python train.py --config configs/gan/wgan_cp.json --experiment_name wgan_cp

# Train WGAN with gradient penalty
python train.py --config configs/gan/wgan_gp.json --experiment_name wgan_gp

# Train VAE
python train.py --config configs/vae/vae.json --experiment_name vae

# Train VQVAE
python train.py --config configs/vae/vqvae.json --experiment_name vqvae

# Train VQVAE with EMA (Exponential Moving Average)
python train.py --config configs/vae/vqvae_ema.json --experiment_name vqvae_ema

# Train DDPM
python train.py --config configs/diffusion/ddpm.json --experiment_name ddpm

# ... and many more
```

## 📊 Evaluation Metrics
Assess the quality and performance of your generative models with:

1. Inception Score (IS)
2. Fréchet Inception Distance (FID)
3. Precision and Recall Distribution (PRD)
4. Density and Coverage metrics
5. Reconstruction-FID

## 🤝 Contributing
All contributions are welcome! Open an issue for discussions or submit a pull request directly.

## 📩 Contact
For queries or feedback, email me at lsjj096@gmail.com.

## 📚 References
This repo is highly motivated by below amazing works:
- https://github.com/nocotan/pytorch-lightning-gans
- https://nn.labml.ai/
- https://github.com/eriklindernoren/PyTorch-GAN
