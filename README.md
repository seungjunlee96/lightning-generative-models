# ⚡️ Lightning Generative Models
Harness the capabilities of **[PyTorch Lightning](https://lightning.ai/)** and **[Weights & Biases (W&B)](https://wandb.ai/site)** to implement and train a variety of deep generative models. Seamlessly log and visualize your experiments for an efficient and impactful machine learning development process!

## 👀 Overview
Lightning Generative Models is designed to provide an intuitive and robust framework for working with different types of generative models. It leverages the simplicity of PyTorch Lightning and the comprehensive tracking capabilities of Weights & Biases.

![generative_models](assets/generative_models.png)

*Figure modified from https://lilianweng.github.io/*

## 🌟 Supported Models

| Category             | Model    | Status | Paper Link                                                                                                             |
|----------------------|----------|--------|------------------------------------------------------------------------------------------------------------------------|
| **GANs**             | GAN      | ✅     | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)                                                     |
|                      | CGAN     | ✅     | [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)                                             |
|                      | InfoGAN  | ✅     | [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) |
|                      | DCGAN    | ✅     | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) |
|                      | LSGAN    | ✅     | [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)                                      |
|                      | WGAN     | ✅     | [Wasserstein GAN](https://arxiv.org/abs/1701.07875)                                                                    |
|                      | WGAN-GP  | ✅     | [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)                                              |
|                      | R1GAN    | ✅     | [Which Training Methods for GANs do actually Converge?](https://arxiv.org/pdf/1801.04406.pdf)                          |
|                      | CycleGAN | -      | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/) |
| **VAEs**             | VAE      | ✅     | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)                                                     |
|                      | VQVAE    | ✅     | [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)                                            |
| **Autoregressive Models** | PixelCNN | -  | [Conditional Image Generation with PixelCNN Decoders](https://ar5iv.org/abs/1606.05328)                                |
| **Normalizing Flows**| NICE     | -      |                                                                                                                        |
|                      | RealNVP  | -      |                                                                                                                        |
|                      | Glow     | -      |                                                                                                                        |
| **Diffusion Models** | DDPM     | ✅     | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)                                           |
|                      | DDIM     | ✅     | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)                                               |

## 📝 Wandb Experiment Logging
**Weights & Biases** provides tools to track experiments, visualize model training and metrics, and optimize machine learning models. Below is an example of the experiments logging interface on the Wandb platform.

![Wandb Experiments](assets/wandb_experiments.png)
Visit the [Wandb Experiment Page](https://wandb.ai/i_am_seungjun/Lightning%2520generative%2520models?workspace%253Duser-i_am_seungjun) for more details.

## 🔧 Installation
Tested on both **Apple Silicon (M1 Max)** and **Ubuntu NVIDIA GPUs**, supporting **GPU acceleration** and **Distributed Data Parallel (DDP)**.

#### 🐍 For Conda Users
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

#### 🐳 For Docker Users
```bash
cd environments
chmod +x ./install_and_run_docker.sh
./install_and_run_docker.sh
```

## 🚀 Train
Easily train different generative models using the config parser with train.py. Examples include:

```bash
# Train GAN
python train.py --config configs/gan/gan.json --experiment_name gan

# Train LSGAN
python train.py --config configs/gan/lsgan.json --experiment_name lsgan

# Train CGAN
python train.py --config configs/gan/cgan.json --experiment_name cgan

# Train InfoGAN
python train.py --config configs/gan/infogan.json --experiment_name infogan

# Train DCGAN
python train.py --config configs/gan/dcgan.json --experiment_name dcgan

# Train WGAN with weight clipping
python train.py --config configs/gan/wgan_cp.json --experiment_name wgan_cp

# Train WGAN with gradient penalty
python train.py --config configs/gan/wgan_gp.json --experiment_name wgan_gp

# Train GAN with R1 penalty
python train.py --config configs/gan/r1gan.json --experiment_name r1gan

# Train VAE
python train.py --config configs/vae/vae.json --experiment_name vae

# Train VQVAE
python train.py --config configs/vae/vqvae.json --experiment_name vqvae

# Train VQVAE with EMA (Exponential Moving Average)
python train.py --config configs/vae/vqvae_ema.json --experiment_name vqvae_ema

# Train DDPM
python train.py --config configs/diffusion/ddpm.json --experiment_name ddpm

# Train DDIM
python train.py --config configs/diffusion/ddim.json --experiment_name ddim

# ... and many more
```

## 📊 Evaluation Metrics
Assess the quality and performance of your generative models with:

#### 1. [Inception Score (IS)](https://arxiv.org/abs/1606.03498)
$$
  IS(G) = \exp \left( \mathbb{E}_{\mathbf{x}\sim p_g} \left[ KL(p(y|\mathbf{x}) || p(y)) \right] \right)
$$
  - The Inception Score measures the diversity and quality of images generated by a model. Higher scores indicate better image quality and variety, suggesting the model's effectiveness in producing diverse, high-fidelity images.
  - $G$: Generative model
  - $\mathbf{x}$: Data samples generated by $G$
  - $p_g$: Probability distribution of generated samples
  - $p(y|\mathbf{x})$: Conditional probability distribution of labels given sample $\mathbf{x}$
  - $p(y)$: Marginal distribution of labels over the dataset
  - $KL$: Kullback-Leibler divergence

#### 2. [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1512.00567)
$$
  FID = ||\mu_x - \mu_g||^2 + \text{Tr}(\Sigma_x + \Sigma_g - 2(\Sigma_x \Sigma_g)^{1/2})
$$
  - The Fréchet Inception Distance evaluates the quality of generated images by comparing the feature distribution of generated images to that of real images. Lower scores indicate that the generated images are more similar to real images, implying higher quality.
  - $\mu_x$, $\Sigma_x$: Mean and covariance of the feature vectors of real images
  - $\mu_g$, $\Sigma_g$: Mean and covariance of the feature vectors of generated images

#### 3. [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401)
$$
  KID = \frac{1}{m(m-1)} \sum_{i \neq j} k(x_i, x_j) + \frac{1}{n(n-1)} \sum_{i \neq j} k(y_i, y_j) - \frac{2}{mn} \sum_{i,j} k(x_i, y_j)
$$
  - The Kernel Inception Distance computes the distance between the feature representations of real and generated images. Lower KID scores suggest a higher similarity between the generated images and real images, indicating better generative model performance.
  - $k$: Kernel function measuring similarity between image features
  - $x_i$, $x_j$: Feature vectors of real images
  - $y_i$, $y_j$: Feature vectors of generated images
  - $m$, $n$: Number of real and generated images respectively

#### 4. TODO: [Precision and Recall Distribution (PRD)](https://arxiv.org/abs/1806.00035)
- **Description:** PRD offers a novel way to assess generative models by disentangling the evaluation of sample quality from the coverage of the target distribution. Unlike one-dimensional scores, PRD provides a two-dimensional evaluation that separately quantifies the precision and recall of a distribution, offering a more nuanced understanding of a model's performance.

## 🤝 Contributing
All contributions are welcome! Open an issue for discussions or submit a pull request directly.

## 📩 Contact
For queries or feedback, email me at lsjj096@gmail.com.

## 📚 References
This repo is highly motivated by below amazing works:
- https://github.com/nocotan/pytorch-lightning-gans
- https://nn.labml.ai/
- https://github.com/eriklindernoren/PyTorch-GAN
