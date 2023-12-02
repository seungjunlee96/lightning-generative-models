# Base image
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG PYTHON_VERSION=3.11

# Install Python and system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python3-pip \
    ca-certificates build-essential software-properties-common apt cmake zip unzip \
    curl wget vim git ssh sudo libx11-6 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/*.list

# Python libraries installation
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    torch \
    torchvision \
    pytorch-lightning \
    wandb \
    pydicom \
    pre-commit \
    black[e,ipykernel] \
    lmdb \
    torchinfo \
    einops \
    ema_pytorch \
    scipy

# Default command
CMD ["bash"]
