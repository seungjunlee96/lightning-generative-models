import os

import pytorch_lightning as pl


def seed_everything(
    seed: int = 42,
    workers: bool = False,
):
    """
    Seed all randomness sources and configure PyTorch for deterministic algorithms.

    Args:
        seed (int): The seed to use for random number generation.
        workers (bool): If True, seeds the workers for PyTorch DataLoader for reproducibility.
        cuda_deterministic (bool): Whether to enforce CUDA deterministic algorithms. This controls
                                   both the use of deterministic algorithms and cuDNN settings.

    Returns:
        int: The seed used for all random number generators.

    Note: Deterministic operations may impact performance. This function configures the global
          environment and affects all CUDA BLAS operations.

    References:
    - https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    """
    seed = pl.seed_everything(seed=seed, workers=workers)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # if cuda_deterministic:
    #     torch.use_deterministic_algorithms(True)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'

    # # Filling uninitialized memory can be enabled for further determinism at the cost of performance
    # torch.utils.deterministic.fill_uninitialized_memory = True

    return seed
