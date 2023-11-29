import multiprocessing
import os
from typing import Optional

import torch
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy, Strategy


def is_master_process() -> bool:
    """
    Determines if the current process is the master process in a distributed setting.

    Returns:
        bool: True if the current process is the master process, False otherwise.
    """
    if torch.cuda.device_count() > 1:
        return dist.get_rank() == 0
    else:
        return True


def configure_strategy() -> Strategy:
    """
    Configures the appropriate PyTorch Lightning strategy based on the available hardware.

    Automatically detects the environment and chooses an optimal strategy.
    Supports:
    - Multiple GPUs on a Linux machine (using DDPStrategy).
    - A single GPU (including M1 Max GPU) (using SingleDeviceStrategy).
    - CPU (using SingleDeviceStrategy).

    Returns:
        Strategy: A PyTorch Lightning strategy object suitable for the detected hardware.
    """
    # Check for CUDA GPUs availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        # Use DDPStrategy for multiple GPUs
        if num_gpus > 1:
            print(f"Using DDPStrategy for distributed training on {num_gpus} GPUs.")
            return DDPStrategy(find_unused_parameters=True)

        # Use SingleDeviceStrategy for a single GPU
        print("Using SingleDeviceStrategy for training on a single GPU.")
        return SingleDeviceStrategy("cuda")

    # Check for Apple Silicon GPU availability
    elif torch.backends.mps.is_available():
        print("Using SingleDeviceStrategy for training on Apple Silicon GPU.")
        return SingleDeviceStrategy("mps")

    # Default to CPU strategy if no GPU is available
    print("Using SingleDeviceStrategy for training on CPU.")
    return SingleDeviceStrategy("cpu")


def configure_num_workers(num_gpus: Optional[int] = None) -> int:
    """
    Configures the optimal number of workers for PyTorch DataLoader based on the system's resources.

    Args:
        num_gpus (Optional[int]): The number of GPUs available for training.
                                   If None, the function will try to detect the number of GPUs.

    Returns:
        int: Recommended number of workers for DataLoader.
    """
    # Detect the number of GPUs if not provided
    # This allows for dynamic adjustment based on the available hardware
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    num_cpus = multiprocessing.cpu_count()

    # In a DDP setup (multiple GPUs), it's crucial to balance the CPU load across all GPUs.
    # Overloading the CPU with too many workers can lead to bottlenecks in data loading,
    # reducing overall training efficiency.
    if num_gpus > 1:
        # Assign fewer workers per DataLoader relative to the number of GPUs
        # This helps to prevent the CPU from becoming a bottleneck in multi-GPU setups
        num_workers = max(1, num_cpus // (num_gpus * 2))
    else:
        # In single-GPU or CPU-only environments, the strategy differs based on the OS.
        if os.name == "posix":  # Linux and macOS
            if "linux" in os.uname().sysname.lower():
                # Linux generally handles multi-threading more efficiently,
                # allowing us to use more workers per DataLoader
                num_workers = max(1, num_cpus // 2)
            else:
                # macOS, especially with M1 chip, may not scale as well with too many workers
                # due to differences in multiprocessing and I/O handling.
                num_workers = max(1, num_cpus // 4)
        else:
            # For other environments, a conservative approach with a single worker
            # helps avoid unforeseen issues.
            num_workers = 1

    return num_workers
