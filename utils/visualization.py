import numpy as np
from PIL import Image


def tensor_to_pil_img(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.

    Parameters:
        tensor (torch.Tensor): The tensor to convert. Should have shape (C, H, W).

    Returns:
        PIL.Image: The converted image.
    """

    # Handle both 3D and 4D tensors (single and batch of images)
    if len(tensor.shape) == 3:
        channels, height, width = tensor.shape
    elif len(tensor.shape) == 4:
        channels, height, width = tensor.shape[1:]
    else:
        raise ValueError("Expected tensor of shape (C, H, W) or (B, C, H, W).")

    array = tensor.detach().cpu().numpy()

    if channels == 1:  # Grayscale
        array = array.squeeze().astype(np.uint8)  # Remove the channel dimension
        return Image.fromarray(array, "L")
    elif channels == 3:  # RGB
        array = array.transpose(1, 2, 0).astype(np.uint8)
        return Image.fromarray(array)
    else:
        raise ValueError("Only 1 or 3 channels are supported")
