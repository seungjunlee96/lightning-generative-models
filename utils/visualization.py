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


def make_grid(tensor, nrows=None, ncols=None):
    """
    Convert a batch of PyTorch tensors into a single grid image.

    Parameters:
        tensor (torch.Tensor): The tensor to convert. Should have shape (B, C, H, W).
        nrows (int, optional): Number of rows in the grid. If None, it is inferred.
        ncols (int, optional): Number of columns in the grid. If None, it is inferred.

    Returns:
        PIL.Image: The grid image.
    """

    num_images, channels, height, width = tensor.shape

    # Infer grid size if not provided
    if nrows is None and ncols is None:
        nrows = int(np.ceil(np.sqrt(num_images)))
        ncols = nrows
    elif nrows is None:
        nrows = (num_images + ncols - 1) // ncols
    elif ncols is None:
        ncols = (num_images + nrows - 1) // nrows

    grid_img = Image.new(
        "RGB" if channels == 3 else "L", (width * ncols, height * nrows)
    )

    for idx, img_tensor in enumerate(tensor):
        img = tensor_to_pil_img(img_tensor)
        row = idx // ncols
        col = idx % ncols
        grid_img.paste(img, (width * col, height * row))

    return grid_img
