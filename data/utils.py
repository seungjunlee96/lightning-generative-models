from typing import Union

import torch
from torch import Tensor


class CenterCropMinXY:
    """
    Custom transform that performs a center crop on the image in the smaller dimension (X or Y).
    """

    def __call__(self, image: Union[Tensor, any]) -> Tensor:
        """
        Perform the center crop on the image.

        :param image: The input image as a torch.Tensor.
        :return: The cropped image as a torch.Tensor.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError('Input image should be a torch.Tensor')

        # Get the height and width of the image
        _, h, w = image.shape

        # Determine the smaller dimension
        min_dim = min(h, w)

        # Calculate top and left coordinates for cropping
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2

        # Perform the crop
        image = image[:, top: top + min_dim, left: left + min_dim]

        return image
