import importlib

GENERATIVE_MODELS = [
    "autoencoder",
    "autoregressive",
    "diffusion",
    "flow",
    "gan",
    "vae",
]


def load_model(
    model: str,
    img_channels: int,
    img_size: int,
):
    """
    Load a model based on its name from predefined model families.

    Args:
        model_name (str): The name of the model, e.g., 'VAE', 'GAN'.
        img_channels (int): Image channels.
        img_size (int): Image size (height/width).

    Returns:
        object: An instance of the desired model.

    Raises:
        ValueError: If the model isn't found in the families.
    """
    for generative_model in GENERATIVE_MODELS:
        try:
            # Try to import the model from the current generative_model
            module = importlib.import_module(
                f"models.generative.{generative_model}.{model.lower()}"
            )
            model_class = getattr(module, model.upper())
            return model_class(
                img_channels=img_channels,
                img_size=img_size,
            )
        except ImportError:
            continue
    raise ValueError(f"Model {model} not found in any generative_model.")
