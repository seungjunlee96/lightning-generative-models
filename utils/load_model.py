import importlib
from typing import Dict

GENERATIVE_MODELS = [
    "autoencoder",
    "autoregressive",
    "diffusion",
    "flow",
    "gan",
    "vae",
]


def load_model(model_config: Dict):
    """
    Load a model based on its name from predefined model families.

    Args:
        model_config (dict): The dictionary storing model configurations

    Returns:
        object: An instance of the desired model.

    Raises:
        ValueError: If the model isn't found in the families.
    """
    model_name = model_config["name"]
    errors = []
    for generative_model in GENERATIVE_MODELS:
        try:
            # Try to import the model from the current generative_model
            module = importlib.import_module(
                f"models.generative.{generative_model}.{model_name.lower()}"
            )
            model_class = getattr(module, model_name.upper())
            return model_class(**model_config["args"])
        except ImportError as e:
            errors.append(str(e))
            continue
    raise ValueError(
        f"Failed to import {model_name}. Errors encountered: {', '.join(errors)}"
    )
