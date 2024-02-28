import importlib
import json
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
            model_class = getattr(module, model_name)
            return model_class(**model_config["args"])
        except ImportError as e:
            errors.append(str(e))
            continue
    raise ValueError(
        f"Failed to import {model_name}. Errors encountered: {', '.join(errors)}"
    )


def load_config(config_path: str):
    """
    Loads a configuration file and performs a sanity check to ensure that img_channels and img_size
    match between the model and dataset configurations.

    Args:
    - config_path (str): Path to the configuration file.

    Returns:
    - dict: The loaded configuration if the sanity check passes.

    Raises:
    - FileNotFoundError: If the configuration file does not exist.
    - ValueError: If the img_channels or img_size do not match.
    - json.JSONDecodeError: If the file is not a valid JSON.
    """
    # Load the configuration file
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")
    except json.JSONDecodeError:
        raise ValueError(f"The file at '{config_path}' is not a valid JSON.")

    # Extract model and dataset configurations
    model_config = config.get("model", {}).get("args", {})
    dataset_config = config.get("dataset", {})

    # Sanity check for img_channels and img_size
    if model_config.get("img_channels") != dataset_config.get("img_channels"):
        raise ValueError(
            "Mismatch in 'img_channels' between model and dataset configurations."
        )
    if model_config.get("img_size") != dataset_config.get("img_size"):
        raise ValueError(
            "Mismatch in 'img_size' between model and dataset configurations."
        )

    return config
