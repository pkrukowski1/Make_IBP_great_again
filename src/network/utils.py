import torch.nn as nn
import torch

def load_linear_model(model: nn.Module, weight_path: str, device: str = "cpu") -> None:
    """
    Loads the weights of an MLP model from a specified file path.
    Args:
        model (nn.Module): The PyTorch model instance to load the weights into.
        weight_path (str): The file path to the saved model weights.
        device (str, optional): The device to map the model weights to. Defaults to "cpu".
    Notes:
        - The function attempts to extract the "state_dict" key from the loaded weights.
        - If the state dictionary is a list, it assumes a specific naming convention for keys
          and adjusts them accordingly before loading into the model.
        - The `strict` parameter in `load_state_dict` is set to `False`, allowing for partial
          loading of weights if the model architecture does not match exactly.
    Returns:
        None
    """
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict.get("state_dict", state_dict)

    if isinstance(state_dict, list):
        state_dict = state_dict[0]
    state_dict = {f"{int(key.split('.')[0])-1}.{key.split('.')[1]}": val for key, val in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def load_conv_model(model: nn.Module, weight_path: str, device: str = "cpu") -> None:
    """
    Loads the weights of a CNN model from a specified file path.
    Args:
        model (nn.Module): The PyTorch model instance to load the weights into.
        weight_path (str): The file path to the saved model weights.
        device (str, optional): The device to map the model weights to. Defaults to "cpu".
    Notes:
        - Attempts to extract the "state_dict" key from the loaded weights.
        - Maps state dictionary keys to match the model's expected naming convention.
        - Logs mismatched keys and raises an error if strict loading fails.
    Returns:
        None
    """
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = state_dict.get("state_dict", state_dict)

    if isinstance(state_dict, list):
        state_dict = state_dict[0]

    model_state_dict = model.state_dict()
    adjusted_state_dict = {}

    for key, val in state_dict.items():
        try:
            parts = key.split('.')
            if len(parts) == 2 and parts[1] in ['weight', 'bias']:
                new_key = f"{parts[0]}.0.{parts[1]}"
            else:
                new_key = key
        except (ValueError, IndexError):
            new_key = key

        if new_key in model_state_dict:
            if model_state_dict[new_key].shape == val.shape:
                adjusted_state_dict[new_key] = val
            else:
                print(f"Shape mismatch for key {new_key}: expected {model_state_dict[new_key].shape}, got {val.shape}")
        else:
            for model_key in model_state_dict:
                if model_key.endswith(parts[-1]) and model_state_dict[model_key].shape == val.shape:
                    print(f"Mapping {key} to {model_key} based on partial match")
                    adjusted_state_dict[model_key] = val
                    break
            else:
                print(f"Key {new_key} (from {key}) not found in model state_dict")

    if not adjusted_state_dict:
        raise RuntimeError("No matching keys found in adjusted state_dict. Cannot load weights.")
    model.load_state_dict(adjusted_state_dict, strict=True)
