import os

from safetensors import safe_open

import torch
from functools import partial
from typing import Callable


def load_safetensors_to_dict(directory):
    safetensors_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    safetensors_dict[key] = f.get_tensor(key)
    return safetensors_dict


def construct_layer_dict(safetensors_dict, num_hidden_layers):
    layer_dict = {}
    is_mamba_layer = [False for _ in range(num_hidden_layers)]
    prefix = "model.layers."
    for full_key, tensor in safetensors_dict.items():
        if full_key.startswith(prefix):
            parts = full_key[len(prefix) :].split(".", 1)
            layer_id = int(parts[0])
            param_name = parts[1]
            if layer_id not in layer_dict:
                layer_dict[layer_id] = {}
            if "mamba" in param_name:
                is_mamba_layer[layer_id] = True
            layer_dict[layer_id][param_name] = tensor
    return layer_dict, is_mamba_layer

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)