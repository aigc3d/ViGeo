import gc

import torch
import torch.nn as nn
from typing import List, Dict, Optional


def strip_prefixes(key: str, prefixes=("model.", "module.")) -> str:
    while True:
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            return key


def matches_module_prefix(key: str, prefixes: Optional[List[str]]) -> bool:
    if prefixes is None:
        return False

    for prefix in prefixes:
        prefix = prefix.rstrip(".")
        if key == prefix or key.startswith(f"{prefix}."):
            return True
    return False


def get_module_by_path(model: nn.Module, path: str):
    module = model
    for name in path.split("."):
        if not hasattr(module, name):
            return None
        module = getattr(module, name)
    return module


def load_weights_by_keys(
    model: nn.Module,
    ckpt_path: Optional[str],
    load_keys: Optional[List[str]] = None,
    ignore_keys: Optional[List[str]] = None,
    freeze_keys: Optional[List[str]] = None,
    copy_mapping: Optional[Dict[str, str]] = None,
    strict: bool = False,
    load_target: Optional[str] = None,
):
    if ckpt_path is None:
        return

    print(f"Loading weights from {ckpt_path}...")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    if state_dict is not ckpt:
        del ckpt
        gc.collect()

    if any(k.startswith(("model.", "module.model.")) for k in state_dict):
        state_dict = {
            k: v for k, v in state_dict.items()
            if k.startswith(("model.", "module.model."))
        }
    clean_state_dict = {strip_prefixes(k): v for k, v in state_dict.items()}

    filtered_dict = {}
    for k, v in clean_state_dict.items():
        if load_keys is not None and not matches_module_prefix(k, load_keys):
            continue
        if matches_module_prefix(k, ignore_keys):
            continue
        filtered_dict[k] = v

    load_module = get_module_by_path(model, load_target) if load_target else model
    if load_module is None:
        raise KeyError(f"Unknown load_target: {load_target}")

    missing_keys, unexpected_keys = load_module.load_state_dict(filtered_dict, strict=strict)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")

    if copy_mapping:        
        for target_name, source_name in copy_mapping.items():
            target_module = get_module_by_path(model, target_name)
            source_module = get_module_by_path(model, source_name)
            if target_module is not None and source_module is not None:
                print(f"Copying weights from {source_name} to {target_name}")
                target_module.load_state_dict(source_module.state_dict())

    if freeze_keys:
        frozen_count = 0
        for name, param in model.named_parameters():
            if matches_module_prefix(name, freeze_keys):
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} parameters based on freeze_keys.")