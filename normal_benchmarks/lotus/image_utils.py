from typing import Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image


def _torch_resize_mode(resample_method: str):
    return "nearest" if resample_method in {"nearest", "nearest-exact"} else resample_method


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: str = "bilinear",
) -> torch.Tensor:
    assert img.dim() == 4, f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width,
        max_edge_resolution / original_height,
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)
    mode = _torch_resize_mode(resample_method)
    return F.interpolate(img, size=(new_height, new_width), mode=mode)


def resize_back(
    img: Union[torch.Tensor, np.ndarray, PIL.Image.Image, list[PIL.Image.Image]],
    target_size: Union[int, tuple[int, int]],
    resample_method: Union[str, int] = "bilinear",
) -> Union[torch.Tensor, np.ndarray, PIL.Image.Image, list[PIL.Image.Image]]:
    if isinstance(img, torch.Tensor):
        mode = _torch_resize_mode(resample_method)
        return F.interpolate(img, size=target_size, mode=mode)
    if isinstance(img, np.ndarray):
        img_tensor = torch.tensor(img).permute(0, 3, 1, 2)
        mode = _torch_resize_mode(resample_method)
        resized = F.interpolate(img_tensor, size=target_size, mode=mode)
        return resized.permute(0, 2, 3, 1).numpy()
    if isinstance(img, PIL.Image.Image):
        return img.resize((target_size[1], target_size[0]), resample_method)
    if isinstance(img, list) and all(isinstance(item, PIL.Image.Image) for item in img):
        return [item.resize((target_size[1], target_size[0]), resample_method) for item in img]
    raise TypeError(f"Unsupported image type for resize_back: {type(img)}")


def get_pil_resample_method(method_str: str) -> int:
    resample_method_dict = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST,
    }
    resample_method = resample_method_dict.get(method_str)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {method_str}")
    return resample_method


def get_tv_resample_method(method_str: str) -> str:
    resample_method_dict = {
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "nearest": "nearest",
        "nearest-exact": "nearest",
    }
    resample_method = resample_method_dict.get(method_str)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {method_str}")
    return resample_method
