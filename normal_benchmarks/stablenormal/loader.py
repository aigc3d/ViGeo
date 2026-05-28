from enum import Enum
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from checkpoint_utils import hf_from_pretrained

from .pipeline_stablenormal import StableNormalPipeline
from .pipeline_yoso_normal import YOSONormalsPipeline
from .scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler


YOSO_MODEL_ID = "Stable-X/yoso-normal-v0-3"
STABLENORMAL_MODEL_ID = "Stable-X/stable-normal-v0-1"


class DataType(Enum):
    INDOOR = "indoor"


def resize_image(input_image: Image.Image, resolution: int = 1024) -> Image.Image:
    if not isinstance(input_image, Image.Image):
        raise ValueError("input_image should be a PIL Image object")

    h, w = np.asarray(input_image).shape[:2]
    scale = float(resolution) / max(float(h), float(w))
    new_h = int(np.round(h * scale / 64.0)) * 64
    new_w = int(np.round(w * scale / 64.0)) * 64
    return input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)


class StableNormalPredictor:
    def __init__(self, model, yoso_version: str = YOSO_MODEL_ID):
        self.model = model
        self.yoso_version = yoso_version

    def to(self, device: Union[str, torch.device] = "cuda", dtype: torch.dtype = torch.float16):
        self.model.to(device, dtype)
        return self

    @torch.no_grad()
    def __call__(
        self,
        img: Image.Image,
        resolution: int = 1024,
        match_input_resolution: bool = True,
        data_type: Union[DataType, str] = DataType.INDOOR,
        num_inference_steps: Optional[int] = None,
    ) -> Image.Image:
        if isinstance(data_type, str):
            data_type = DataType(data_type.lower())
        if data_type is not DataType.INDOOR:
            raise ValueError("Only indoor StableNormal inference is wired for benchmark evaluation.")

        orig_size = img.size
        img = resize_image(img.convert("RGB"), resolution)

        kwargs = {}
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        pipe_out = self.model(img, match_input_resolution=match_input_resolution, **kwargs)

        prediction = pipe_out.prediction[0]
        normal_map = (prediction.clip(-1, 1) + 1) / 2
        normal_map = (normal_map * 255).astype(np.uint8)
        normal_map = Image.fromarray(normal_map)
        if match_input_resolution:
            normal_map = normal_map.resize(orig_size, Image.Resampling.LANCZOS)
        return normal_map


def load_stablenormal(device):
    dtype = torch.float16
    common_kwargs = {
        "variant": "fp16",
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "safety_checker": None,
    }
    x_start_pipeline = hf_from_pretrained(
        YOSONormalsPipeline.from_pretrained,
        YOSO_MODEL_ID,
        **common_kwargs,
    ).to(device)

    pipe = hf_from_pretrained(
        StableNormalPipeline.from_pretrained,
        STABLENORMAL_MODEL_ID,
        **common_kwargs,
        scheduler=HEURI_DDIMScheduler(
            prediction_type="sample",
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="scaled_linear",
        ),
    )
    pipe.x_start_pipeline = x_start_pipeline
    pipe.to(device)
    pipe.prior.to(device, dtype)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return StableNormalPredictor(pipe)
