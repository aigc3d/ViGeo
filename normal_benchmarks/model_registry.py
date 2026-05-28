import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(PROJECT_ROOT), str(CURRENT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from checkpoint_utils import checkpoint_path, hf_from_pretrained


NORMAL_MODELS = ['dsine', 'normalcrafter', 'stablenormal', 'lotus']
CHECKPOINT_DIR = CURRENT_DIR / 'checkpoints'
LOTUS_MODEL_ID = 'jingheya/lotus-normal-d-v1-1'


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_name):
    model_name = model_name.lower()
    device = _device()
    if model_name == 'dsine':
        return torch.hub.load(
            "hugoycj/DSINE-hub",
            "DSINE",
            local_file_path=str(checkpoint_path(CHECKPOINT_DIR, 'dsine.pt')),
            trust_repo=True,
        )

    if model_name == 'normalcrafter':
        from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
        from diffusers import AutoencoderKLTemporalDecoder
        from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline

        unet_path = "Yanrui95/NormalCrafter"
        pre_train_path = "stabilityai/stable-video-diffusion-img2vid-xt"
        unet = hf_from_pretrained(
            DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained,
            unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        vae = hf_from_pretrained(
            AutoencoderKLTemporalDecoder.from_pretrained,
            unet_path,
            subfolder="vae",
        )
        vae.to(dtype=torch.float16)
        unet.to(dtype=torch.float16)

        model = hf_from_pretrained(
            NormalCrafterPipeline.from_pretrained,
            pre_train_path,
            unet=unet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        model.enable_model_cpu_offload()
        try:
            model.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        return model

    if model_name == 'stablenormal':
        from stablenormal import load_stablenormal

        return load_stablenormal(device)

    if model_name == 'lotus':
        from lotus.pipeline import LotusDPipeline

        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        pipeline = hf_from_pretrained(
            LotusDPipeline.from_pretrained,
            LOTUS_MODEL_ID,
            torch_dtype=dtype,
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipeline.lotus_timestep = 999
        pipeline.lotus_processing_res = 0
        return pipeline

    raise ValueError(f"Model '{model_name}' not implemented.")
