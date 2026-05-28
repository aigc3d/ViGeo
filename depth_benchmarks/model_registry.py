from pathlib import Path
import sys

import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(PROJECT_ROOT), str(CURRENT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from checkpoint_utils import hf_from_pretrained, load_checkpoint

CHECKPOINT_DIR = CURRENT_DIR / 'checkpoints'

VIDEO_DEPTH_MODELS = [
    'da3',
    'flashdepth',
    'pi3',
    'stream3r',
    'vda',
    'geometrycrafter',
    'vggt',
    'vggt_omega',
    'streamvggt',
    'depthcrafter',
    'infinitevggt',
]

MONO_DEPTH_MODELS = [
    'da3',
    'flashdepth',
    'pi3',
    'stream3r',
    'vda',
    'geometrycrafter',
    'vggt',
    'vggt_omega',
    'streamvggt',
    'depthcrafter',
    'infinitevggt',
]

POINTMAP_MODELS = [
    'geometrycrafter',
    'vggt',
    'vggt_omega',
    'pi3',
    'da3',
    'streamvggt',
    'stream3r',
]

ALL_MODEL_NAMES = sorted(set(VIDEO_DEPTH_MODELS + MONO_DEPTH_MODELS + POINTMAP_MODELS))


class GeometryCrafterWrapper:
    """
    Wrapper for GeometryCrafter components to maintain a unified interface.
    """
    def __init__(self, pipe, vae, prior):
        self.pipe = pipe
        self.vae = vae
        self.prior = prior

def get_model(model_name):
    """
    Instantiates and returns the specified depth/geometry estimation model.
    """
    model_name = model_name.lower()

    # --------------------------------------------------------------------------
    # 1. DepthAnything V3
    # --------------------------------------------------------------------------
    if model_name == 'da3':
        from depth_anything_3.api import DepthAnything3
        model = hf_from_pretrained(DepthAnything3.from_pretrained, "depth-anything/DA3-GIANT-1.1")

    # --------------------------------------------------------------------------
    # 2. FlashDepth
    # --------------------------------------------------------------------------
    elif model_name == 'flashdepth':
        from flashdepth.model import FlashDepth
        model = FlashDepth()
        ckpt = load_checkpoint(CHECKPOINT_DIR, 'flashdepth', 'iter_10001.pth')
        model.load_state_dict(ckpt['model'], strict=False)

    # --------------------------------------------------------------------------
    # 3. Pi3
    # --------------------------------------------------------------------------
    elif model_name == 'pi3':
        from pi3.models.pi3 import Pi3
        model = hf_from_pretrained(Pi3.from_pretrained, "yyfz233/Pi3")

    # --------------------------------------------------------------------------
    # 4. VGGT
    # --------------------------------------------------------------------------
    elif model_name == 'vggt':
        from vggt.models.vggt import VGGT
        model = hf_from_pretrained(VGGT.from_pretrained, "facebook/VGGT-1B")

    # --------------------------------------------------------------------------
    # 5. VGGT-Omega
    # --------------------------------------------------------------------------
    elif model_name == 'vggt_omega':
        from vggt_omega.models import VGGTOmega
        model = VGGTOmega()
        model.load_state_dict(load_checkpoint(CHECKPOINT_DIR, 'vggt_omega', 'vggt_omega_1b_512.pt'))

    # --------------------------------------------------------------------------
    # 6. STream3R (Returns a Session object)
    # --------------------------------------------------------------------------
    elif model_name == 'stream3r':
        from stream3r.models.stream3r import STream3R
        from stream3r.stream_session import StreamSession
        base_model = hf_from_pretrained(STream3R.from_pretrained, "yslan/STream3R")
        base_model.cuda().eval()
        return StreamSession(base_model, mode="causal")

    # --------------------------------------------------------------------------
    # 7. StreamVGGT
    # --------------------------------------------------------------------------
    elif model_name == 'streamvggt':
        from streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT(total_budget=None)
        model.load_state_dict(load_checkpoint(CHECKPOINT_DIR, 'streamvggt', 'checkpoints.pth'), strict=True)

    # --------------------------------------------------------------------------
    # 8. Video Depth Anything (VDA)
    # --------------------------------------------------------------------------
    elif model_name == 'vda':
        from vda.video_depth import VideoDepthAnything
        model = VideoDepthAnything(**{
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        })
        model.load_state_dict(load_checkpoint(CHECKPOINT_DIR, 'vda', 'video_depth_anything_vitl.pth'), strict=True)

    # --------------------------------------------------------------------------
    # 9. GeometryCrafter (Complex Initialization)
    # --------------------------------------------------------------------------
    elif model_name == 'geometrycrafter':
        from moge.model.v1 import MoGeModel
        from geometrycrafter import (
            GeometryCrafterDiffPipeline,
            PMapAutoencoderKLTemporalDecoder,
            UNetSpatioTemporalConditionModelVid2vid
        )

        # Inner MoGe Class for Point Map Prior
        class MoGe(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = hf_from_pretrained(MoGeModel.from_pretrained, 'Ruicheng/moge-vitl').eval()

            @torch.no_grad()
            def forward_image(self, image: torch.Tensor, **kwargs):
                output = self.model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
                return output['points'], output['mask']

        # UNet Setup (FP16)
        unet = hf_from_pretrained(
            UNetSpatioTemporalConditionModelVid2vid.from_pretrained,
            'TencentARC/GeometryCrafter', subfolder='unet_diff',
            low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).requires_grad_(False).to("cuda", dtype=torch.float16)

        # VAE Setup (FP32)
        point_map_vae = hf_from_pretrained(
            PMapAutoencoderKLTemporalDecoder.from_pretrained,
            'TencentARC/GeometryCrafter', subfolder='point_map_vae',
            low_cpu_mem_usage=True, torch_dtype=torch.float32
        ).requires_grad_(False).to("cuda", dtype=torch.float32)

        # Prior Model (FP32)
        prior_model = MoGe().requires_grad_(False).to('cuda', dtype=torch.float32)

        # Diffusion Pipeline
        pipe = hf_from_pretrained(
            GeometryCrafterDiffPipeline.from_pretrained,
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

        # Optimization for Inference
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass
        pipe.enable_attention_slicing()

        return GeometryCrafterWrapper(pipe, point_map_vae, prior_model)

    # --------------------------------------------------------------------------
    # 10. DepthCrafter
    # --------------------------------------------------------------------------
    elif model_name == 'depthcrafter':
        from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
        from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

        print("Loading DepthCrafter UNet...")
        unet = hf_from_pretrained(
            DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained,
            "tencent/DepthCrafter",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )

        print("Loading DepthCrafter Pipeline...")
        pipe = hf_from_pretrained(
            DepthCrafterPipeline.from_pretrained,
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet, torch_dtype=torch.float16, variant="fp16"
        )

        # Enable sequential/model offload to prevent OOM
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass

        return pipe

    # --------------------------------------------------------------------------
    # 11. InfiniteVGGT
    # --------------------------------------------------------------------------
    elif model_name == 'infinitevggt':
        from streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT(total_budget=1200000)
        model.load_state_dict(load_checkpoint(CHECKPOINT_DIR, 'streamvggt', 'checkpoints.pth'), strict=True)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Final Setup for standard models
    model.cuda()
    model.eval()

    return model
