from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

from .conv_head import ConvHead
from .layers.prior_dinov2 import PriorDINOv2
from .moge.model.v2 import MoGeModel
from .poisson_completion import poisson_completion
from .utils import log, median


MOGE_VITS_CONFIG = {
    "encoder": {
        "backbone": "dinov2_vits14",
        "intermediate_layers": [5, 11],
        "dim_out": 384,
    },
    "neck": {
        "dim_in": [386, 2, 2, 2, 2],
        "dim_out": None,
        "dim_res_blocks": [384, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "points_head": {
        "dim_in": [384, 256, 128, 64, 32],
        "dim_out": [None, None, None, None, 3],
        "dim_res_blocks": [384, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "mask_head": {
        "dim_in": [384, 256, 128, 64, 32],
        "dim_out": [None, None, None, None, 1],
        "dim_res_blocks": [384, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "scale_head": {"dims": [384, 384, 384, 1]},
    "remap_output": "exp",
    "num_tokens_range": [1200, 3600],
}


class videoldcm(nn.Module):
    def __init__(
        self,
        encoder: Literal["vitl"] = "vitl",
    ):
        super().__init__()
        self.patch_size = 14

        self.moge = MoGeModel(**MOGE_VITS_CONFIG)
        for param in self.moge.parameters():
            param.requires_grad = False

        self.pretrained = PriorDINOv2(model_name=encoder, use_additional_prior=True)

        conv_kwargs = dict(
            dim_proj=2 * self.pretrained.embed_dim,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True,
        )
        self.point_head = ConvHead(dim_out=3, **conv_kwargs)
        self.conf_head = ConvHead(dim_out=1, **conv_kwargs)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        filename: str = "videoldcm.pt",
        **hf_kwargs,
    ) -> "videoldcm":
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            checkpoint_path = path / filename
        elif path.exists():
            checkpoint_path = path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=filename,
                repo_type="model",
                **hf_kwargs,
            )

        model = cls()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("state_dict", state_dict.get("model", state_dict))
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith("moge.normal_head.")
        }
        model.load_state_dict(state_dict, strict=True)
        return model

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        target_h = ((height + self.patch_size - 1) // self.patch_size) * self.patch_size
        target_w = ((width + self.patch_size - 1) // self.patch_size) * self.patch_size
        return target_h, target_w

    def _resize_sequence(self, x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x

        batch, frames, channels = x.shape[:3]
        kwargs = dict(size=size, mode=mode)
        if mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        x = F.interpolate(x.reshape(batch * frames, channels, *x.shape[-2:]), **kwargs)
        return x.reshape(batch, frames, channels, *size)

    def _resize_points(self, points: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if points.shape[2:4] == size:
            return points

        batch, frames = points.shape[:2]
        points = points.permute(0, 1, 4, 2, 3).reshape(batch * frames, 3, *points.shape[2:4])
        points = F.interpolate(points, size=size, mode="bilinear", align_corners=False)
        return points.reshape(batch, frames, 3, *size).permute(0, 1, 3, 4, 2)

    def _valid_depth_mask(self, depth: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(depth) & (depth > 0)

    def _masked_log_depth(
        self,
        depth: torch.Tensor,
        valid_mask: torch.Tensor,
        medians: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(valid_mask, log(depth / medians), torch.zeros_like(depth))

    def _remap(self, points: torch.Tensor, medians: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xy, z = points.split([2, 1], dim=2)
        z = z.clamp(-10, 10).exp() * medians
        return torch.cat([xy * z, z], dim=2), z

    def _prepare_depth_priors(
        self,
        prior: torch.Tensor,
        coarse_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prior_mask = self._valid_depth_mask(prior)
        coarse_mask = mask.bool() & self._valid_depth_mask(coarse_depth)
        prior = prior.masked_fill(~prior_mask, 0.0)
        coarse_depth = coarse_depth.masked_fill(~coarse_mask, 0.0)

        medians = median(prior)
        prior = self._masked_log_depth(prior, prior_mask, medians)
        coarse_depth = self._masked_log_depth(coarse_depth, coarse_mask, medians)
        return prior, coarse_depth, medians, prior_mask, coarse_mask

    def _predict_completion(
        self,
        image: torch.Tensor,
        prior: torch.Tensor,
        coarse_depth: torch.Tensor,
        medians: torch.Tensor,
        prior_mask: torch.Tensor,
        coarse_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames, _, height, width = image.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        device_type = image.device.type

        x = (image - self.image_mean) / self.image_std
        with torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
            features = self.pretrained(
                x,
                x_depth=prior,
                x_depth_coarse=coarse_depth,
                x_depth_mask=prior_mask,
                x_depth_coarse_mask=coarse_mask,
            )
            hidden = rearrange(features["x_norm_patchtokens"], "b s n c -> (b s) n c")

        with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=device_type == "cuda"):
            points = self.point_head(hidden, patch_h, patch_w).reshape(batch, frames, -1, height, width)
            conf = self.conf_head(hidden, patch_h, patch_w).reshape(batch, frames, -1, height, width)
            points, depth = self._remap(points, medians)

        return {
            "points_pred": points.permute(0, 1, 3, 4, 2),
            "depth_pred": depth,
            "conf_pred": conf,
        }

    def forward(
        self,
        image: torch.Tensor,
        prior: torch.Tensor,
        coarse_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prior, coarse_depth, medians, prior_mask, coarse_mask = self._prepare_depth_priors(
            prior, coarse_depth, mask
        )
        output = self._predict_completion(image, prior, coarse_depth, medians, prior_mask, coarse_mask)
        output["mask"] = mask.bool()
        return output

    @torch.no_grad()
    def infer(
        self,
        image: torch.Tensor,
        sparse_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = self.image_mean.device
        image = image.to(device, non_blocking=True)
        sparse_depth = sparse_depth.to(device, non_blocking=True).float()
        sparse_depth = sparse_depth.masked_fill(~self._valid_depth_mask(sparse_depth), 0.0)

        moge_dtype = next(self.moge.parameters()).dtype

        moge_out = self.moge.infer(
            image.to(dtype=moge_dtype),
            apply_mask=False,
            force_projection=True,
        )
        moge_mask = moge_out["mask"].unsqueeze(1).bool()
        mono_depth = moge_out["depth"].unsqueeze(1).float()
        mono_depth = mono_depth.masked_fill(~moge_mask, 0.0)

        coarse_depth = poisson_completion(
            sparse=sparse_depth,
            mono_depth=mono_depth,
            confidence=moge_mask.float(),
            num_scales=5,
            thres=3.0,
            lamda=5.0,
            rtol=1e-5,
            max_iter_per_scale=[5000, 2000, 1000, 500, 250],
            max_resolution_ratio=0.5,
        )
        coarse_depth = coarse_depth.masked_fill(~moge_mask, 0.0)

        return self.infer_without_poisson(
            image=image,
            prior=sparse_depth,
            coarse_depth=coarse_depth,
            mask=moge_mask,
        )

    @torch.no_grad()
    def infer_without_poisson(
        self,
        image: torch.Tensor,
        prior: torch.Tensor,
        coarse_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = image.shape[-2:]
        original_size = (height, width)
        target_size = self._target_size(height, width)
        device = self.image_mean.device

        image = image.unsqueeze(0).to(device, non_blocking=True)
        prior = prior.unsqueeze(0).to(device, non_blocking=True)
        coarse_depth = coarse_depth.unsqueeze(0).to(device, non_blocking=True)
        mask = mask.unsqueeze(0).to(device, non_blocking=True).bool()

        image = self._resize_sequence(image, target_size, "bilinear")
        prior = self._resize_sequence(prior, target_size, "nearest")
        coarse_depth = self._resize_sequence(coarse_depth, target_size, "bilinear")
        mask = self._resize_sequence(mask.float(), target_size, "nearest").bool()

        output = self.forward(image, prior, coarse_depth, mask)

        if target_size != original_size:
            output["points_pred"] = self._resize_points(output["points_pred"], original_size)
            output["depth_pred"] = self._resize_sequence(output["depth_pred"], original_size, "bilinear")
            output["conf_pred"] = self._resize_sequence(output["conf_pred"], original_size, "bilinear")
            output["mask"] = self._resize_sequence(output["mask"].float(), original_size, "nearest").bool()

        return {key: value.squeeze(0) for key, value in output.items()}
