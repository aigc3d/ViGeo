from typing import Literal

import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .conv_head import ConvHead
from .layers.prior_dinov2 import PriorDINOv2
from .moge.model.v2 import MoGeModel
from .poisson_completion import poisson_completion
from .utils import log, median


@mmengine.MODELS.register_module()
class videoldcm(nn.Module):
    def __init__(
        self,
        moge_path: str,
        encoder: Literal["vitl"] = "vitl",
        train_conf: bool = True,
    ):
        super().__init__()
        self.patch_size = 14

        self.moge = MoGeModel.from_pretrained(moge_path)
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
        if train_conf:
            self.conf_head = ConvHead(dim_out=1, **conv_kwargs)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    @torch.no_grad()
    def prepare_input(self, image, prior):
        b, s = image.shape[:2]
        image_input = image.reshape(b * s, *image.shape[2:]).half()

        moge_result = self.moge.infer(image_input)
        mono_depth = moge_result["depth"].unsqueeze(1).float()
        mask = moge_result["mask"].unsqueeze(1)
        mono_depth = torch.where(mask, mono_depth, 0.0)

        device_type = image.device.type
        with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=device_type == "cuda"):
            prior_mask = torch.isfinite(prior) & (prior > 0)
            prior = torch.where(prior_mask, prior, torch.zeros_like(prior))
            medians = median(prior)

            coarse_depth = poisson_completion(
                sparse=prior.reshape(b * s, *prior.shape[2:]),
                mono_depth=mono_depth,
                confidence=mask.float(),
                num_scales=5,
                thres=3.0,
                lamda=5.0,
                rtol=1e-5,
                max_iter_per_scale=[5000, 2000, 1000, 500, 250],
                max_resolution_ratio=0.5,
            )
            coarse_depth = torch.where(mask.reshape_as(coarse_depth), coarse_depth, 0.0)
            coarse_depth = coarse_depth.reshape(b, s, *coarse_depth.shape[1:])
            coarse_mask = mask.reshape_as(coarse_depth).bool() & torch.isfinite(coarse_depth) & (coarse_depth > 0)

            prior = torch.where(prior_mask, log(prior / medians), torch.zeros_like(prior))
            coarse_depth = torch.where(coarse_mask, log(coarse_depth / medians), torch.zeros_like(coarse_depth))

        return prior, coarse_depth, medians, mask, prior_mask, coarse_mask

    def forward(self, sample):
        image, prior = sample["image"], sample["prior"]
        b, s, _, h, w = image.shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        image_input = (image - self.image_mean) / self.image_std
        prior, coarse_depth, medians, mask, prior_mask, coarse_mask = self.prepare_input(image, prior)

        device_type = image.device.type
        with torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
            features = self.pretrained(
                image_input,
                x_depth=prior,
                x_depth_coarse=coarse_depth,
                x_depth_mask=prior_mask,
                x_depth_coarse_mask=coarse_mask,
            )
            hidden = rearrange(features["x_norm_patchtokens"], "b s n c -> (b s) n c")

        with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=device_type == "cuda"):
            points = self.point_head(hidden, patch_h, patch_w).reshape(b, s, -1, h, w)
            conf = self.conf_head(hidden, patch_h, patch_w).reshape(b, s, -1, h, w) \
                if hasattr(self, "conf_head") else None

            xy, z = points.split([2, 1], dim=2)
            z = z.clamp(-10, 10).exp() * medians
            points = torch.cat([xy * z, z], dim=2)

        return {
            "points_pred": points.permute(0, 1, 3, 4, 2),
            "depth_pred": z,
            "conf_pred": conf,
            "mask": mask.reshape(b, s, 1, h, w),
        }

    def _target_size(self, height, width):
        target_h = ((height + self.patch_size - 1) // self.patch_size) * self.patch_size
        target_w = ((width + self.patch_size - 1) // self.patch_size) * self.patch_size
        return target_h, target_w

    def _resize_video_tensor(self, x, size, mode):
        if x.shape[-2:] == size:
            return x
        b, s, c = x.shape[:3]
        kwargs = dict(size=size, mode=mode)
        if mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        x = F.interpolate(x.flatten(0, 1), **kwargs)
        return x.view(b, s, c, *size)

    def _resize_points(self, points, size):
        if points.shape[2:4] == size:
            return points
        b, s = points.shape[:2]
        points = points.permute(0, 1, 4, 2, 3).flatten(0, 1)
        points = F.interpolate(points, size=size, mode="bilinear", align_corners=False)
        return points.view(b, s, 3, *size).permute(0, 1, 3, 4, 2)

    @torch.no_grad()
    def infer(self, sample):
        image, prior = sample["image"], sample["prior"]

        has_batch_dim = image.dim() == 5
        if image.dim() == 4:
            image = image.unsqueeze(0)
            prior = prior.unsqueeze(0)
        elif image.dim() != 5:
            raise ValueError(f"image must have shape [T, C, H, W] or [B, T, C, H, W], got {tuple(image.shape)}")

        _, _, _, height, width = image.shape
        original_size = (height, width)
        target_size = self._target_size(height, width)
        device = self.image_mean.device

        image = self._resize_video_tensor(image.to(device, non_blocking=True), target_size, "bilinear")
        prior = self._resize_video_tensor(prior.to(device, non_blocking=True), target_size, "nearest")
        output = self.forward({"image": image, "prior": prior})

        if target_size != original_size:
            output["points_pred"] = self._resize_points(output["points_pred"], original_size)
            output["depth_pred"] = self._resize_video_tensor(output["depth_pred"], original_size, "bilinear")
            output["mask"] = self._resize_video_tensor(output["mask"].float(), original_size, "nearest").bool()
            if output["conf_pred"] is not None:
                output["conf_pred"] = self._resize_video_tensor(output["conf_pred"], original_size, "bilinear")

        if not has_batch_dim:
            output = {key: value.squeeze(0) if value is not None else None for key, value in output.items()}

        return output
