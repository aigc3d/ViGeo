import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from mmengine import MODELS
from einops import rearrange
from .conv_head import ConvHead
from .layers.dinov2 import DINOv2
from .camera_head import CameraHead
from .transformer_decoder import TransformerDecoder
from .layers import RotaryPositionEmbedding2D, PositionGetter
from .utils import depth_to_pointmap, recover_focal_from_xy

@MODELS.register_module()
class ViGeoTrain(nn.Module):
    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"],
        train_normal: bool = False,
        train_conf: bool = False,
        train_mask: bool = True,
        warm_epoch: int = 10,
        mode: Literal["offline", "chunk", "online"] = "offline",
        seed: int = 42,
        epoch: int = 0,
    ):
        super().__init__()
        self.patch_size = 14
        self.warm_epoch = warm_epoch
        self.mode = mode

        rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()
        self.pretrained = DINOv2(encoder, rope=rope)

        self.decoder = TransformerDecoder(
            in_dim=2*self.pretrained.embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=rope)

        conv_kwargs = dict(
            projects=nn.Identity(), dim_proj=1024, dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2, num_res_blocks=2,
            last_res_blocks=0, last_conv_channels=32, last_conv_size=1, using_uv=True
        )

        self.point_head = ConvHead(dim_out=3, **conv_kwargs)
        self.ray_head = ConvHead(dim_out=6, **conv_kwargs)
        self.camera_head = CameraHead(dim_in=2*self.pretrained.embed_dim)

        if train_normal:
            self.normal_head = ConvHead(dim_out=3, **conv_kwargs)

        if train_conf:
            self.conf_head = ConvHead(dim_out=1, **conv_kwargs)

        if train_mask:
            self.mask_head = ConvHead(dim_out=1, **conv_kwargs)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.chunk_generator = torch.Generator(device='cpu').manual_seed(seed)
        self.epoch = epoch
        self.set_epoch()

    def _prepare_rope(self, B, S, H, W, device):
        pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)
        pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
        if self.pretrained.patch_start_idx > 0:
            pos_special = torch.zeros(B, S, self.pretrained.patch_start_idx, 2, dtype=pos.dtype, device=device)
            pos = torch.cat([pos_special, pos + 1], dim=2) # [b, s, n, c]
        return pos

    def _remap(self, points):
        xy, z = points.split([2, 1], dim=-3)
        z = z.clamp(-10, 10).exp()
        return torch.cat([xy * z, z], dim=-3), z

    def forward(self, sample):
        if self.training:
            return self.forward_train(sample)
        else:
            return self.infer(sample)

    def forward_train(self, sample):
        image = sample["image"]
        B, S, _, H, W = image.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        x = (image - self.image_mean) / self.image_std

        segment_lengths = generate_train_segments(self.mode, S, generator=self.chunk_generator, warm=self.epoch < self.warm_epoch)

        with torch.amp.autocast('cuda', enabled=True):
            pos = self._prepare_rope(B, S, H, W, device=image.device)
            x = self.pretrained(x, pos=pos, segment_lengths=segment_lengths)
            hidden = self.decoder(x["x_norm_patchtokens"], pos=pos) # [(bs) n c]

        with torch.amp.autocast('cuda', dtype=torch.float32):
            point_logits = self.point_head(hidden, patch_h, patch_w)
            points, depth_pred = self._remap(point_logits)
            ray_pred = self.ray_head(hidden, patch_h, patch_w)
            normal_pred = self.normal_head(hidden, patch_h, patch_w) if hasattr(self, "normal_head") else None
            conf_pred = self.conf_head(hidden, patch_h, patch_w) if hasattr(self, "conf_head") else None
            mask_pred = self.mask_head(hidden, patch_h, patch_w) if hasattr(self, "mask_head") else None
            pose_pred = self.camera_head(x["camera_tokens"]) # [b, n, 7]
            
            def format_pred(t, permute=True):
                if t is None:
                    return None
                t = t.reshape(B, S, -1, H, W)
                return t.permute(0, 1, 3, 4, 2) if permute else t

        return {
            'points_pred': format_pred(points),
            'depth_pred': format_pred(depth_pred, permute=False),
            'ray_pred': format_pred(ray_pred),
            'pose_pred': pose_pred,
            'normal_pred': format_pred(normal_pred, permute=False),
            'conf_pred': format_pred(conf_pred, permute=False),
            'mask_pred': format_pred(mask_pred, permute=False),
        }

    def _infer_chunk(self, x_chunk, pos_chunk, patch_h, patch_w, use_cache, kv_caches):
        B, curr_step, _, H, W = x_chunk.shape

        with torch.amp.autocast('cuda', enabled=True):
            out = self.pretrained(x_chunk, pos=pos_chunk, use_cache=use_cache, past_key_values=kv_caches)
            new_kv_caches = out["new_kv_caches"] if use_cache else kv_caches

            hidden = self.decoder(out["x_norm_patchtokens"], pos=pos_chunk)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            point_logits = self.point_head(hidden, patch_h, patch_w)
            ray = self.ray_head(hidden, patch_h, patch_w)
            pose = self.camera_head(out["camera_tokens"])
            normal = self.normal_head(hidden, patch_h, patch_w) if hasattr(self, "normal_head") else None
            conf = self.conf_head(hidden, patch_h, patch_w) if hasattr(self, "conf_head") else None
            mask = self.mask_head(hidden, patch_h, patch_w) if hasattr(self, "mask_head") else None

            def format_pred(t):
                if t is None:
                    return None
                return t.reshape(B, curr_step, -1, H, W)

        return (
            format_pred(point_logits),
            format_pred(ray),
            pose,
            format_pred(normal),
            format_pred(conf),
            format_pred(mask),
            new_kv_caches,
        )

    @torch.no_grad()
    def infer(
        self,
        sample,
        mode: Literal["offline", "chunk", "online"] = "offline",
        chunk_size: int = 16,
        num_tokens: int = 1369,
    ):
        image = sample["image"]
        image = image.to(dtype=self.dtype, device=self.device)
        B, S, C, orig_H, orig_W = image.shape

        aspect_ratio = orig_W / orig_H
        patch_h = max(1, round((num_tokens / aspect_ratio) ** 0.5))
        patch_w = max(1, round((num_tokens * aspect_ratio) ** 0.5))
        H, W = int(patch_h * self.patch_size), int(patch_w * self.patch_size)

        if H != orig_H or W != orig_W:
            image = F.interpolate(image.flatten(0, 1), size=(H, W), mode='bilinear', align_corners=False).view(B, S, C, H, W)

        x = (image - self.image_mean) / self.image_std
        if mode == "offline":
            step_size, use_cache = S, False
        elif mode == "chunk":
            step_size, use_cache = max(1, min(chunk_size, S)), True
        elif mode == "online":
            step_size, use_cache = 1, True
        else:
            raise ValueError(f"Unsupported inference mode: {mode}")

        with torch.amp.autocast('cuda', enabled=True):
            full_pos = self._prepare_rope(B, S, H, W, device=self.device)

        kv_caches = None
        all_point_logits, all_poses, all_rays, all_normals, all_confs, all_masks = [], [], [], [], [], []

        for t in range(0, S, step_size):
            end_t = min(t + step_size, S)

            point_logits, ray, pose, normal, conf, mask, kv_caches = self._infer_chunk(
                x[:, t:end_t],
                full_pos[:, t:end_t],
                patch_h,
                patch_w,
                use_cache,
                kv_caches,
            )

            all_point_logits.append(point_logits)
            all_rays.append(ray)
            all_poses.append(pose)
            if normal is not None:
                all_normals.append(normal)
            if conf is not None:
                all_confs.append(conf)
            if mask is not None:
                all_masks.append(mask)

        point_logits_pred = torch.cat(all_point_logits, dim=1)
        ray_pred = torch.cat(all_rays, dim=1)
        pose_pred = torch.cat(all_poses, dim=1)
        normal_pred = torch.cat(all_normals, dim=1) if all_normals else None
        conf_pred = torch.cat(all_confs, dim=1) if all_confs else None
        mask_pred = torch.cat(all_masks, dim=1) if all_masks else None
        resized_output = H != orig_H or W != orig_W

        if resized_output:
            def resize(t):
                if t is None: return None
                return F.interpolate(t.flatten(0, 1), size=(orig_H, orig_W), mode='bilinear', align_corners=False).view(B, S, -1, orig_H, orig_W)

            point_logits_pred = resize(point_logits_pred)
            ray_pred, normal_pred, conf_pred, mask_pred = resize(ray_pred), resize(normal_pred), resize(conf_pred), resize(mask_pred)

        points_chw, depth_pred = self._remap(point_logits_pred)
        focal_length = recover_focal_from_xy(point_logits_pred[:, :, :2]) if resized_output else None
        points_pred = (
            depth_to_pointmap(depth_pred, focal_length)
            if resized_output else points_chw.permute(0, 1, 3, 4, 2)
        )

        return {
            'points_pred': points_pred,
            'depth_pred': depth_pred,
            'pose_pred': pose_pred,
            'ray_pred': ray_pred.permute(0, 1, 3, 4, 2),
            'normal_pred': normal_pred.permute(0, 1, 3, 4, 2) if normal_pred is not None else None,
            'conf_pred': conf_pred,
            'mask_pred': mask_pred,
            'focal_length': focal_length,
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def set_epoch(self, epoch: int | None = None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

def generate_train_segments(mode: Literal["offline", "chunk", "online"], S: int, generator: torch.Generator = None, warm: bool = False):
    """Generate training-only temporal segments for global attention.

    Inference and evaluation do not use these segments; chunk/online behavior
    is controlled by input window size and KV cache there.
    """
    if warm or mode == "offline":
        return None

    if mode == "online":
        return [1] * S

    if mode != "chunk":
        raise ValueError(f"Unsupported training mode: {mode}")

    if S <= 1:
        return [S]

    num_segments = torch.randint(1, S + 1, (1,), generator=generator).item()

    if num_segments == 1:
        return [S]
    if num_segments == S:
        return [1] * S

    split_points = torch.randperm(S - 1, generator=generator)[:num_segments - 1] + 1

    boundaries = torch.cat([torch.tensor([0, S], device=split_points.device), split_points])
    boundaries, _ = torch.sort(boundaries)

    return torch.diff(boundaries).tolist()
