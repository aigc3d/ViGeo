from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_BENCHMARK_DIR = PROJECT_ROOT / "depth_benchmarks"
for path in (str(PROJECT_ROOT), str(DEPTH_BENCHMARK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from checkpoint_utils import hf_from_pretrained  # noqa: E402
from recon_benchmarks.datasets import ReconstructionScene  # noqa: E402


OFFLINE_MODELS = ("pi3", "vggt", "da3")
ONLINE_MODELS = ("streamvggt", "stream3r", "infinitevggt")
VIGEO_MODES = ("offline", "online", "chunk")
VIGEO_MODELS = ("vigeo_offline", "vigeo_online", "vigeo_chunk")
ALL_MODELS = OFFLINE_MODELS + ONLINE_MODELS + VIGEO_MODELS + ("vigeo",)
MODEL_TYPES = {
    **{name: "offline" for name in OFFLINE_MODELS},
    **{name: "online" for name in ONLINE_MODELS},
    "vigeo_offline": "offline",
    "vigeo_online": "online",
    "vigeo_chunk": "chunk",
    "vigeo": "vigeo",
}


@dataclass
class ReconstructionPrediction:
    points: torch.Tensor
    confidence: torch.Tensor | None = None


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    return state_dict


def load_model(model_name: str, checkpoint: str | Path | None = None):
    model_name = model_name.lower()
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else None

    if model_name in ("streamvggt", "infinitevggt"):
        from streamvggt.models.streamvggt import StreamVGGT

        if checkpoint_path is None:
            checkpoint_path = DEPTH_BENCHMARK_DIR / "checkpoints" / "streamvggt" / "checkpoints.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing StreamVGGT checkpoint: {checkpoint_path}")
        model = StreamVGGT(total_budget=1_200_000) if model_name == "infinitevggt" else StreamVGGT()
        model.load_state_dict(_state_dict_from_checkpoint(checkpoint_path), strict=True)

    elif model_name == "vggt":
        from vggt.models.vggt import VGGT

        model = hf_from_pretrained(VGGT.from_pretrained, "facebook/VGGT-1B")

    elif model_name == "pi3":
        from pi3.models.pi3 import Pi3

        model = hf_from_pretrained(Pi3.from_pretrained, "yyfz233/Pi3")

    elif model_name == "da3":
        from depth_anything_3.api import DepthAnything3

        model = hf_from_pretrained(DepthAnything3.from_pretrained, "depth-anything/DA3-GIANT-1.1")

    elif model_name == "stream3r":
        from stream3r.models.stream3r import STream3R
        from stream3r.stream_session import StreamSession

        base_model = hf_from_pretrained(STream3R.from_pretrained, "yslan/STream3R").cuda().eval()
        return StreamSession(base_model, mode="causal")

    elif model_name in VIGEO_MODELS:
        from vigeo.vigeo import ViGeo

        if checkpoint_path is None:
            raise ValueError(f"--checkpoint is required for {model_name}")
        model = ViGeo()
        incompatible = model.load_state_dict(_state_dict_from_checkpoint(checkpoint_path), strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("mask_head.")]
        required_missing = [key for key in incompatible.missing_keys if not key.startswith("normal_head.")]
        if unexpected or required_missing:
            raise RuntimeError(
                f"ViGeo checkpoint mismatch. Missing={required_missing[:8]}, unexpected={unexpected[:8]}"
            )
    else:
        raise ValueError(f"Unknown reconstruction model: {model_name}")

    model.cuda().eval()
    return model


def _resize_pointmap(points: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if points.shape[1:3] == target_hw:
        return points
    resized = F.interpolate(
        points.permute(0, 3, 1, 2),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1)


@torch.no_grad()
def _infer_streamvggt(model, images: torch.Tensor, total_budget: int | None) -> ReconstructionPrediction:
    device = next(model.parameters()).device
    images = images.to(device)
    past_key_values = [None] * model.aggregator.depth
    past_key_values_camera = [None] * model.camera_head.trunk_depth
    points, confidences = [], []
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    for frame_index, frame in enumerate(images):
        frame_batch = frame[None, None]
        with torch.autocast(device_type="cuda", dtype=dtype):
            aggregator_output = model.aggregator(
                frame_batch,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=frame_index,
                total_budget=total_budget,
            )
        if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
            aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
        else:
            aggregated_tokens, patch_start_idx = aggregator_output

        with torch.autocast(device_type="cuda", enabled=False):
            _, past_key_values_camera = model.camera_head(
                aggregated_tokens,
                past_key_values_camera=past_key_values_camera,
                use_cache=True,
            )
            pointmap, confidence = model.point_head(
                aggregated_tokens,
                images=frame_batch,
                patch_start_idx=patch_start_idx,
            )
        points.append(pointmap[0, 0].float().cpu())
        confidences.append(confidence[0, 0].float().cpu())

    return ReconstructionPrediction(torch.stack(points), torch.stack(confidences))


@torch.no_grad()
def _infer_vggt(model, images: torch.Tensor) -> ReconstructionPrediction:
    output = model(images.cuda())
    return ReconstructionPrediction(
        output["world_points"][0].float().cpu(),
        output["world_points_conf"][0].float().cpu(),
    )


@torch.no_grad()
def _infer_pi3(model, images: torch.Tensor, target_hw: tuple[int, int]) -> ReconstructionPrediction:
    output = model(images.cuda())
    points = output["points"][0].float()
    camera_poses = output.get("camera_poses")
    if camera_poses is not None:
        camera_poses = camera_poses[0].float()
        first_to_world = camera_poses[0]
        world_to_first = torch.linalg.inv(first_to_world)
        points_h = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
        points = torch.einsum("ij,shwj->shwi", world_to_first, points_h)[..., :3]
    points = _resize_pointmap(points, target_hw).cpu()
    confidence = output.get("conf")
    if confidence is not None:
        confidence = _resize_pointmap(confidence[0].float(), target_hw).squeeze(-1).cpu()
    return ReconstructionPrediction(points, confidence)


def _depth_to_reference_points(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> torch.Tensor:
    depth_t = torch.from_numpy(depth).float()
    intrinsics_t = torch.from_numpy(intrinsics).float()
    extrinsics_t = torch.from_numpy(extrinsics).float()
    if extrinsics_t.shape[-2:] == (3, 4):
        bottom = torch.zeros((*extrinsics_t.shape[:-2], 1, 4), dtype=extrinsics_t.dtype)
        bottom[..., 0, 3] = 1.0
        extrinsics_t = torch.cat((extrinsics_t, bottom), dim=-2)
    sequence, height, width = depth_t.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xx, yy = xx.float(), yy.float()
    result = []
    for index in range(sequence):
        z = depth_t[index]
        x = (xx - intrinsics_t[index, 0, 2]) / intrinsics_t[index, 0, 0] * z
        y = (yy - intrinsics_t[index, 1, 2]) / intrinsics_t[index, 1, 1] * z
        camera = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
        camera_to_world = torch.linalg.inv(extrinsics_t[index])
        world = torch.einsum("ij,hwj->hwi", camera_to_world, camera)
        reference = torch.einsum("ij,hwj->hwi", extrinsics_t[0], world)[..., :3]
        result.append(reference)
    return torch.stack(result)


@torch.no_grad()
def _infer_da3(model, images: torch.Tensor, target_hw: tuple[int, int]) -> ReconstructionPrediction:
    pil_images = [
        Image.fromarray((image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8))
        for image in images
    ]
    prediction = model.inference(
        pil_images,
        ref_view_strategy="first",
        process_res=max(target_hw),
        process_res_method="upper_bound_resize",
    )
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 reconstruction requires predicted extrinsics and intrinsics.")
    points = _depth_to_reference_points(prediction.depth, prediction.intrinsics, prediction.extrinsics)
    points = _resize_pointmap(points, target_hw)
    confidence = None
    if prediction.conf is not None:
        confidence = F.interpolate(
            torch.from_numpy(prediction.conf).float().unsqueeze(1),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return ReconstructionPrediction(points.cpu(), confidence)


@torch.no_grad()
def _infer_stream3r(session, images: torch.Tensor) -> ReconstructionPrediction:
    session.clear()
    output = None
    for frame in images.cuda():
        output = session.forward_stream(frame[None])
    points = output["world_points"][0].float().cpu()
    confidence = output.get("world_points_conf")
    return ReconstructionPrediction(points, confidence[0].float().cpu() if confidence is not None else None)


def _transform_local_points(points: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    if poses.shape[-2:] == (3, 4):
        bottom = torch.zeros((*poses.shape[:-2], 1, 4), dtype=poses.dtype, device=poses.device)
        bottom[..., 0, 3] = 1
        poses = torch.cat((poses, bottom), dim=-2)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    world = torch.einsum("sij,shwj->shwi", poses, homogeneous)
    first_inverse = torch.linalg.inv(poses[0])
    return torch.einsum("ij,shwj->shwi", first_inverse, world)[..., :3]


@torch.no_grad()
def _infer_vigeo(model, images: torch.Tensor, model_name: str, chunk_size: int, total_budget: int) -> ReconstructionPrediction:
    mode = model_name.removeprefix("vigeo_")
    output = model.infer(
        images,
        mode=mode,
        chunk_size=chunk_size,
        total_budget=total_budget,
        resize_output=True,
    )
    points = output["points_pred"].float()
    poses = output["pose_pred"].float()
    points = _transform_local_points(points, poses)
    confidence = output.get("conf_pred")
    if confidence is not None:
        confidence = confidence.squeeze(1) if confidence.ndim == 4 else confidence
    return ReconstructionPrediction(points.cpu(), confidence.cpu() if confidence is not None else None)


def infer_scene(
    model,
    model_name: str,
    scene: ReconstructionScene,
    chunk_size: int = 16,
    total_budget: int = 1_200_000,
) -> ReconstructionPrediction:
    target_hw = tuple(scene.depths.shape[-2:])
    if model_name == "streamvggt":
        return _infer_streamvggt(model, scene.images, total_budget=None)
    if model_name == "infinitevggt":
        return _infer_streamvggt(model, scene.images, total_budget=total_budget)
    if model_name == "vggt":
        return _infer_vggt(model, scene.images)
    if model_name == "pi3":
        return _infer_pi3(model, scene.images, target_hw)
    if model_name == "da3":
        return _infer_da3(model, scene.images, target_hw)
    if model_name == "stream3r":
        return _infer_stream3r(model, scene.images)
    if model_name in VIGEO_MODELS:
        return _infer_vigeo(model, scene.images, model_name, chunk_size, total_budget)
    raise ValueError(f"Unsupported reconstruction model: {model_name}")
