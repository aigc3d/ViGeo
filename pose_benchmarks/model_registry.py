from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_BENCHMARK_DIR = PROJECT_ROOT / "depth_benchmarks"
for path in (str(PROJECT_ROOT), str(DEPTH_BENCHMARK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from checkpoint_utils import hf_from_pretrained  # noqa: E402
from pose_benchmarks.metrics import c2w_to_pose_rows, invert_poses  # noqa: E402


POSE_MODELS = ("pi3", "vggt", "da3", "streamvggt", "stream3r")


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    return state_dict


def load_model(model_name: str, checkpoint: str | Path | None = None):
    model_name = model_name.lower()
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else None
    if model_name == "pi3":
        from pi3.models.pi3 import Pi3

        model = hf_from_pretrained(Pi3.from_pretrained, "yyfz233/Pi3")
    elif model_name == "vggt":
        from vggt.models.vggt import VGGT

        model = hf_from_pretrained(VGGT.from_pretrained, "facebook/VGGT-1B")
    elif model_name == "da3":
        from depth_anything_3.api import DepthAnything3

        model = hf_from_pretrained(DepthAnything3.from_pretrained, "depth-anything/DA3-GIANT-1.1")
    elif model_name == "streamvggt":
        from streamvggt.models.streamvggt import StreamVGGT

        if checkpoint_path is None:
            checkpoint_path = DEPTH_BENCHMARK_DIR / "checkpoints" / "streamvggt" / "checkpoints.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing StreamVGGT checkpoint: {checkpoint_path}")
        model = StreamVGGT()
        model.load_state_dict(_state_dict_from_checkpoint(checkpoint_path), strict=True)
    elif model_name == "stream3r":
        from stream3r.models.stream3r import STream3R
        from stream3r.stream_session import StreamSession

        base_model = hf_from_pretrained(STream3R.from_pretrained, "yslan/STream3R").cuda().eval()
        return StreamSession(base_model, mode="causal")
    else:
        raise ValueError(f"Unsupported pose model: {model_name}")
    return model.cuda().eval()


def _load_and_resize14(image_paths: list[Path], width: int, device: torch.device) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("No pose images provided.")
    original_width, original_height = images[0].size
    target_width = width
    target_height = round(original_height * (target_width / original_width) / 14) * 14
    to_tensor = tvf.ToTensor()
    tensors = [
        to_tensor(image.resize((target_width, target_height), Image.Resampling.LANCZOS))
        for image in images
    ]
    stacked = torch.stack(tensors).to(device)
    patch_h, patch_w = stacked.shape[-2] // 14, stacked.shape[-1] // 14
    return F.interpolate(
        stacked,
        (patch_h * 14, patch_w * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).unsqueeze(0)


def _model_device(model) -> torch.device:
    base_model = model.model if hasattr(model, "model") else model
    return next(base_model.parameters()).device


def _images_to_pil(images: torch.Tensor) -> list[Image.Image]:
    return [
        Image.fromarray((image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8"))
        for image in images
    ]


def _extrinsics_to_c2w(extrinsics: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(invert_poses(extrinsics.float().detach().cpu()))[:, :3]


@torch.no_grad()
def _infer_streamvggt_pose(model, images: torch.Tensor, total_budget: int | None = None) -> torch.Tensor:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = next(model.parameters()).device
    past_key_values = [None] * model.aggregator.depth
    past_key_values_camera = [None] * model.camera_head.trunk_depth
    pose_encs = []
    for frame_index, frame in enumerate(images):
        frame_batch = frame[None, None].to(device)
        aggregator_output = model.aggregator(
            frame_batch,
            past_key_values=past_key_values,
            use_cache=True,
            past_frame_idx=frame_index,
            total_budget=total_budget,
        )
        if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
            aggregated_tokens, _, past_key_values = aggregator_output
        else:
            aggregated_tokens, _ = aggregator_output
        pose_enc, past_key_values_camera = model.camera_head(
            aggregated_tokens,
            past_key_values_camera=past_key_values_camera,
            use_cache=True,
        )
        pose_encs.append(pose_enc[-1][:, 0])
    pose_enc = torch.stack(pose_encs, dim=1)
    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_enc.float(),
        image_size_hw=tuple(images.shape[-2:]),
        build_intrinsics=False,
    )
    return _extrinsics_to_c2w(extrinsics[0])


@torch.no_grad()
def _infer_stream3r_pose(session, images: torch.Tensor) -> torch.Tensor:
    from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri

    session.clear()
    output = None
    for frame in images.cuda():
        output = session.forward_stream(frame[None])
    if output is None or "pose_enc" not in output:
        raise RuntimeError("STream3R did not return pose_enc.")
    extrinsics, _ = pose_encoding_to_extri_intri(
        output["pose_enc"].float(),
        image_size_hw=tuple(images.shape[-2:]),
        build_intrinsics=False,
    )
    return _extrinsics_to_c2w(extrinsics[0])


@torch.no_grad()
def infer_pose(
    model,
    model_name: str,
    image_paths: list[Path],
    width: int = 512,
    chunk_size: int = 16,
    total_budget: int = 1_200_000,
) -> tuple:
    device = _model_device(model)
    images = _load_and_resize14(image_paths, width=width, device=device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    with torch.autocast(device_type="cuda", dtype=dtype):
        if model_name == "pi3":
            output = model(images[0])
            poses_c2w = output["camera_poses"][0].float().cpu()
        elif model_name == "vggt":
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri

            output = model(images)
            extrinsics, _ = pose_encoding_to_extri_intri(
                output["pose_enc"].float(),
                image_size_hw=tuple(images.shape[-2:]),
                build_intrinsics=False,
            )
            poses_c2w = torch.from_numpy(invert_poses(extrinsics[0].float().cpu()))[:, :3]
        elif model_name == "da3":
            output = model.inference(
                _images_to_pil(images[0]),
                ref_view_strategy="first",
                process_res=max(images.shape[-2:]),
                process_res_method="upper_bound_resize",
            )
            if output.extrinsics is None:
                raise RuntimeError("DA3 pose estimation requires predicted extrinsics.")
            poses_c2w = torch.from_numpy(invert_poses(output.extrinsics))[:, :3]
        elif model_name == "streamvggt":
            poses_c2w = _infer_streamvggt_pose(model, images[0], total_budget=None)
        elif model_name == "stream3r":
            poses_c2w = _infer_stream3r_pose(model, images[0])
        else:
            raise ValueError(f"Unsupported pose model: {model_name}")

    return c2w_to_pose_rows(poses_c2w)
