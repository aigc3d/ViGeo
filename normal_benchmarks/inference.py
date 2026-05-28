import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in [str(PROJECT_ROOT), str(CURRENT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from metric_func import as_normal_stack


PIL_NORMAL_MODELS = {'normalcrafter', 'stablenormal', 'lotus'}


def read_rgb_image(img_path, model_name, dataset_name=None):
    if model_name == 'dsine':
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        if dataset_name == 'nyuv2':
            img = img[45:472, 43:608]

        return img

    if model_name in PIL_NORMAL_MODELS:
        img = Image.open(img_path).convert("RGB")

        if dataset_name == 'nyuv2':
            img = img.crop((43, 45, 608, 472))

        return img

    raise ValueError(f"Model {model_name} not supported for reading")


def encoded_normal_to_tensor(normal):
    if isinstance(normal, Image.Image):
        normal = np.asarray(normal.convert("RGB")).copy()

    normal = as_normal_stack(normal).float()
    if normal.max() > 2.0:
        normal = normal / 255.0
    if normal.min() >= 0.0 and normal.max() <= 1.0:
        normal = normal * 2.0 - 1.0
    return normal


def _lotus_task_embedding(device):
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    return torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)


def _lotus_rgb_tensor(img, device):
    img_np = np.asarray(img).astype(np.float32)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return (tensor / 127.5 - 1.0).to(device)


def _run_lotus_frame(model, img):
    device = model.device
    autocast_ctx = (
        torch.autocast(device.type)
        if device.type not in {'cpu', 'mps'}
        else nullcontext()
    )
    with autocast_ctx:
        pred = model(
            rgb_in=_lotus_rgb_tensor(img, device),
            prompt='',
            num_inference_steps=1,
            output_type='pt',
            timesteps=[getattr(model, 'lotus_timestep', 999)],
            task_emb=_lotus_task_embedding(device),
            processing_res=getattr(model, 'lotus_processing_res', 0),
            match_input_res=True,
        ).images[0]
    return encoded_normal_to_tensor(pred)


def run_inference(model, model_name, img_files, dataset_name=None):
    with torch.inference_mode():
        if model_name == 'dsine':
            normal_preds = [model.infer_cv2(read_rgb_image(p, model_name, dataset_name))[0] for p in img_files]
            normal_pred = torch.stack(normal_preds).cuda()

        elif model_name == 'normalcrafter':
            frames = [read_rgb_image(p, model_name, dataset_name) for p in img_files]
            orig_w, orig_h = frames[0].size
            if len(frames) == 1:
                frames = [frames[0]] * 14

            res = model(frames, decode_chunk_size=7, time_step_size=10, window_size=14).frames[0]
            normal_pred = torch.from_numpy(res).permute(0, 3, 1, 2)
            normal_pred[:, 0, ...] *= -1
            if normal_pred.shape[-2:] != (orig_h, orig_w):
                normal_pred = F.interpolate(normal_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        elif model_name == 'stablenormal':
            normal_preds = [
                encoded_normal_to_tensor(model(read_rgb_image(p, model_name, dataset_name)))
                for p in img_files
            ]
            normal_pred = torch.cat(normal_preds, dim=0)

        elif model_name == 'lotus':
            normal_preds = [
                _run_lotus_frame(model, read_rgb_image(p, model_name, dataset_name))
                for p in img_files
            ]
            normal_pred = torch.cat(normal_preds, dim=0)

        else:
            raise ValueError(f"Model '{model_name}' not implemented.")

    if model_name == 'normalcrafter' and len(img_files) == 1:
        normal_pred = normal_pred[0:1]
    return as_normal_stack(normal_pred)
