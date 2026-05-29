<div align="center">
  <h1 align="center"><sup>ViGeo</sup></h1>
    <p>
        <strong>
        <a href="https://pkqbajng.github.io/" style="text-decoration: none; color: inherit;">Zhu Yu</a><sup>*</sup>,
        <a href="https://g-1nonly.github.io/" style="text-decoration: none; color: inherit;">Jingnan Gao</a><sup>*</sup>,
        <a href="https://rm-zhang.github.io/" style="text-decoration: none; color: inherit;">Runmin Zhang</a>,
        <a href="https://lingtengqiu.github.io/" style="text-decoration: none; color: inherit;">Linteng Qiu</a>,
        et al.
        </strong>
    </p>
    <p>
        <a href="https://pkqbajng.github.io/ViGeo/" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Homepage-ViGeo-blue?style=flat" alt="Homepage"></a>
        <a href="https://arxiv.org/abs/2605.30060" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat&logo=arxiv" alt="arXiv"></a>
        <a href="" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow?style=flat&logo=huggingface" alt="Model"></a>
</div>

ViGeo estimates scene geometry from either video clips or single-frame inputs, including depth, 3D points, normals, confidence, and camera poses for sequences. VideoLDCM supports depth completion for both videos and single images; in our paper, it is used as the data-refinement model to turn sparse or noisy depth observations into cleaner dense depth supervision.

ViGeo supports both `offline` and `online` inference. Use `offline` when the full input is available, `online` for streaming frame-by-frame inference, and `chunk` for long videos that should be processed in segments with cached context.

For training, please refer to the [train branch](https://github.com/aigc3d/ViGeo/tree/train).
For full benchmark evaluation, please refer to the [benchmark branch](https://github.com/aigc3d/ViGeo/tree/benchmark).

## To Do List

We have fixed several numerical errors in the paper and submitted an updated version to arXiv. Before the update is reflected on arXiv, please refer to assets/paper.pdf for the correct version.

- [ ] Release ViGeo
- [ ] Release Hugging Face demo
- [ ] Update pose benchmarks

## Installation

ViGeo uses Python 3.10. We use PyTorch 2.7.1 with CUDA 12.6 by default, following the training environment; other compatible PyTorch/CUDA versions should also work.

```bash
conda create -n vigeo python=3.10 -y
conda activate vigeo

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/aigc3d/ViGeo.git
cd ViGeo
pip install -r requirements.txt
pip install -e .
```

The lightweight inference package does not require `utils3d`. For the optional VideoLDCM data refinement demo, install the extra dependencies:

```bash
pip install xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_refine.txt
```

## Pretrained Models

| Model | Download | Description |
| --- | --- | --- |
| ViGeo | - | Main visual geometry model for depth, points, normals, poses, and confidence. |
| VideoLDCM | [LINK](https://huggingface.co/pkqbajng/VideoLDCM) | Data-refinement model for sparse-depth filtering, Poisson completion, and depth refinement. |

## Quick Start for ViGeo

Set `checkpoint_path` to the ViGeo checkpoint before loading the model.

Inputs are RGB tensors in `[0, 1]` with shape `[T, 3, H, W]` or `[B, T, 3, H, W]`.

```python
import torch

from vigeo import ViGeo
from utils import load_image_sequence

device = torch.device("cuda")
image_paths = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
images = load_image_sequence(image_paths).to(device)  # [T, 3, H, W], RGB in [0, 1]
checkpoint_path = "path/to/vigeo.pt"

model = ViGeo().to(device).eval()
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict.get("model", state_dict), strict=True)

with torch.inference_mode():
    output = model.infer(images, mode="offline")

depth = output["depth_pred"]      # [T, 1, H, W]
points = output["points_pred"]    # [T, H, W, 3]
normals = output["normal_pred"]   # [T, H, W, 3], inward normals
normals_out = -normals            # outward normals for visualization/evaluation
poses = output["pose_pred"]       # [T, 3, 4], camera-to-world
conf = output["conf_pred"]        # [T, 1, H, W]
```

For batched input `[B, T, 3, H, W]`, tensor outputs keep the leading batch dimension.

ViGeo uses a right-handed camera coordinate system with `(X, Y, Z) = (right, down, front)`. The raw `normal_pred` output follows the inward normal convention. The demo and normal benchmarks use outward normals for RGB normal-map visualization and evaluation; for example, a fronto-parallel wall facing the camera is visualized/evaluated with normal `(0, 0, 1)`. Please use `normals = -normal_pred` when outward normals are needed.

## Inference Modes

ViGeo provides `offline`, `chunk`, and `online` inference modes. `offline` processes the full input sequence at once and is preferred when the complete video or image set is available.

```python
output = model.infer(images, mode="offline")
```

`chunk` is designed for long-video inference. The caller can split a long sequence into segments and keep `kv_caches` between calls. The default chunk size is `16`.

```python
kv_caches = None
for image_chunk in images.split(16, dim=0):
    output = model.infer(
        image_chunk,
        mode="chunk",
        chunk_size=16,
        kv_caches=kv_caches,
    )
    kv_caches = output["kv_caches"]
```

`online` supports streaming inference, usually with one new frame per call.

```python
kv_caches = None
for image_chunk in images.split(1, dim=0):
    output = model.infer(
        image_chunk,
        mode="online",
        kv_caches=kv_caches,
    )
    kv_caches = output["kv_caches"]
```

## Quick Start for VideoLDCM

`videoldcm/` is kept as a package beside `vigeo/`. This demo loads RGB images and sparse depth maps, then `infer` runs MoGe, Poisson completion, and VideoLDCM refinement.

```python
import torch

from videoldcm import videoldcm
from utils import load_depth_sequence, load_image_sequence

image_paths = ["path/to/image_000.png", "path/to/image_001.png"]
sparse_depth_paths = ["path/to/sparse_depth_000.npy", "path/to/sparse_depth_001.npy"]
device = torch.device("cuda")

image = load_image_sequence(image_paths).to(device)          # [S, 3, H, W]
sparse_depth = load_depth_sequence(sparse_depth_paths).to(device)  # [S, 1, H, W]

completion_model = videoldcm.from_pretrained("pkqbajng/VideoLDCM").eval().to(device)

with torch.inference_mode():
    output = completion_model.infer(image=image, sparse_depth=sparse_depth)
    refined_depth = output["depth_pred"]  # [S, 1, H, W]
```

## VideoLDCM Data Refinement

For data refinement, you can expose the sparse-depth filtering and Poisson completion steps explicitly.

<details>
<summary>Show data refinement code</summary>

```python
import torch
import utils3d

from videoldcm import videoldcm
from videoldcm.poisson_completion import poisson_completion
from utils import (
    load_depth_sequence,
    load_image_sequence,
    load_intrinsic,
    multi_scale_filter_depth,
)

image_paths = ["path/to/image_000.png", "path/to/image_001.png"]
sparse_depth_paths = ["path/to/sparse_depth_000.npy", "path/to/sparse_depth_001.npy"]
device = torch.device("cuda")

image = load_image_sequence(image_paths).to(device)                # [S, 3, H, W]
sparse_depth = load_depth_sequence(sparse_depth_paths).to(device)  # [S, 1, H, W]

S, _, H, W = image.shape
intrinsic, focal = load_intrinsic("path/to/intrinsic.npy", H, W)
intrinsic = intrinsic.to(device)  # [3, 3]
focal = focal.to(device)          # scalar

completion_model = videoldcm.from_pretrained("pkqbajng/VideoLDCM").eval().to(device)

with torch.inference_mode():
    moge_out = completion_model.moge.infer(
        image,
        apply_mask=False,
        force_projection=True,
    )
    moge_mask = moge_out["mask"].unsqueeze(1)  # [S, 1, H, W]
    mono_depth = moge_out["depth"].unsqueeze(1).float().masked_fill(~moge_mask, 0.0)

    points_gt = utils3d.pt.depth_map_to_point_map(
        sparse_depth.squeeze(1),
        intrinsics=intrinsic.unsqueeze(0).expand(S, -1, -1),
    )  # [S, H, W, 3]

    # Removes wrong sparse-depth points with multi-scale geometry consistency.
    filtered_mask = multi_scale_filter_depth(
        moge_out["points"],
        points_gt,
        moge_mask & (sparse_depth > 0.0001),
        focal=focal.expand(S),
    )  # [S, H, W]
    prior = sparse_depth * filtered_mask.unsqueeze(1).float()

    coarse_depth = poisson_completion(
        sparse=prior,
        mono_depth=mono_depth,
        confidence=moge_mask.float(),
        num_scales=5,
        max_iter_per_scale=[5000, 2000, 1000, 500, 250],
        max_resolution_ratio=0.5,
    )  # [S, 1, H, W]
    output = completion_model.infer_without_poisson(
        image=image,
        prior=prior,
        coarse_depth=coarse_depth,
        mask=moge_mask,
    )
    refined_depth = output["depth_pred"]  # [S, 1, H, W]
```

</details>

## License

ViGeo is licensed under the Apache License, Version 2.0. See `LICENSE` for details.

## Bibtex
```bibtex
@article{yu2026vigeo,
  title={Towards Consistent Video Geometry Estimation},
  author={Yu, Zhu and Gao, Jingnan and Zhang, Runmin and Qiu, Lingteng and Zhao, Zhengyi and Peng, Rui
          and Yan, Yichao and Qiu, Kejie and Zhu, Siyu and Dong, Zilong and Cao, Si-Yuan and Shen, Hui-Liang},
  journal={arXiv:2605.30060},
  year={2026}
}
```

