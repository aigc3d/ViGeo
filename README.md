<div align="center">
  <h1 align="center"><sup>ViGeo</sup></h1>
    <p>
        <strong>
        <a href="https://pkqbajng.github.io/" style="text-decoration: none; color: inherit;">Zhu Yu</a><sup>*</sup>,
        <a href="https://g-1nonly.github.io/" style="text-decoration: none; color: inherit;">Jingnan Gao</a><sup>*</sup>,
        <a href="https://rm-zhang.github.io/" style="text-decoration: none; color: inherit;">Runmin Zhang</a>,
        <a href="https://lingtengqiu.github.io/" style="text-decoration: none; color: inherit;">Linteng Qiu</a>,
        <a href="https://scholar.google.com/citations?user=iuJ1buQAAAAJ&amp;hl=en" style="text-decoration: none; color: inherit;">Zhengyi Zhao</a>,
        <a href="https://prstrive.github.io/" style="text-decoration: none; color: inherit;">Rui Peng</a>,
        <a href="https://daodaofr.github.io/" style="text-decoration: none; color: inherit;">Yichao Yan</a>,
        <a href="https://sites.google.com/site/kejieqiujack/home" style="text-decoration: none; color: inherit;">Kejie Qiu</a>,
        <a href="https://sites.google.com/site/zhusiyucs/home" style="text-decoration: none; color: inherit;">Siyu Zhu</a>,
        <a href="https://scholar.google.com/citations?user=GHOQKCwAAAAJ&amp;hl=en" style="text-decoration: none; color: inherit;">Zilong Dong</a>,
        <a href="https://scholar.google.com/citations?user=G35eh30AAAAJ&amp;hl=zh-CN" style="text-decoration: none; color: inherit;">Si-Yuan Cao</a>,
        <a href="https://scholar.google.com/citations?user=qfAg9nUAAAAJ&amp;hl=en" style="text-decoration: none; color: inherit;">Hui-Liang Shen</a>
        </strong>
    </p>
    <p>
        <a href="https://pkqbajng.github.io/ViGeo/" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Homepage-ViGeo-blue?style=flat" alt="Homepage"></a>
        <a href="https://arxiv.org/abs/2605.30060" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat&logo=arxiv" alt="arXiv"></a>
        <a href="https://huggingface.co/pkqbajng/ViGeo" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow?style=flat&logo=huggingface" alt="Model"></a>
        <a href="https://www.modelscope.cn/models/RunminZhang/ViGeo" style="text-decoration: none; margin: 0 8px;"><img src="https://img.shields.io/badge/Model-ModelScope-blueviolet?style=flat&logo=modelscope" alt="ModelScope"></a>
</div>

ViGeo estimates scene geometry from either video clips or single-frame inputs, including depth, 3D points, normals, confidence, and camera poses for sequences. VideoLDCM supports depth completion for both videos and single images; in our paper, it is used as the data-refinement model to turn sparse or noisy depth observations into cleaner dense depth supervision.

ViGeo supports both `offline` and `online` inference. Use `offline` when the full input is available, `online` for streaming frame-by-frame inference, and `chunk` for long videos that should be processed in segments with cached context.

For training, please refer to the [train branch](https://github.com/aigc3d/ViGeo/tree/train).
For full benchmark evaluation, please refer to the [benchmark branch](https://github.com/aigc3d/ViGeo/tree/benchmark).

## News

- **[2026-10-15]** Updated the **Poisson reconstruction** stage used to construct VideoLDCM's coarse depth prior, adding robust sparse-observation weighting and edge-aware monocular guidance. The update improves both RMSE and AbsRel on the KITTI selected validation set, ScanNet++, and synthetic cases with analytical ground truth. The updated solver also runs **4.34–6.71× faster** across the tested resolutions on NVIDIA H20-3e.

| Resolution (W × H) | Inference time (ms) | Speedup |
| --- | ---: | ---: |
| 256 × 192 | 27.23 → 6.27 | **4.34×** |
| 512 × 384 | 41.07 → 9.21 | **4.46×** |
| 518 × 345 | 62.48 → 10.88 | **5.74×** |
| 518 × 388 | 54.11 → 10.32 | **5.24×** |
| 1216 × 352 | 176.30 → 26.29 | **6.71×** |

| Dataset | RMSE (m) | AbsRel (%) |
| --- | ---: | ---: |
| KITTI | 2.294 → **2.205** | 3.91 → **3.65** |
| ScanNet++ | 0.108 → **0.089** | 2.80 → **1.83** |
| Synthetic | 0.372 → **0.274** | 4.13 → **2.66** |

<details>
<summary>Method and benchmark details</summary>

Building on Eq. (4) in the [paper](assets/paper.pdf), we retain the paper's
notation: $\mathbf{D}^{\mathrm{prior}}$ denotes the dense coarse depth prior,
$\mathbf{D}^{\mathrm{mono}}$ the initial monocular relative depth prediction,
$\tilde{\mathbf{D}}$ the filtered sparse depth, and
$\mathcal{M}^{\mathrm{valid}}$ the set of valid sparse observations.
For the implementation below, let
$\bar{\mathbf{D}}^{\mathrm{mono}}=a\mathbf{D}^{\mathrm{mono}}+b$ denote the
affine-aligned monocular prediction. At each pyramid level, the solver minimizes
the following robust objective:

$$
\begin{aligned}
\min_{\mathbf{D}^{\mathrm{prior}}>0}
&\Biggl[\sum_{(p,q)\in\mathcal{E}} w_{pq}
\left(\log\mathbf{D}^{\mathrm{prior}}_q-\log\mathbf{D}^{\mathrm{prior}}_p-g_{pq}\right)^2
\\[6pt]
&\quad+\lambda\sum_{p\in\mathcal{M}^{\mathrm{valid}}}\rho_p
\left(\log\mathbf{D}^{\mathrm{prior}}_p-\log\tilde{\mathbf{D}}_p\right)^2\Biggr],
\end{aligned}
$$

Here, $\mathcal{E}$ is the set of horizontally and vertically adjacent pixel
pairs, and $g_{pq}$ is the clipped log-depth difference induced by the aligned
monocular prediction:

$$
g_{pq}=\operatorname{clip}\!\left(
\log\bar{\mathbf{D}}^{\mathrm{mono}}_q-\log\bar{\mathbf{D}}^{\mathrm{mono}}_p,-3,3
\right).
$$

Unlike the compact formulation in Eq. (4), the updated implementation evaluates
the sparse-data term in log-depth space. It also introduces the following
edge-aware weights:

$$
\begin{aligned}
w_{pq}=\max\Bigl(&10^{-4},\;\min(c_p,c_q)\max\Bigl(0.01,\\[6pt]
&\exp\!\Bigl[-\dfrac{
|\log\bar{\mathbf{D}}^{\mathrm{mono}}_q-\log\bar{\mathbf{D}}^{\mathrm{mono}}_p|
}{0.01}\Bigr]\Bigr)\Bigr),
\end{aligned}
$$

Here, $c$ denotes the optional confidence and defaults to one when confidence is
not provided. For directional confidence, the horizontal or vertical channel is
used for the corresponding edge. Consequently, large monocular depth
discontinuities weaken the coupling across likely object boundaries.

For each usable sparse observation, define
$r_p=\log\tilde{\mathbf{D}}_p-\log\bar{\mathbf{D}}^{\mathrm{mono}}_p$.
An adaptive local window estimates the neighboring correction mean $\mu_p$ and
standard deviation $\sigma_p$, excluding the center observation itself. The
observation reliability is then

$$
\rho_p=\max\!\left(0.05,\;
\dfrac{1}{1+\left(\dfrac{r_p-\mu_p}{\max(0.15,2\sigma_p)}\right)^2}\right).
$$

The window size adapts to the per-image sparse-observation density and targets
approximately 12 usable observations. If fewer than three usable neighbors are
available, the reliability remains one. These reliability weights are computed
once, without a learned module or an outer robust-optimization loop. At coarser
pyramid levels, sparse depths are pooled using the reliability weights, and
reliability is averaged over occupied pixels. A conjugate-gradient solver then
proceeds from coarse to fine before exponentiating the log-depth solution and
resizing it to the input resolution.

**Acknowledgment.** We thank the authors of
[MetricScenes](https://github.com/MetricScenes/MetricScenes), whose edge-aware
Poisson completion implementation inspired our edge-weighting design.

</details>

- **[2026-10-14]** Accelerated VideoLDCM's multi-scale depth filtering by **19.14×**, based on mean per-image latency across 120 images from Waymo, ARKitScenes, and ScanNet++ on NVIDIA H20-3e.

| Resolution | Inference time (s) | Speedup | Avg. mask diff. |
| --- | ---: | ---: | ---: |
| 518 × 388 | 3.595 → 0.188 | **19.14×** | **0.175px / image** |

## To Do List
- [x] Release ViGeo

  ViGeo 1.0 reproduces the paper results. ViGeo 1.1 additionally provides a mask head for separating valid scene regions from sky or invalid regions.

- [ ] Release Hugging Face demo
- [x] Update pose benchmarks
- [ ] Update ViPE code with VideoLDCM

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

Poisson completion can also be imported on its own, without the refinement model dependencies. The standard installation includes Ninja, and available CUDA acceleration is selected automatically. Native reductions build on first use when a CUDA compiler is available; otherwise the solver keeps the Triton/CUDA Graph path, or uses PyTorch if Triton is unavailable.

## Pretrained Models

| Model | Download | Description |
| :---: | :---: | --- |
| ViGeo 1.0 | [HuggingFace](https://huggingface.co/pkqbajng/ViGeo)<br>[ModelScope](https://www.modelscope.cn/models/RunminZhang/ViGeo) | Visual geometry model for depth, points, normals, poses, and confidence. |
| ViGeo 1.1 | [HuggingFace](https://huggingface.co/pkqbajng/ViGeo1.1) | Updated model with an additional valid-region mask head. |
| VideoLDCM | [HuggingFace](https://huggingface.co/pkqbajng/VideoLDCM)<br>[ModelScope](https://www.modelscope.cn/models/RunminZhang/VideoLDCM) | Data-refinement model for sparse-depth filtering, Poisson completion, and depth refinement. |

> **Scale alignment note.** The released ViGeo checkpoints follow the paper training setup, where GT camera translation was not scale-aligned before constructing ray supervision. Its possible impact on pose accuracy is currently only a hypothesis, which we plan to verify when resources permit; see the [detailed training note](https://github.com/aigc3d/ViGeo/tree/train#ray-loss-scale-alignment).

## Quick Start for ViGeo

Inputs are RGB tensors in `[0, 1]` with shape `[T, 3, H, W]` or `[B, T, 3, H, W]`.

```python
import torch

from vigeo import ViGeo
from utils import load_image_sequence

device = torch.device("cuda")
image_paths = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
images = load_image_sequence(image_paths).to(device)  # [T, 3, H, W], RGB in [0, 1]
model = ViGeo.from_pretrained("pkqbajng/ViGeo1.1").to(device).eval()

with torch.inference_mode():
    output = model.infer(images, mode="offline")

depth = output["depth_pred"]      # [T, 1, H, W]
points = output["points_pred"]    # [T, H, W, 3]
normals = output["normal_pred"]   # [T, H, W, 3], inward normals
normals_out = -normals            # outward normals for visualization/evaluation
poses = output["pose_pred"]       # [T, 3, 4], camera-to-world
conf = output["conf_pred"]        # [T, 1, H, W]
mask = output["mask_pred"]        # [T, 1, H, W], or None for ViGeo 1.0
```

`from_pretrained` detects the checkpoint version automatically. For batched input `[B, T, 3, H, W]`, tensor outputs keep the leading batch dimension.

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

completion_model = videoldcm.from_pretrained("pkqbajng/VideoLDCM").to(device).eval()

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

completion_model = videoldcm.from_pretrained("pkqbajng/VideoLDCM").to(device).eval()

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

    valid_mask = (moge_mask & (sparse_depth > 0.0001)).squeeze(1) # [S, H, W]
    # Removes wrong sparse-depth points with multi-scale geometry consistency.
    filtered_mask = multi_scale_filter_depth(
        moge_out["points"], # [S, H, W, 3]
        points_gt, # [S, H, W, 3]
        valid_mask, # [S, H, W]
        focal=focal.expand(S), # [S]
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

If you find the depth refinement component useful for your project, please also consider citing:

```bibtex
@inproceedings{yu2026ldcm,
  title={Large Depth Completion Model from Sparse Observations},
  author={Yu, Zhu and Zhao, Zhengyi and Zhang, Runmin and Qiu, Lingteng and Qiu, Kejie
          and He, Yisheng and Zhu, Siyu and Dong, Zilong and Cao, Si-Yuan and Shen, Hui-Liang},
  booktitle={ICLR},
  year={2026}
}
```
