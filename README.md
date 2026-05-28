# ViGeo Evaluation

This repository contains evaluation code for ViGeo and a set of third-party visual geometry baselines.

## Overview

The evaluation covers four tasks:

- Monocular depth estimation
- Video depth estimation
- Point-map estimation
- Surface normal estimation

The entry scripts are under `scripts/` and are intended to be launched from the repository root. They write summary CSV files directly to the repository root.

## Repository Layout

```text
.
├── eval.py                         # ViGeo evaluation entry point
├── depth_benchmarks/eval.py         # Depth / point-map baseline evaluation
├── normal_benchmarks/eval.py        # Surface normal baseline evaluation
├── scripts/
│   ├── eval_vigeo.sh
│   ├── eval_depth_benchmarks.sh
│   └── eval_normal_benchmarks.sh
├── benchmark_defs.py                # Task, dataset, and metric definitions
├── dataset_io.py                    # Dataset readers and file layout helpers
├── metric_func.py                   # Shared metric implementations
└── THIRD_PARTY_NOTICES.md
```

## Dataset Preparation

Put evaluation datasets under one root directory, for example `benchmark_datasets/`, or pass another path with `--data-root`.

Download links:

- [bonn.zip](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/bonn.zip)
- [hammer.zip](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/hammer.zip)
- [kitti.zip](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/kitti.zip)
- [nyuv2.zip](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/nyuv2.zip)
- [sintel.zip](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/sintel.zip)

Example:

```bash
mkdir -p benchmark_datasets

wget -P benchmark_datasets https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/bonn.zip
wget -P benchmark_datasets https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/hammer.zip
wget -P benchmark_datasets https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/kitti.zip
wget -P benchmark_datasets https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/nyuv2.zip
wget -P benchmark_datasets https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/ViGeo_Benchmark/sintel.zip

unzip -q benchmark_datasets/bonn.zip -d benchmark_datasets
unzip -q benchmark_datasets/hammer.zip -d benchmark_datasets
unzip -q benchmark_datasets/kitti.zip -d benchmark_datasets
unzip -q benchmark_datasets/nyuv2.zip -d benchmark_datasets
unzip -q benchmark_datasets/sintel.zip -d benchmark_datasets
```

Expected dataset names are:

- `sintel`
- `bonn`
- `kitti`
- `nyuv2`
- `hammer`

Sintel evaluation uses the `clean` pass.

Expected high-level layout:

```text
benchmark_datasets/
├── sintel/
│   ├── clean/<scene>/*.png
│   ├── depth/<scene>/*.dpt
│   ├── normal/<scene>/*.png
│   └── camdata_left/<scene>/*.cam
├── bonn/
│   └── <scene>/{rgb,depth}/
├── kitti/
│   ├── image/<sequence>/image_02/data/
│   ├── depth/<sequence>/proj_depth/groundtruth/image_02/
│   └── calib/<date>/calib_cam_to_cam.txt
├── nyuv2/test/
│   ├── <id>_img.png
│   └── <id>_normal.png
└── hammer/
    └── <scene>/{rgb,_gt,normal}/
        # or <scene>/polarization/{rgb,_gt,normal}/
```

Normal maps for Sintel and Hammer can be generated with the helper scripts in `preprocess/` when needed.

## Checkpoints

ViGeo weights are passed explicitly with `--checkpoint-path`.

Third-party baseline weights are not included in this repository. Download them from the original projects or model hubs and follow the corresponding upstream licenses. Local checkpoint directories are only load locations:

- `depth_benchmarks/checkpoints/`
- `normal_benchmarks/checkpoints/`

Some baselines load from the default Hugging Face cache. If a model is already cached locally, the evaluation can run without downloading it again; otherwise the loader will try to download it.

## ViGeo Evaluation

Run all configured ViGeo evaluations:

```bash
bash scripts/eval_vigeo.sh \
    --data-root benchmark_datasets \
    --checkpoint-path path/to/vigeo.pt
```

Defaults:

- `--chunk-size 16`, which can be changed from the command line
- fp16 enabled

Optional flags:

```bash
bash scripts/eval_vigeo.sh \
    --data-root benchmark_datasets \
    --checkpoint-path path/to/vigeo.pt \
    --chunk-size 16 \
    --no-fp16
```

The script evaluates:

- `video_depth` on Sintel, Bonn, and KITTI with `offline` and `online` modes
- `pointmap` on Sintel, Bonn, and KITTI with `offline` and `online` modes
- `mono_depth` on Sintel, Bonn, and KITTI with `offline` mode
- `normal` on Hammer, Sintel, and NYUv2 with `offline` mode
- long video-depth sequences with `chunk` mode

Output:

```text
eval_results_vigeo_summary.csv
```

## Depth and Point-Map Baselines

Run all configured depth and point-map baselines:

```bash
bash scripts/eval_depth_benchmarks.sh --data-root benchmark_datasets
```

Evaluated task groups:

- Video depth: DA3, VGGT, STream3R, Video Depth Anything, GeometryCrafter, StreamVGGT, DepthCrafter, Pi3
- Monocular depth: Video Depth Anything, DepthCrafter, VGGT, VGGT-Omega, Pi3, DA3, FlashDepth
- Long video depth: Video Depth Anything, DepthCrafter, GeometryCrafter, InfiniteVGGT
- Point map: GeometryCrafter, VGGT, VGGT-Omega, Pi3, DA3, StreamVGGT, STream3R

Outputs:

```text
eval_results_video_depth_summary.csv
eval_results_mono_depth_summary.csv
eval_results_pointmap_summary.csv
```

## Surface Normal Baselines

Run all configured normal baselines:

```bash
bash scripts/eval_normal_benchmarks.sh --data-root benchmark_datasets
```

Evaluated baselines:

- DSINE
- NormalCrafter
- StableNormal
- Lotus

Output:

```text
eval_results_normal_summary.csv
```

## Metrics

Depth and point-map tasks report:

- `absrel`
- `d1`

Normal estimation reports:

- `mean`
- `median`
- `a3`

CSV rows use:

```text
task,dataset,benchmark,<metrics...>
```

Values are written incrementally after each dataset finishes.

## Licenses and Third-Party Code

ViGeo benchmark code is licensed under the Apache License, Version 2.0. See
`LICENSE` for this project and `THIRD_PARTY_NOTICES.md` for third-party
baseline code and model component notices.

Third-party baseline weights are not distributed here.
