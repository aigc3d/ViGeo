# Installation

This project uses Python 3.10 and PyTorch CUDA 12.6 wheels.

> **Note**
> We use PyTorch 2.7.1 with CUDA 12.6 by default. Other compatible PyTorch/CUDA versions should also work.

## Create Environment

```bash
conda create -n vigeo python=3.10 -y
conda activate vigeo
```

## Install PyTorch

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

## Install xFormers

```bash
pip install xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu126
```

## Install utils3d

```bash
git clone https://github.com/EasternJournalist/utils3d.git
cd utils3d
git checkout 3fab839f0be9931dac7c8488eb0e1600c236e183
pip install .
cd ..
```

## Install Dependencies

`mmcv` is not required by the current ViGeo training and inference code, so it is intentionally skipped.

Install the remaining dependencies with:

```bash
pip install -r docs/requirements.txt
```
