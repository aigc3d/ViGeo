# ViGeo Training

This branch contains the training code for ViGeo and VideoLDCM. Run all commands from the repository root.

Preprocessing instructions are intentionally left out for now and will be added later.

## Installation

Create the environment and install the dependencies:

```bash
git clone -b train https://github.com/aigc3d/ViGeo.git
cd ViGeo

conda create -n vigeo python=3.10 -y
conda activate vigeo

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/EasternJournalist/utils3d.git
cd utils3d
git checkout 3fab839f0be9931dac7c8488eb0e1600c236e183
pip install .
cd ..

pip install -r docs/requirements.txt
```

## Directory Layout

- `configs/video_depth/`: ViGeo training configs.
- `configs/depth_completion/`: VideoLDCM training configs.
- `src/model/`: ViGeo model definitions.
- `src/depth_completion_model/`: VideoLDCM model definitions.
- `preprocess/`: dataset preprocessing and index-generation scripts.
- `logs/`: default training output directory used by the commands below.
- `ckpts/`: expected location for pretrained checkpoints referenced by configs.
- `data/`: expected dataset root in the provided configs.
- `train_test_split/`: expected split-file root in the provided configs.

## Training

Set `NPROC_PER_NODE` to the number of GPUs used on each node. Use `--resume_from path/to/last.ckpt` only when resuming an interrupted run. For stage-to-stage initialization, the configs already point to the previous stage checkpoint through `weight_cfg`.

```bash
export NPROC_PER_NODE=4
export NNODES=1

run_train() {
  local config_path=$1
  local log_folder=$2
  local port=$3

  python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --master_port="${port}" \
    main.py \
    --num_nodes "${NNODES}" \
    --config_path "${config_path}" \
    --log_folder "${log_folder}" \
    --log_interval 50 \
    --check_val_every_n_epoch 1 \
    --seed 1234
}

# ViGeo stage 1
run_train configs/video_depth/vigeo_stage1_224x224.py logs/vigeo_stage1_224x224 2243

# ViGeo stage 2
run_train configs/video_depth/vigeo_stage2_336x336_518x518.py logs/vigeo_stage2_336x336_518x518 2244

# ViGeo stage 3
run_train configs/video_depth/vigeo_stage3_336x336_518x518_normal.py logs/vigeo_stage3_336x336_518x518_normal 2245

# ViGeo stage 4
run_train configs/video_depth/vigeo_stage4_336x336_518x518_conf.py logs/vigeo_stage4_336x336_518x518_conf 2246

# VideoLDCM stage 1
run_train configs/depth_completion/videoldcm_stage1.py logs/videoldcm_stage1 2251

# VideoLDCM stage 2
run_train configs/depth_completion/videoldcm_stage2.py logs/videoldcm_stage2 2252
```

## Data Preparation

Dataset preprocessing and split generation live under `preprocess/`. The configs currently expect processed datasets under `data/` and split files under `train_test_split/`. Detailed preprocessing commands will be added later.
