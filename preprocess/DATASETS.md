# Dataset Preparation

The training datasets used by ViGeo and VideoLDCM are listed below. Download
each dataset from its official release page and follow the corresponding
license and access requirements.

## Training Datasets

- [Hypersim](https://github.com/apple/ml-hypersim)
- [LightwheelOcc](https://huggingface.co/datasets/OpenDriveLab/LightwheelOcc)
- [TartanAir](https://huggingface.co/datasets/theairlabcmu/tartanair/tree/main)
- [TartanGround](https://huggingface.co/datasets/theairlabcmu/TartanGround)
- [GTA-SfM](https://github.com/HKUST-Aerial-Robotics/Flow-Motion-Depth/tree/master/extracted_dataset)
- [PointOdyssey](https://huggingface.co/datasets/aharley/pointodyssey)
- [BEDLAM](https://huggingface.co/datasets/Intelligent-Systems/BEDLAM/tree/main)
- [Dynamic Replica](https://huggingface.co/datasets/ZhengGuangze/DynamicReplica)
- [MatrixCity](https://huggingface.co/datasets/BoDai/MatrixCity/tree/main)
- [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
- [OmniWorld](https://huggingface.co/datasets/InternRobotics/OmniWorld)
- [SYNTHIA](https://synthia-dataset.net/downloads/)
- [OmniObject3D](https://omniobject3d.github.io/)
- [Aria Synthetic Environments](https://www.projectaria.com/datasets/ase/)
- [Spring](https://spring-benchmark.org/)
- [TransPhy3D](https://huggingface.co/datasets/Daniellesry/TransPhy3D)
- [CarlaOcc](https://huggingface.co/datasets/fengyi233/CarlaOcc)
- [WildRGB-D](https://huggingface.co/datasets/HarrisonPENG/wildrgbd)
- [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)
- [ARKitScenes](https://github.com/apple/ARKitScenes)
- [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/)
- [DL3DV-10K](https://huggingface.co/datasets/zhangify/CUT3R_release/tree/main)
- [BlendedMVS](https://github.com/yoyo000/blendedmvs)

Some datasets require registration, license acceptance, or an access request
before their files can be downloaded.

## Preprocessing

Run all commands from the repository root. The final dataset layout is
`data/<dataset-name>`. For converters that need separate input and output
directories, extract the original files to `data/raw/<dataset-name>`; datasets
processed in place should be extracted directly to their final directory.

Set the number of CPU workers and create the common directories first:

```bash
export NUM_WORKERS=32
mkdir -p data/raw
```

Then run the preprocessing required by each dataset:

```bash
# Hypersim: raw data -> processed RGB, depth, cameras, and normals
python preprocess/preprocess_datasets/preprocess_hypersim.py \
  --hypersim_dir data/raw/hypersim --output_dir data/hypersim
python preprocess/preprocess_datasets/preprocess_hypersim_normal.py \
  --data_path data/hypersim --num_workers "${NUM_WORKERS}"

# LightwheelOcc: generate normals in place
python preprocess/preprocess_datasets/preprocess_lightwheelocc_normal.py \
  --data_path data/lightwheelocc --num_workers "${NUM_WORKERS}"

# TartanAir: generate normals in place
python preprocess/preprocess_datasets/preprocess_tartanair_normal.py \
  --data_path data/tartanair --num_workers "${NUM_WORKERS}"

# TartanGround: generate normals alongside the downloaded data
python preprocess/preprocess_datasets/preprocess_tartanground_normal.py \
  --data_path data/tartanground --nas_out data/tartanground \
  --num_workers "${NUM_WORKERS}"

# GTA-SfM: unpack HDF5 sequences, then generate normals
python preprocess/preprocess_datasets/preprocess_gtasfm.py \
  --data_path data/raw/gtasfm --output_path data/gtasfm
python preprocess/preprocess_datasets/preprocess_gtasfm_normal.py \
  --data_path data/gtasfm --num_workers "${NUM_WORKERS}"

# PointOdyssey: generate normals in place
python preprocess/preprocess_datasets/preprocess_pointodyssey_normal.py \
  --data_path data/pointodyssey --num_workers "${NUM_WORKERS}"

# BEDLAM: generate camera parameters and normals in place
python preprocess/preprocess_datasets/preprocess_bedlam.py \
  --data_path data/bedlam
python preprocess/preprocess_datasets/preprocess_bedlam_normal.py \
  --data_path data/bedlam --num_workers "${NUM_WORKERS}"

# Dynamic Replica: generate camera parameters and normals in place
python preprocess/preprocess_datasets/preprocess_dynamic_replica.py \
  --data_path data/dynamic_replica
python preprocess/preprocess_datasets/preprocess_dynamic_replica_normal.py \
  --data_path data/dynamic_replica --num_workers "${NUM_WORKERS}"

# MatrixCity: flatten sequences, then generate normals
python preprocess/preprocess_datasets/preprocess_matrixcity.py \
  --data_path data/raw/matrixcity --output_path data/matrixcity
python preprocess/preprocess_datasets/preprocess_matrixcity_normal.py \
  --data_path data/matrixcity --num_workers "${NUM_WORKERS}"

# MVS-Synth: convert images, depth, and cameras, then generate normals
python preprocess/preprocess_datasets/preprocess_mvssynth.py \
  --root_dir data/raw/mvssynth --out_dir data/mvssynth \
  --num_workers "${NUM_WORKERS}"
python preprocess/preprocess_datasets/preprocess_mvssynth_normal.py \
  --data_path data/mvssynth --num_workers "${NUM_WORKERS}"

# OmniWorld: generate normals in place
python preprocess/preprocess_datasets/preprocess_omniworld_normal.py \
  --data_path data/omniworld --num_workers "${NUM_WORKERS}"

# SYNTHIA: reorganize sequences, then generate normals
python preprocess/preprocess_datasets/preprocess_synthia.py \
  --data_path data/raw/synthia --output_path data/synthia
python preprocess/preprocess_datasets/preprocess_synthia_normal.py \
  --data_path data/synthia --num_workers "${NUM_WORKERS}"

# OmniObject3D: generate camera parameters and normals in place
python preprocess/preprocess_datasets/preprocess_omniobject.py \
  --data_path data/omniobject --output_path data/omniobject
python preprocess/preprocess_datasets/preprocess_omniobject_normal.py \
  --data_path data/omniobject --num_workers "${NUM_WORKERS}"

# Aria Synthetic Environments: convert the release, then generate normals
python preprocess/preprocess_datasets/preprocess_ase.py \
  --data_path data/raw/ase --output_path data/ase \
  --vignette_mask preprocess/preprocess_datasets/vignette.png \
  --num_workers "${NUM_WORKERS}"
python preprocess/preprocess_datasets/preprocess_ase_normal.py \
  --data_path data/ase --num_workers "${NUM_WORKERS}"

# Spring: copy the required modalities, then generate normals
python preprocess/preprocess_datasets/preprocess_spring.py \
  --data_path data/raw/spring --output_path data/spring
python preprocess/preprocess_datasets/preprocess_spring_normal.py \
  --data_path data/spring --num_workers "${NUM_WORKERS}"

# TransPhy3D already provides RGB, depth, normals, and camera metadata.
# Extract the release directly to data/transphy3d.

# CarlaOcc: generate camera parameters in place; normals are included.
python preprocess/preprocess_datasets/preprocess_carlaocc.py \
  --data_root data/carlaocc

# WildRGB-D is already in the expected layout.
# Extract the release directly to data/wildrgbd.

# Waymo uses the prepared release linked above.
# Extract it directly to data/waymo.

# ARKitScenes uses the prepared RGB-D release directly. The low- and
# high-resolution loaders share the same files.
# Extract it to data/arkitscenes, then create the second configured path:
ln -sfn arkitscenes data/arkitscenes_highres

# ScanNet++: render depth first, then undistort DSLR and iPhone streams.
python preprocess/preprocess_datasets/preprocess_scannetpp_render.py \
  --data_path data/scannetpp --device dslr
python preprocess/preprocess_datasets/preprocess_scannetpp_render.py \
  --data_path data/scannetpp --device iphone
python preprocess/preprocess_datasets/preprocess_scannetpp_dslr.py \
  --data_path data/scannetpp --num_workers "${NUM_WORKERS}"
python preprocess/preprocess_datasets/preprocess_scannetpp_iphone.py \
  --data_path data/scannetpp --num_workers "${NUM_WORKERS}"

# DL3DV uses the prepared CUT3R release linked above.
# Extract it directly to data/dl3dv.

# BlendedMVS uses the prepared RGB, depth, and camera files directly.
# Extract them to data/blendedmvs.
```

After preprocessing, every training dataset must remain under `data/`; the
temporary `data/raw/` inputs can be removed once their converted outputs have
been verified.

## Generate Indexes

Generate the training indexes after all preprocessing steps finish. Every
command reads from `data/<dataset-name>` and writes its index to
`train_test_split/`.

```bash
mkdir -p train_test_split

# Hypersim
python preprocess/generate_indexes/generate_indexes_hypersim.py \
  --data_path data/hypersim --output_path train_test_split

# LightwheelOcc
python preprocess/generate_indexes/generate_indexes_lightwheelocc.py \
  --data_path data/lightwheelocc --output_path train_test_split

# TartanAir
python preprocess/generate_indexes/generate_indexes_tartanair.py \
  --data_path data/tartanair --output_path train_test_split

# TartanGround
python preprocess/generate_indexes/generate_indexes_tartanground.py \
  --data_path data/tartanground --output_path train_test_split

# GTA-SfM
python preprocess/generate_indexes/generate_indexes_gtasfm.py \
  --data_path data/gtasfm --output_path train_test_split

# PointOdyssey
python preprocess/generate_indexes/generate_indexes_pointodyssey.py \
  --data_path data/pointodyssey --output_path train_test_split

# BEDLAM
python preprocess/generate_indexes/generate_indexes_bedlam.py \
  --data_path data/bedlam --output_path train_test_split

# Dynamic Replica
python preprocess/generate_indexes/generate_indexes_dynamic_replica.py \
  --data_path data/dynamic_replica --output_path train_test_split

# MatrixCity
python preprocess/generate_indexes/generate_indexes_matrixcity.py \
  --data_path data/matrixcity --output_path train_test_split

# MVS-Synth
python preprocess/generate_indexes/generate_indexes_mvssynth.py \
  --data_path data/mvssynth --output_path train_test_split

# OmniWorld
python preprocess/generate_indexes/generate_indexes_omniworld.py \
  --data_path data/omniworld --output_path train_test_split

# SYNTHIA
python preprocess/generate_indexes/generate_indexes_synthia.py \
  --data_path data/synthia --output_path train_test_split

# OmniObject3D
python preprocess/generate_indexes/generate_indexes_omniobject.py \
  --data_path data/omniobject --output_path train_test_split

# Aria Synthetic Environments
python preprocess/generate_indexes/generate_indexes_ase.py \
  --data_path data/ase --output_path train_test_split

# Spring
python preprocess/generate_indexes/generate_indexes_spring.py \
  --data_path data/spring --output_path train_test_split

# TransPhy3D
python preprocess/generate_indexes/generate_indexes_transphy3d.py \
  --data_path data/transphy3d --output_path train_test_split

# CarlaOcc
python preprocess/generate_indexes/generate_indexes_carlaocc.py \
  --data_path data/carlaocc --output_path train_test_split

# WildRGB-D
python preprocess/generate_indexes/generate_indexes_wildrgbd.py \
  --data_path data/wildrgbd --output_path train_test_split

# Waymo Open Dataset
python preprocess/generate_indexes/generate_indexes_waymo.py \
  --data_path data/waymo --output_path train_test_split

# ARKitScenes, low- and high-resolution indexes
python preprocess/generate_indexes/generate_indexes_arkitscenes.py \
  --data_path data/arkitscenes --output_path train_test_split
python preprocess/generate_indexes/generate_indexes_arkitscenes_highres.py \
  --data_path data/arkitscenes_highres --output_path train_test_split

# ScanNet++
python preprocess/generate_indexes/generate_indexes_scannetpp.py \
  --data_path data/scannetpp --output_path train_test_split

# DL3DV-10K
python preprocess/generate_indexes/generate_indexes_dl3dv.py \
  --data_path data/dl3dv --output_path train_test_split

# BlendedMVS
python preprocess/generate_indexes/generate_indexes_blendedmvs.py \
  --data_path data/blendedmvs --output_path train_test_split
```
