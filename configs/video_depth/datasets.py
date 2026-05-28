# Dataset config.
#
# Helper functions are used to avoid repeating nearly identical dataset and
# pipeline definitions. The final values are still plain Python dicts.

data_path = 'data/'
data_path2 = '/mnt/data/yuzhu/dataset/'
train_test_split_path = 'train_test_split/'

IMAGE_AUGMENTATIONS = ['jittering', 'jpeg_loss', 'blurring', 'shot_noise', 'dof']
TRAIN_TENSOR_KEYS = ['image', 'depth', 'intrinsic', 'sky_mask', 'pose', 'normal']
EVAL_TENSOR_KEYS = ['image', 'depth', 'intrinsic']
META_KEYS = ['instance', 'image_path', 'depth_path', 'data_name']

DEFAULT_CROP = dict(
    fov_range_absolute=[45, 135],
    center_augmentation=0.25,
    fov_range_relative=[0.75, 1.0],
)

WIDE_CROP = dict(
    fov_range_absolute=[30, 150],
    center_augmentation=0.5,
    fov_range_relative=[0.33, 1.0],
)


def train_pipeline(crop_cfg=DEFAULT_CROP):
    return [
        dict(type='perspective_crop', image_augmentation_list=IMAGE_AUGMENTATIONS, **crop_cfg),
        dict(type='convert_to_tensor', keys=TRAIN_TENSOR_KEYS, meta_keys=META_KEYS),
    ]


def eval_pipeline():
    return [
        dict(type='perspective_crop_eval', ensure_multiple_of=14),
        dict(type='convert_to_tensor', keys=EVAL_TENSOR_KEYS, meta_keys=META_KEYS),
    ]


def train_dataset(
    name,
    data_root,
    max_interval,
    max_depth,
    crop_cfg=DEFAULT_CROP,
    min_depth=0.0001,
    refine=False,
    split_ext='json',
    shuffle_sequence_prob=0.3,
):
    cfg = dict(
        type=f'{name}_data',
        data_path=data_root,
        data_name=name,
        train_test_split=f'{train_test_split_path}{name}_train.{split_ext}',
        max_interval=max_interval,
        min_depth=min_depth,
        max_depth=max_depth,
        shuffle_sequence_prob=shuffle_sequence_prob,
        pipeline=train_pipeline(crop_cfg),
    )
    if refine:
        cfg['refine'] = True
    return cfg


def val_dataset(name, data_root, start=0, end=None, min_depth=0.0001, max_depth=80.0):
    return dict(
        type=f'{name}_data',
        data_path=data_root,
        data_name=name,
        train_test_split=f'{train_test_split_path}{name}_eval.json',
        pipeline=eval_pipeline(),
        start=start,
        end=end,
        min_depth=min_depth,
        max_depth=max_depth,
        shuffle_sequence_prob=0.0,
    )


val_datasets = dict(
    kitti=val_dataset('kitti', data_path + 'kitti', start=0, end=110),
    bonn=val_dataset('bonn', data_path + 'bonn', start=30, end=140),
    sintel=val_dataset('sintel', data_path + 'sintel', start=0, end=None),
)


train_datasets = dict(
    tartanair=train_dataset('tartanair', data_path + 'tartanair/', 12, 80),
    gtasfm=train_dataset('gtasfm', data_path + 'gtasfm/', 4, 80),
    pointodyssey=train_dataset('pointodyssey', data_path + 'pointodyssey/', 30, 50, crop_cfg=WIDE_CROP),
    bedlam=train_dataset('bedlam', data_path + 'bedlam/', 30, 50, crop_cfg=WIDE_CROP),
    dynamic_replica=train_dataset('dynamic_replica', data_path + 'dynamic_replica/', 30, 50, crop_cfg=WIDE_CROP),
    hypersim=train_dataset('hypersim', data_path + 'hypersim', 1, 50, crop_cfg=WIDE_CROP),
    lightwheelocc=train_dataset('lightwheelocc', data_path + 'lightwheelocc/', 2, 80),
    mvssynth=train_dataset('mvssynth', data_path + 'mvssynth', 12, 80),
    matrixcity=train_dataset('matrixcity', data_path + 'matrixcity', 4, 80),
    omniworld=train_dataset('omniworld', data_path + 'omniworld', 24, 80),
    synthia=train_dataset('synthia', data_path + 'synthia', 30, 80),
    omniobject=train_dataset('omniobject', data_path + 'omniobject', 4, 80),
    transphy3d=train_dataset('transphy3d', data_path + 'transphy3d', 4, 80),
    synmirrorv2=train_dataset('synmirrorv2', data_path + 'synmirrorv2', 4, 80),
    ase=train_dataset('ase', data_path + 'ase', 10, 50, crop_cfg=WIDE_CROP),
    spring=train_dataset('spring', data_path + 'spring', 16, 80),
    carlaocc=train_dataset('carlaocc', data_path2 + 'carlaocc', 30, 80),
    tartanground=train_dataset('tartanground', data_path2 + 'tartanground', 30, 80, split_ext='parquet'),
    wildrgbd=train_dataset('wildrgbd', data_path + 'wildrgbd', 1, 3, crop_cfg=WIDE_CROP, refine=True),
    waymo=train_dataset('waymo', data_path + 'waymo', 8, 80, refine=True),
    arkitscenes=train_dataset('arkitscenes', data_path + 'arkitscenes', 8, 50, crop_cfg=WIDE_CROP, refine=True),
    arkitscenes_highres=train_dataset('arkitscenes_highres', data_path + 'arkitscenes_highres', 8, 50, crop_cfg=WIDE_CROP),
    scannetpp=train_dataset('scannetpp', data_path + 'scannetpp/data', 16, 50, crop_cfg=WIDE_CROP),
    dl3dv=train_dataset('dl3dv', data_path + 'dl3dv', 20, 80, crop_cfg=WIDE_CROP, refine=True),
    blendedmvs=train_dataset('blendedmvs', data_path + 'blendedmvs', 20, 80, crop_cfg=WIDE_CROP, refine=True),
)


weights = dict(
    tartanair=0.074,
    gtasfm=0.054,
    pointodyssey=0.010,
    bedlam=0.047,
    dynamic_replica=0.035,
    hypersim=0.035,
    lightwheelocc=0.044,
    mvssynth=0.032,
    matrixcity=0.040,
    omniworld=0.100,
    synthia=0.020,
    omniobject=0.051,
    ase=0.047,
    spring=0.060,
    wildrgbd=0.020,
    waymo=0.044,
    arkitscenes=0.044,
    arkitscenes_highres=0.030,
    scannetpp=0.074,
    transphy3d=0.050,
    dl3dv=0.074,
    blendedmvs=0.030,
    carlaocc=0.074,
    tartanground=0.050,
)
