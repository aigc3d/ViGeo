# Dataset config for depth completion training.

data_path = './data/'
train_test_split_path = 'train_test_split/'

IMAGE_AUGMENTATIONS = ['jittering', 'jpeg_loss', 'blurring', 'shot_noise', 'dof']
TRAIN_TENSOR_KEYS = ['image', 'depth', 'intrinsic', 'sky_mask', 'pose', 'prior', 'normal']
EVAL_TENSOR_KEYS = ['image', 'depth', 'intrinsic', 'prior']
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

TRAIN_SAMPLE_POINTS = dict(
    mode=['random', 'sift', 'orb', 'virtual_lidar', 'partial'],
    weights=[0.4, 0.1, 0.1, 0.2, 0.2],
    depth_noise=[0.0, 0.2],
    sample_num_range=[0.0001, 0.1],
    partial_masks_dir='./partial_masks_dir',
    split='train',
)


def train_pipeline(crop_cfg=DEFAULT_CROP):
    return [
        dict(type='perspective_crop', image_augmentation_list=IMAGE_AUGMENTATIONS, **crop_cfg),
        dict(type='sample_points', **TRAIN_SAMPLE_POINTS),
        dict(type='convert_to_tensor', keys=TRAIN_TENSOR_KEYS, meta_keys=META_KEYS),
    ]


def eval_pipeline(sample_num_range):
    return [
        dict(type='perspective_crop_eval', ensure_multiple_of=14),
        dict(
            type='sample_points',
            mode=['random'],
            weights=[1],
            depth_noise=None,
            sample_num_range=sample_num_range,
            split='eval',
        ),
        dict(type='convert_to_tensor', keys=EVAL_TENSOR_KEYS, meta_keys=META_KEYS),
    ]


def train_dataset(
    name,
    data_root,
    max_interval,
    max_depth,
    crop_cfg=DEFAULT_CROP,
    min_depth=0.0001,
):
    return dict(
        type=f'{name}_data',
        data_path=data_root,
        data_name=name,
        train_test_split=f'{train_test_split_path}{name}_train.json',
        max_interval=max_interval,
        min_depth=min_depth,
        max_depth=max_depth,
        pipeline=train_pipeline(crop_cfg),
    )


def val_dataset(
    name,
    data_root,
    sample_num_range,
    split='eval',
    start=0,
    end=None,
    min_depth=0.0001,
    max_depth=80.0,
):
    return dict(
        type=f'{name}_data',
        data_path=data_root,
        data_name=name,
        train_test_split=f'{train_test_split_path}{name}_{split}.json',
        pipeline=eval_pipeline(sample_num_range),
        start=start,
        end=end,
        min_depth=min_depth,
        max_depth=max_depth,
    )


val_datasets = dict(
    kitti=val_dataset('kitti', data_path + 'kitti', [1, 1], start=0, end=110, max_depth=200.0),
    bonn=val_dataset('bonn', data_path + 'bonn', [0.0016, 0.0016], start=0, end=50, max_depth=50.0),
    transphy3d=val_dataset('transphy3d', data_path + 'transphy3d', [0.0016, 0.0016], split='train', max_depth=50.0),
    dl3dv=val_dataset('dl3dv', data_path + 'dl3dv', [1, 1], split='train', max_depth=50.0),
)


train_datasets = dict(
    tartanair=train_dataset('tartanair', data_path + 'tartanair/', 12, 200),
    gtasfm=train_dataset('gtasfm', data_path + 'gtasfm/', 4, 200),
    pointodyssey=train_dataset('pointodyssey', data_path + 'pointodyssey/', 30, 50, crop_cfg=WIDE_CROP),
    bedlam=train_dataset('bedlam', data_path + 'bedlam/', 30, 50, crop_cfg=WIDE_CROP),
    dynamic_replica=train_dataset('dynamic_replica', data_path + 'dynamic_replica/', 30, 50, crop_cfg=WIDE_CROP),
    hypersim=train_dataset('hypersim', data_path + 'hypersim', 1, 50, crop_cfg=WIDE_CROP),
    lightwheelocc=train_dataset('lightwheelocc', data_path + 'lightwheelocc/', 2, 200),
    mvssynth=train_dataset('mvssynth', data_path + 'mvssynth', 12, 200),
    matrixcity=train_dataset('matrixcity', data_path + 'matrixcity', 4, 200),
    omniworld=train_dataset('omniworld', data_path + 'omniworld', 24, 200),
    synthia=train_dataset('synthia', data_path + 'synthia', 30, 200),
    omniobject=train_dataset('omniobject', data_path + 'omniobject', 4, 200),
    transphy3d=train_dataset('transphy3d', data_path + 'transphy3d', 4, 200),
    ase=train_dataset('ase', data_path + 'ase', 10, 50, crop_cfg=WIDE_CROP),
    spring=train_dataset('spring', data_path + 'spring', 16, 200),
    wildrgbd=train_dataset('wildrgbd', data_path + 'wildrgbd', 1, 50, crop_cfg=WIDE_CROP),
    waymo=train_dataset('waymo', data_path + 'waymo', 8, 200),
    arkitscenes=train_dataset('arkitscenes', data_path + 'arkitscenes', 8, 50, crop_cfg=WIDE_CROP),
    arkitscenes_highres=train_dataset('arkitscenes_highres', data_path + 'arkitscenes_highres', 8, 50, crop_cfg=WIDE_CROP),
    scannetpp=train_dataset('scannetpp', data_path + 'scannetpp/data', 16, 50, crop_cfg=WIDE_CROP),
)


weights = dict(
    tartanair=0.074,
    gtasfm=0.054,
    pointodyssey=0.032,
    bedlam=0.047,
    dynamic_replica=0.035,
    hypersim=0.015,
    lightwheelocc=0.054,
    mvssynth=0.010,
    matrixcity=0.040,
    omniworld=0.300,
    synthia=0.020,
    omniobject=0.051,
    ase=0.047,
    spring=0.002,
    wildrgbd=0.030,
    waymo=0.074,
    arkitscenes=0.074,
    arkitscenes_highres=0.030,
    scannetpp=0.074,
    transphy3d=0.074,
)
