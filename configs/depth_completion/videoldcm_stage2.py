_base_ = ['./datasets.py']

train_data_lists = [
    'tartanair', 'gtasfm', 'pointodyssey', 'bedlam',
    'dynamic_replica', 'lightwheelocc', 'hypersim',
    'omniobject', 'mvssynth', 'matrixcity', 'omniworld',
    'synthia', 'ase', 'spring', 'transphy3d',
]

train_data = dict(
    type='train_data',
    datasets=[_base_.train_datasets[item] for item in train_data_lists],
)

val_data = _base_.val_datasets['kitti']

train_sampler_config = dict(
    weights=[_base_.weights[item] for item in train_data_lists],
    num_iterations=2000,
    warm_epoch=0,
    batch_size=None,
    ensure_multiple_of=14,
    max_image_num=16,
    image_num=None,
    image_num_range=[2, 16],
    area_range=[112896, 268324],
    aspect_ratio_range=[0.5, 2.0],
)

loss_cfg = dict(
    mode='depth_completion',
    losses=dict(
        conf_loss=1.0,
    ),
)

model = dict(
    type='videoldcm',
    encoder='vitl',
    moge_path='ckpts/moge-2-vits-normal/model.pt',
    train_conf=True,
)

checkpoint_path = 'logs/videoldcm_stage1/tensorboard/version_0/checkpoints/last.ckpt'
load_target = None
load_keys = ['pretrained', 'point_head']
ignore_keys = None
freeze_keys = ['moge', 'pretrained', 'point_head']
copy_mapping = None
weight_cfg = dict(
    ckpt_path=checkpoint_path,
    load_target=load_target,
    load_keys=load_keys,
    ignore_keys=ignore_keys,
    freeze_keys=freeze_keys,
    copy_mapping=copy_mapping,
)

learning_rate = 5e-5
training_steps = 20e3

optimizer = dict(
    type='AdamW',
    lr=learning_rate,
    weight_decay=0.01,
)

lr_scheduler = dict(
    type='OneCycleLR',
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy='cos',
    interval='step',
    frequency=1,
)
