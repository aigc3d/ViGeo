_base_ = ['./datasets.py']

train_data_lists = ['tartanair', 'gtasfm', 'pointodyssey', 'bedlam', 
'dynamic_replica', 'lightwheelocc', 'hypersim', 
'omniobject', 'mvssynth', 'matrixcity', 'omniworld', 'synthia', 'ase', 'spring', 
'wildrgbd', 'arkitscenes', 'arkitscenes_highres', 'waymo', 'scannetpp', 'dl3dv', 'blendedmvs', 'transphy3d', 'carlaocc', 'tartanground']

train_data = dict(
    type='train_data',
    datasets=[_base_.train_datasets[item] for item in train_data_lists])

val_data = _base_.val_datasets['sintel']

train_sampler_config = dict(
    weights=[_base_.weights[item] for item in train_data_lists],
    num_iterations=2000,
    warm_epoch=0,
    batch_size=None,
    ensure_multiple_of=14,
    max_image_num=24,
    image_num=None,
    image_num_range=[2, 24],
    area_range=[112896, 268324],
    aspect_ratio_range=[0.5, 2.0])

loss_cfg=dict(
    losses=dict(
        scale_invariant_pointmap_loss=1.0,
        camera_loss=1.0,
        normal_loss=1.0,
        ray_loss=1.0,
        normal_map_loss=0.1,
        mask_loss=1.0
    )
)

model=dict(
    type='ViGeoTrain',
    encoder='vitg',
    train_normal=True,
    train_mask=True,
    train_conf=False,
    epoch=0,
    warm_epoch=0,
    mode='chunk'
)

weight_cfg=dict(
    ckpt_path='logs/vigeo_stage2_336x336_518x518/tensorboard/version_0/checkpoints/last.ckpt',
    load_keys=['pretrained', 'decoder', 'point_head', 'camera_head', 'ray_head', 'mask_head'],
    ignore_keys=None,
    freeze_keys=None,
    copy_mapping=dict(normal_head='point_head'),
    load_target=None,
)

# """Training params."""
learning_rate=1e-5
training_steps=20e3

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.01,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    final_div_factor=4,
    frequency=1
)
