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
    warm_epoch=5,
    batch_size=None,
    ensure_multiple_of=14,
    max_image_num=24,
    image_num=None,
    image_num_range=[2, 24],
    area_range=[50176, 50176],
    aspect_ratio_range=[0.5, 2.0])

loss_cfg=dict(
    losses=dict(
        scale_invariant_pointmap_loss=1.0,
        camera_loss=1.0,
        normal_loss=1.0,
        ray_loss=1.0,
    )
)

model=dict(
    type='ViGeoTrain',
    encoder='vitg',
    train_normal=False,
    epoch=0,
    warm_epoch=5,
    mode='chunk'
)

weight_cfg = dict(
    ckpt_path='ckpts/da3_vitg_pretrain.pth',
    load_keys=None,
    ignore_keys=None,
    freeze_keys=None,
    copy_mapping=None,
    load_target='pretrained'
)


# """Training params."""
learning_rate=1e-5
training_steps=50e3

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    final_div_factor=4,
    frequency=1
)