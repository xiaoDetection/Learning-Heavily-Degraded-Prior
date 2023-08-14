_base_=[
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    './dataset.py', 
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_2x.py'
]

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(
        type='OurHook',
        cfg='configs/rftm/rftm_x101.py',
        cp='checkpoints/cascade_rcnn_x101_dfui.pth',
        priority=30)
]

pretrained_model = dict(
    type='ResNeXtCustom',
    use_RFTM=False,
    depth=101,
    groups=64,
    base_width=4,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    style='pytorch',
    init_cfg=None)

model = dict(
    type='CascadeRCNNCustom',
    backbone=dict(
        type='ResNeXtCustom',
        use_RFTM=True,
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=None),
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/cascade_rcnn_x101_dfui.pth'))

optimizer = dict(
    constructor='DefaultOptimizerConstructorCustom',
    optimizer_finetune = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    optimizer_RFTM = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
)
optimizer_config = dict(_delete_=True, type='OptimizerHookCustom',grad_clip=dict(max_norm=32, norm_type=2))
lr_config = dict(
    step=[8, 14, 19]
)
runner = dict(type='EpochBasedRunnerCustom', max_epochs=24)

data = dict(
    workers_per_gpu=4
)
# to use cuda in dataloader
mp_start_method = 'spawn'