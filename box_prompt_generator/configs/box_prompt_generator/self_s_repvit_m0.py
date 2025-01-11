#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.501
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.716
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.565
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.345
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.348
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.662
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.533
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.533
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.359
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.365
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.710
# 11/08 16:22:59 - mmengine - INFO - bbox_mAP_copypaste: 0.501 0.716 0.565 0.345 0.348 0.662
# 11/08 16:22:59 - mmengine - INFO - Epoch(val) [300][7/7]    coco/TA_precision: 0.6040  coco/W_precision: nan  coco/SQR_precision: nan  coco/L_precision: 0.6430  coco/FV_precision: 0.6580  coco/CM_precision: 0.6320  coco/TV_precision: 0.4730  coco/QF_precision: 0.5550  coco/C_precision: 0.4650  coco/T_precision: 0.1680  coco/JYZ_precision: 0.5280  coco/QS_precision: 0.2870  coco/bbox_mAP: 0.5010  coco/bbox_mAP_50: 0.7160  coco/bbox_mAP_75: 0.5650  coco/bbox_mAP_s: 0.3450  coco/bbox_mAP_m: 0.3480  coco/bbox_mAP_l: 0.6620  data_time: 0.0334  time: 0.0785

# 11/08 16:22:59 - mmengine - INFO - 
# +----------+-------+--------+--------+-------+-------+-------+
# | category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
# +----------+-------+--------+--------+-------+-------+-------+
# | TA       | 0.604 | 0.851  | 0.644  | 0.377 | 0.663 | 0.742 |
# | W        | nan   | nan    | nan    | nan   | nan   | nan   |
# | SQR      | nan   | nan    | nan    | nan   | nan   | nan   |
# | L        | 0.643 | 0.851  | 0.851  | nan   | 0.0   | 0.754 |
# | FV       | 0.658 | 0.813  | 0.781  | 0.266 | 0.491 | 0.862 |
# | CM       | 0.632 | 0.847  | 0.719  | 0.269 | 0.519 | 0.771 |
# | TV       | 0.473 | 0.579  | 0.579  | 0.454 | 0.314 | 0.788 |
# | QF       | 0.555 | 0.871  | 0.59   | nan   | 0.0   | 0.635 |
# | C        | 0.465 | 0.663  | 0.663  | nan   | 0.6   | 0.404 |
# | T        | 0.168 | 0.337  | 0.0    | nan   | 0.0   | 0.5   |
# | JYZ      | 0.528 | 0.794  | 0.553  | 0.397 | 0.582 | 0.718 |
# | QS       | 0.287 | 0.554  | 0.271  | 0.303 | 0.308 | 0.446 |
# +----------+-------+--------+--------+-------+-------+-------+

_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './yolox_tta.py'
]

img_scale = (1024, 1024)  # width, height

# model settings
model = dict(
    type='BoundingBoxPromptGenerator',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(1024, 1024),
                size_divisor=32,
                interval=10)
        ]),
    backbone=dict(
        type='rep_vit_m0',
        # checkpoint="./output_substation_equipment/rep_vit_m0_fuse_enc_dec_4m_ft_bp_iter2b_substation_equipment_distill_50e/default/ckpt_epoch_49.pth",
        freeze=True,
        img_size=1024,
        upsample_mode="bicubic",
        fuse=True,
        use_rpn=True,
        out_indices=(2, 3, 4),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[80, 160, 320],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=12,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.5, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = 'data/self/'
dataset_type = 'SelfDataset'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/validation.json',
        data_prefix=dict(img='validation/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/self/annotations/validation.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator

# training settings
max_epochs = 300
num_last_epochs = 15
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(
        norm_decay_mult=0.,
        bias_decay_mult=0.,
        custom_keys={'backbone': dict(lr_mult=0.0)},
    )
)

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
