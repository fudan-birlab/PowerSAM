# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

import power_sam.modeling as modeling
from power_sam.modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, RepViT,
                               SamBatch, PromptEncoderBatch, MaskDecoderBatch)
from power_sam.config import _C, _update_config_from_file
from yacs.config import CfgNode as CN
# from sam2.build_sam import build_sam2


prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size


def build_sam_vit_h(checkpoint=None, **kwargs):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31]
    )
    if kwargs.pop('encoder_only', False):
        return image_encoder
    return _build_sam(image_encoder, checkpoint, **kwargs)


def build_sam_vit_l(checkpoint=None, **kwargs):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23]
    )
    if kwargs.pop('encoder_only', False):
        return image_encoder
    return _build_sam(image_encoder, checkpoint, **kwargs)


def build_sam_vit_b(checkpoint=None, **kwargs):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11]
    )
    if kwargs.pop('encoder_only', False):
        return image_encoder
    return _build_sam(image_encoder, checkpoint, **kwargs)


def build_pruned_sam_vit(checkpoint=None, **kwargs):
    pruned_model = torch.load("../SlimSAM-master/checkpoints_50/vit_b_slim_step2_.pth")
    image_encoder = pruned_model.image_encoder.module
    if kwargs.pop('encoder_only', False):
        return image_encoder
    return _build_sam(image_encoder, checkpoint, **kwargs)


def build_power_sam(checkpoint=None, upsample_mode="bicubic", arch="m1"):
    image_encoder = RepViT(
        arch=arch,
        img_size=image_size,
        upsample_mode=upsample_mode,
        fuse=True
    )
    return _build_sam(image_encoder, checkpoint)


def build_sam2_1_hiera_tiny(checkpoint=None):
    return build_sam2(
        "../sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint,
    )


def build_sam2_1_hiera_small(checkpoint=None):
    return build_sam2(
        "../sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint,
    )


def build_sam2_1_hiera_base_plus(checkpoint=None):
    return build_sam2(
        "../sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
        checkpoint,
    )


def build_sam2_1_hiera_large(checkpoint=None):
    return build_sam2(
        "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint,
    )


sam_model_registry = {
    "default": build_power_sam,
    "power_sam": build_power_sam,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "pruned_vit": build_pruned_sam_vit,
    'hiera_tiny': build_sam2_1_hiera_tiny,
    'hiera_small': build_sam2_1_hiera_small,
    'hiera_base_plus': build_sam2_1_hiera_base_plus,
    'hiera_large': build_sam2_1_hiera_large,
}
build_sam = build_power_sam


def _build_sam_encoder(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
):
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    return image_encoder


def _build_sam(image_encoder, checkpoint, enable_batch=False, enable_distill=False, lora=False, rpn_head=None):
    sam_model = SamBatch if enable_batch else Sam
    prompt_encoder = PromptEncoderBatch if enable_batch else PromptEncoder
    mask_decoder = MaskDecoderBatch if enable_batch else MaskDecoder

    sam_args = dict(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=mask_decoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                lora=lora
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            yield_kd_targets=enable_distill,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if rpn_head != 'none':
        sam_args['rpn_head'] = rpn_head

    sam = sam_model(**sam_args)

    if not enable_distill:
        sam.eval()
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
            if not isinstance(state_dict, dict):
                return sam
            if 'model' in state_dict:
                state_dict = state_dict['model']
            print(sam.load_state_dict(state_dict, strict=False))
    return sam


# build sam model from the yaml config file
def build_sam_from_config(
        cfg_file,
        checkpoint=None,
        enable_distill=False,
        enable_batch=False,
        **kwargs):
    if isinstance(cfg_file, CN):
        config = cfg_file
    else:
        config = _C.clone()
        _update_config_from_file(config, cfg_file)
    model_type = config.MODEL.TYPE
    encoder_only = config.DISTILL.ENCODER_ONLY
    lora = config.DISTILL.LORA
    fuse = config.DISTILL.FUSE
    rpn_head = config.DISTILL.RPN_HEAD

    if model_type in ['vit_h', 'vit_l', 'vit_b', 'pruned_vit']:
        return sam_model_registry[model_type](
            checkpoint if checkpoint else config.MODEL.RESUME,
            enable_distill=enable_distill,
            enable_batch=enable_batch,
            encoder_only=encoder_only,
            lora=lora
        )

    if model_type in ['hiera_tiny', 'hiera_small', 'hiera_base_plus', 'hiera_large']:
        return sam_model_registry[model_type](
            checkpoint if checkpoint else config.MODEL.RESUME,
        )

    kwargs['upsample_mode'] = config.DISTILL.UPSAMPLE_MODE
    image_encoder = getattr(modeling, model_type)(fuse=fuse, **kwargs)

    if encoder_only:
        return image_encoder

    sam = _build_sam(image_encoder, checkpoint, enable_batch, enable_distill, lora, rpn_head)
    return sam
