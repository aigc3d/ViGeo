# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from .dinov2_layers import Mlp, PatchEmbed, NestedTensorBlock as Block
from .dinov2_layers.attention import MemEffAttention, FlashLinearAttention


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        linearization_configs=None,
        skip_connections=False,
        compression_configs=None,
        **kwargs
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        # need to re-init or interpolate original dimensions if patch size != 14
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        else:
            raise NotImplementedError

        block_kwargs = {
            'dim': embed_dim,
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio,
            'qkv_bias': qkv_bias,
            'proj_bias': proj_bias,
            'ffn_bias': ffn_bias,
            'norm_layer': norm_layer,
            'act_layer': act_layer,
            'ffn_layer': ffn_layer,
            'init_values': init_values,
        }

        if linearization_configs is None:
            blocks_list = [
                block_fn(
                    **block_kwargs,
                    drop_path=dpr[i],
                    attn_class=MemEffAttention,
                )
                for i in range(depth)
            ]
        else:
            # TODO: right now these are the layers to NOT linearize!!
            linear_attn_layers = linearization_configs['layers']
            
            blocks_list = []
            for i in range(depth):
                if i not in linear_attn_layers:
                    blocks_list.append(block_fn(**block_kwargs, drop_path=dpr[i], 
                    attn_class=FlashLinearAttention, attn_feature_map=linearization_configs['feature_map']))
                else:
                    blocks_list.append(block_fn(**block_kwargs, drop_path=dpr[i], attn_class=MemEffAttention))

        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))


        if skip_connections:
            self.skip_connections_projections = nn.ModuleList([
                nn.Linear(embed_dim*2, embed_dim) for _ in range(3)
            ])

        self.init_weights()

        if compression_configs is not None:
            from .upstream_fusion import UpstreamFusion
            self.upstream_fusion = UpstreamFusion(**compression_configs)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias
        )
        
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None: # self.register_tokens is None
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x



    def forward_features(self, x, masks=None):

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    
    def forward_with_skip_connections(self, x):
        
        
        # Where do we skip from?  e.g. block 11 grabs from block 4
        get_origin_skip_layer = {
            11: 4,
            17: 11,
            23: 17
        }

        # Which index in ModuleList do we use for the skip 4->11, 11->17, etc.
        proj_index_map = {
            11: 0,
            17: 1,
            23: 2
        }

        skip_features = {}

        for idx, blk in enumerate(self.blocks):
            # 1) Forward pass through the block
            x = blk(x)

            # 2) If this block is a "skip destination," then do the skip from its origin
            if idx in [11, 17, 23]:
                origin_idx = get_origin_skip_layer[idx]   # e.g. 11 -> 4
                skip_x     = skip_features[origin_idx]    # fetch stored features

                # Concat along the embedding dimension (B,L,D -> B,L,2D)
                concat_x = torch.cat([skip_x, x], dim=-1)

                # Project back down to D
                pidx = proj_index_map[idx]  # 11->0, 17->1, 23->2
                x = self.skip_connections_projections[pidx](concat_x)

            # 3) If this block is a "skip origin," then store its output for use by a future block
            if idx in [4, 11, 17]:
                skip_features[idx] = x

        return x

    def _get_intermediate_layers_not_chunked(self, x, blocks_to_take):
        x = self.prepare_tokens_with_masks(x)
        
        output = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_upstream_fusion(self, x, blocks_to_take, **kwargs):
        
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        x = self.prepare_tokens_with_masks(x)
        compression_configs = kwargs['compression_configs']
        high_res_x_tokens = compression_configs['high_res_x_tokens']
        up_to_main_read_layers = compression_configs['up_to_main_read_layers']
        main_to_up_write_layers = compression_configs['main_to_up_write_layers']

        
        output = []

        ## REMOVE THE CLASS TOKEN!!!
        x = x[:, 1:]
        # if not same tokenizer, class token was not added to high res
        if compression_configs['same_tokenizer']:
            high_res_x_tokens = high_res_x_tokens[:, 1:]

        for i, blk in enumerate(self.blocks):
            
            if i in up_to_main_read_layers:
                x = self.upstream_fusion(main_x=x, high_res_x=high_res_x_tokens, layer_idx=i, read_to_main=True, patch_h=patch_h, patch_w=patch_w)

            x = blk(x)

            if i in main_to_up_write_layers:
                high_res_x_tokens = self.upstream_fusion(main_x=x, high_res_x=high_res_x_tokens, layer_idx=i, write_to_high=True, patch_h=patch_h, patch_w=patch_w)

            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output, high_res_x_tokens

    # @torch.compile
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        norm=True,
        **kwargs
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        
        
        if kwargs.get('compression_configs') is not None:
            outputs, high_res_x_tokens = self._get_intermediate_layers_upstream_fusion(x, n, **kwargs)
            high_res_x_tokens = self.norm(high_res_x_tokens)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        
        if norm: # True
            outputs = [self.norm(out) for out in outputs]
    
        # class token already removed in the upstream fusion
        if kwargs.get('compression_configs') is None:
            class_tokens = [out[:, 0] for out in outputs]
            outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs] # self.num_register_tokens=0
        
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if kwargs.get('compression_configs') is not None:
            return outputs, high_res_x_tokens
        else:
            return outputs

    def forward(self, x, **kwargs):

        x = self.prepare_tokens_with_masks(x)
        
        if kwargs.get('skip_connections', False):
            x = self.forward_with_skip_connections(x)

        else:
            for blk in self.blocks:
                x = blk(x)
        
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
        }




def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=14, num_register_tokens=0, linearization_configs=None, skip_connections=False, compression_configs=None, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=Block, #partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        linearization_configs=linearization_configs,
        skip_connections=skip_connections,
        compression_configs=compression_configs,
        **kwargs,
    )
    return model

def vit_small(patch_size=14, num_register_tokens=0, linearization_configs=None, skip_connections=False, compression_configs=None, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=Block, #partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        linearization_configs=linearization_configs,
        skip_connections=skip_connections,
        compression_configs=compression_configs,
        **kwargs,
    )
    return model



def DINOv2(model_name, patch_size, linearization_configs=None, skip_connections=False, compression_configs=None):
    model_zoo = {
        "vits": vit_small, 
        "vitb": vit_base, 
        "vitl": vit_large, 
    }
    
    return model_zoo[model_name](
        img_size=518,
        patch_size=patch_size,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        linearization_configs=linearization_configs,
        skip_connections=skip_connections,
        compression_configs=compression_configs,
    )
