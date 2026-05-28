# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


from flash_attn import flash_attn_func
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, context: Tensor, attn_bias=None) -> Tensor:
        """
        Args:
            x: Query input of shape (B, N, C)
            context: Key/Value input of shape (B, M, C)
            attn_bias: Optional attention bias tensor
        """
 
        B, N, C = x.shape
        _, M, _ = context.shape

        # Project and reshape q to [B, N, num_heads, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        
        # Project and reshape kv to [B, M, 2, num_heads, head_dim]
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads)
        k, v = unbind(kv, 2)  

        #x = memory_efficient_attention(q, k, v)
        x = flash_attn_func(q,k,v) # flash attention allows different sequence length for q and k without mask
       
        x = x.reshape(B, N, C)

        x = self.proj(x)
        return x


class VanillaLinearAttention(nn.Module):
    def __init__(
        self,
        dim: int, 
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim, bias=proj_bias)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        logging.info(f"Using linear attention")

    def forward(self, x, rotary_emb=None, ret_attn=False):
        """
        Compute linear attention (assume no GQA)
        N: sequence length
        h: num_heads
        d: dim//num_heads (head_dim)
        Assume output shape is [B, N, h, d] (required shape to xops attention)
        """
        B, N, C = x.shape
        # Reshape projections to [B, N, num_heads, head_dim]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim) 
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        

        if rotary_emb is not None:
            # Rearrange dimensions to [B, num_heads, N, C//num_heads]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)
            
            # Restore original dimension order [B, N, num_heads, C//num_heads]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

        q = q.softmax(dim=-1) * self.scale    # across head_dim, shape: [B, N, h, d]
        k = k.softmax(dim=1)      # across seq_len (N), shape: [B, N, h, d]
        
    
        # Compute linear attention
        # https://towardsdatascience.com/linear-attention-is-all-you-need-5fa9c845c1b5
        kv = torch.einsum('bnhf,bnhd->bhfd', k, v) # [B, h, 2d, d]
        y = torch.einsum('bnhf,bhfd->bnhd', q, kv) # [B, N, h, d]

        # Reshape to [B, N, C]
        B, N, h, d = y.shape
        y = y.reshape(B, N, h * d)

        return self.o_proj(y)


class GateLinearAttention(nn.Module):
    def __init__(
        self,
        dim: int, 
        num_heads: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.gla = GatedLinearAttention(
            hidden_size=dim,
            num_heads=num_heads,
            expand_k=1.0,
            expand_v=1.0,
        )
        logging.info(f"Using gated linear attention")

    def forward(self, x, rotary_emb=None, ret_attn=False):
        output, _, _ = self.gla(x)
        # final projection in gla has bias=False
        return output

class FlashLinearAttention(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        self.flash_linear_attn = LinearAttention(
            hidden_size=dim, 
            num_heads=num_heads,
            expand_k=1.0,
            expand_v=1.0,
            feature_map=kwargs.get('attn_feature_map', 'identity'), # e.g. hedgehog, elu
            output_norm='identity', # alt is rms norm before final o_proj, shouldn't need this
            # norm_q=True,norm_k=True,
            # softmax_norm=True, # add softmax operations like vanilla linear attention
        )
        # logging.info(f"Using flash linear attention")

    def forward(self, x):
        output = self.flash_linear_attn(x)
        return output


# # https://github.com/Dao-AILab/flash-attention
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# class SlidingWindowAttention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = False,
#         proj_bias: bool = True,
#     ) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)

#     def forward(self, x, context, window_size):
#         """
#         Args:
#             x: Query input of shape (B, N, C)
#             context: Key/Value input of shape (B, M, C)
            
#             If window_size != (-1, -1), implements sliding window local attention. Query at position i
#             will only attend to keys between 
#             [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
#         """


#         B, N, C = x.shape
#         _, M, _ = context.shape

#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
#         kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads)
#         k, v = unbind(kv, 2)  

#         # Apply memory efficient attention
#         x = flash_attn_func(q, k, v, window_size=window_size)
#         x = x.reshape(B, N, C)

#         x = self.proj(x)
#         return x