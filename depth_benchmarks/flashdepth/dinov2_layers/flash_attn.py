# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


import sys 
sys.path.append('/root/gene/video-depth/depth_anything_v2/dinov2_layers')
from fla.modules import RMSNorm
# from fla.modules.feature_map import (DPFPFeatureMap, HadamardFeatureMap,
#                                      HedgehogFeatureMap, T2RFeatureMap)
from fla.ops.linear_attn import chunk_linear_attn

class HedgehogFeatureMap(nn.Module):

    r"""
    Hedgehog feature map as introduced in
    `The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry <https://arxiv.org/abs/2402.04347>`_
    """

    def __init__(
        self,
        head_dim: int
    ):
        super().__init__()
        # Trainable map
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights_()

    def init_weights_(self):
        """Initialize trainable map as identity"""
        with torch.no_grad():
            identity = torch.eye(*self.layer.weight.shape[-2:], dtype=torch.float)
            self.layer.weight.copy_(identity.to(self.layer.weight))
        nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        casttype = x.dtype
        x = self.layer(x)  
        return torch.cat([2*x, -2*x], dim=-1).softmax(-1).to(casttype) # softmax casts to float32


class LearnableFeatureMap(nn.Module):

    r"""
    learnable linear layer, with vanilla scaling of q,k rather than hedgehog
    """

    def __init__(
        self,
        head_dim: int,
        mlp=False
    ):
        super().__init__()
        # Trainable map
        
        if mlp:
            self.layer = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.SiLU(),
                nn.Linear(head_dim, head_dim),
            )
        else:
            self.layer = nn.Linear(head_dim, head_dim)

        self.init_weights_()

    def init_weights_(self):
        """Initialize trainable map as identity"""
        with torch.no_grad():
            if isinstance(self.layer, nn.Sequential):  # mlp=True case
                # Initialize both layers as identity
                first_layer = self.layer[0]
                last_layer = self.layer[-1]
                
                # Set both linear layers to identity
                for layer in [first_layer, last_layer]:
                    identity = torch.eye(*layer.weight.shape[-2:], dtype=torch.float)
                    layer.weight.copy_(identity.to(layer.weight))
                    nn.init.zeros_(layer.bias)
            else:  # mlp=False case
                identity = torch.eye(*self.layer.weight.shape[-2:], dtype=torch.float)
                self.layer.weight.copy_(identity.to(self.layer.weight))
                nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x = self.layer(x) 
        return x


class LinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: str = 1024,
        expand_k: int = 1.0,
        expand_v: int = 1.0,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        feature_map: str = 'elementwise_product',
        tie_feature_map_qk: bool = False,
        output_norm: str = 'rmsnorm',
        norm_q: bool = False,
        norm_k: bool = False,
        # standard linear attention normalization
        do_feature_map_norm: bool = False,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        # **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'learnable':
            self.feature_map_q = LearnableFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = LearnableFeatureMap(head_dim=self.head_qk_dim)
        
        elif feature_map == 'learnable_mlp':
            self.feature_map_q = LearnableFeatureMap(head_dim=self.head_qk_dim, mlp=True)
            self.feature_map_k = LearnableFeatureMap(head_dim=self.head_qk_dim, mlp=True)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.feature_map = feature_map

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)

        if output_norm == 'rmsnorm':
            self.norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        elif output_norm == 'identity':
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{output_norm}`.")

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k

        self.apply(self._initialize_weights)


    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, x):
        mode = self.mode
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v = (repeat(x, '... (h d) -> ... (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v))
        else:
            k, v = (rearrange(x, '... (h d) -> ... h d', h=self.num_kv_heads) for x in (k, v))

        q = self.feature_map_q(q) 
        k = self.feature_map_k(k)

        # qkv all have shape: b, l, h, d

        ## some quick logs for time profiling with attention + DPT + mamba downsampled 10x
        ## here I'm replacing all ViT-L layers
        # 1. softmax attention: 5.2 fps
        # 2. linear attention identity: 7.7 fps
        # 3. linear attention hedgehog: 5.5 fps
        # 4. linear attention learnable softmax: 1.6 fps
        # 5. linear attention learnable softplus (v1): 6.8 fps
        # 6. linear attention learnable hybrid (v2): 6.9 fps


        if self.feature_map == 'learnable':
            # the softmax operation for k is very slow because it's across the sequence length
            # q = q.softmax(dim=-1).to(q.dtype)  # across head_dim 
            # k = k.softmax(dim=1).to(k.dtype)      # across seq_len (l)

            # so I'm doing softplus (to ensure positive values) and then a simple normalization as a replacement for now
            # v1: softplus for both
            q = F.softplus(q).to(q.dtype)
            k = F.softplus(k).to(k.dtype)
            q = q / (q.sum(-1, True, dtype=q.dtype) + 1e-4)
            k = k / (k.sum(-1, True, dtype=k.dtype) + 1e-4)    

            # v2: softplus for k, softmax for q
            # q = q.softmax(dim=-1).to(q.dtype)  # across head_dim 
            # k = F.softplus(k).to(k.dtype)
            # k = k / (k.sum(-1, True, dtype=k.dtype) + 1e-4)


        # equivalent to: https://github.com/fla-org/flash-linear-attention/blob/79d1b615698bd9c42161b047c334818b688683e9/fla/ops/linear_attn/naive.py
        o, final_state = chunk_linear_attn(
            q=q,
            k=k,
            v=v,
            normalize=self.do_feature_map_norm,
            head_first=False
        )
        
        B, N, h, d = o.shape
        o = o.reshape(B, N, h * d)

        o = self.norm(o)
        o = self.o_proj(o)
        return o
