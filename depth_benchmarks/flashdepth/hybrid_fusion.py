import torch
import torch.nn as nn
import torch.nn.functional as F
from .dinov2_layers.attention import MemEffAttention
from .dinov2_layers.attention import CrossAttention
import logging
import math

class Block(nn.Module):
    def __init__(self, d_model, num_heads=2, mlp_expand=4, proj_dim_in=1024, in_proj=False, out_proj=False):
        super().__init__()
        
        '''
        in_proj means first layer of the block
        out_proj means last layer of the block; do projection accordingly
        '''

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CrossAttention(dim=d_model, num_heads=num_heads, zero_init=True)
        self.context_norm = nn.LayerNorm(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_expand),
            nn.GELU(),
            nn.Linear(d_model * mlp_expand, d_model)
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

        # only using proj_dim_in for context projection, if needed
        if proj_dim_in != d_model:
            self.context_proj = nn.Linear(proj_dim_in, d_model)
        else:
            self.context_proj = nn.Identity()



    def forward(self, x, context):

        residual = x
        x = self.norm1(x)
        context = self.context_norm(context)
        x = self.attention(x, context)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        
        return x

class HybridFusion(nn.Module):
    def __init__(
        self, 
        d_model, # from DPT-S output which is 64
        num_blocks=4, # number of blocks per DPT-layer
        num_heads=2,
        mlp_expand=2,
        patch_size=14,
        layers_to_skip=[],
        **kwargs
    ):
        super(HybridFusion, self).__init__()
        '''
        pass
        '''        

        self.patch_size = patch_size
        self.layers_to_skip = layers_to_skip

        self.blocks = nn.ModuleList([
                nn.ModuleList([Block(d_model, num_heads=num_heads, mlp_expand=mlp_expand, proj_dim_in=256, in_proj=i==0, out_proj=i==num_blocks-1) for i in range(num_blocks)])
                for _ in range( 4-len(layers_to_skip) ) 
            ])

    
        #logging.info('HybridFusion: layers to skip, num_blocks, num_heads, mlp_expand: %s, %s, %s, %s', layers_to_skip, num_blocks, num_heads, mlp_expand)


    def forward(self, x, context, path_idx):

        if path_idx in self.layers_to_skip:
            return x

        assert x.ndim ==4 and context.ndim ==4, "x and context must be (B,C,h,w)"
        
        B,D,h,w = x.shape

        # TODO: might be worth trying interpolating context to same resolution, then adding positional embedding
        # currently using original resolution and relying on positional embedding from ViT encoders
        # context = F.interpolate(context, size=(h,w), mode='bilinear', align_corners=False)

        x = self.reshape_to_spatial(x, spatial_to_sequence=True)
        context = self.reshape_to_spatial(context, spatial_to_sequence=True)

    
        for idx, block in enumerate(self.blocks[path_idx]):
            if idx == 0:
                context = block.context_proj(context)
            x = block(x, context)

        x = self.reshape_to_spatial(x, h, w)
        assert x.shape == (B,D,h,w), "x shape must be (B,D,h,w)"

        return x
        
    

    def reshape_to_spatial(self, x, patch_h=None, patch_w=None, spatial_to_sequence=False):
        if spatial_to_sequence:
            B,C,h,w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        else:
            B,L,D = x.shape
            x = x.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)
        return x

