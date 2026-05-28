import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import logging
import torch.distributed as dist
from dataclasses import dataclass, field


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference.
    https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/utils/generation.py#L17C1-L34C44
    """

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    # seq_idx: torch.Tensor = None
    seq_idx_dict: dict = field(default_factory=dict)
    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0



class MambaBlock(nn.Module):
    def __init__(self, d_model, layer_idx, expand, d_state=64, d_conv=4, headdim=64, use_hydra=False):
        super().__init__()
        from mamba_ssm import Mamba2
        
        self.norm1 = nn.LayerNorm(d_model)

        if not use_hydra:
            self.mamba = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
                headdim=headdim
            )
        else:
            from .hydra import Hydra
            self.mamba = Hydra(
                d_model=d_model, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local non-causal convolution width
                expand=expand,    # Block expansion factor
                headdim=headdim,
                layer_idx=layer_idx,
                use_mem_eff_path=False,    # Nightly release. Thanks to Alston Lo
            )
                    
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
      
    
    def forward(self, x, inference_params=None):

        residual = x
        x = self.norm1(x)
        # x = self.forward_mamba(x, inference_params=inference_params)
        x = self.mamba(x, inference_params=inference_params)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

    # @torch.compiler.disable
    # def forward_mamba(self, x, inference_params):
    #     return self.mamba(x, inference_params=inference_params)

class MambaModel(nn.Module):
    # https://github.com/state-spaces/mamba?tab=readme-ov-file
    def __init__(self, dpt_dim, mamba_type, num_mamba_layers, batch_size, mamba_d_conv=256, d_state=256,
                use_hydra=False, mamba_in_dpt_layer=[2], **kwargs):
        super().__init__()
        
        # d_model (dpt_dim) * expand / headdim = multiple of 8
        expand = 2
        headdim = 64 # these two defaults work for ViT-L where dpt_dim=256

        if dpt_dim == 64: # ViT-S
            headdim = 32
            expand = 4
        
        # each mamba layer is roughly 3 * expand * d_model^2 parameters
        self.blocks = nn.ModuleList([
            nn.ModuleList(
                [MambaBlock( 
                    d_model=dpt_dim, # Model (feature) dimension
                    expand=expand, # Expansion factor in MLP
                    d_state=d_state, # SSM state dimension
                    d_conv=mamba_d_conv, # Local convolution kernel size
                    layer_idx=layer_idx,
                    headdim=headdim,
                    use_hydra=use_hydra
                ) for layer_idx in range(num_mamba_layers)]
            ) for _ in range(len(mamba_in_dpt_layer))
        ])

        self.mamba_in_dpt_layer = mamba_in_dpt_layer
    
        if mamba_type == 'modulation':
            self.final_layer = nn.Sequential(
                nn.GELU(),
                nn.Linear(dpt_dim, dpt_dim*2),
            )
            nn.init.zeros_(self.final_layer[1].weight)
            nn.init.zeros_(self.final_layer[1].bias)
            #logging.info('zero init mamba modulation')
            

        elif mamba_type == 'add':
            self.final_layer = nn.Sequential(
                nn.GELU(),
                nn.Linear(dpt_dim, dpt_dim),
            )
            nn.init.zeros_(self.final_layer[1].weight)
            nn.init.zeros_(self.final_layer[1].bias)
            #logging.info('zero init mamba add')

        elif mamba_type == 'rnn':
            # should initialize all blocks, not just last one; but doesn't matter for now
            last_block = self.blocks[-1]
            # Zero init the Mamba projection
            last_block.mamba.out_proj.weight.data.zero_()
            if last_block.mamba.out_proj.bias is not None:
                last_block.mamba.out_proj.bias.data.zero_()
            # Zero init the MLP layers
            last_block.mlp[2].weight.data.zero_()
            last_block.mlp[2].bias.data.zero_()


        self.max_seqlen = 60000 # not used unless using rotary embedding 
        self.max_batch_size = batch_size
        self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen, max_batch_size=self.max_batch_size) for _ in range(len(self.mamba_in_dpt_layer))]


        self.mamba_type = mamba_type

    def start_new_sequence(self):
        """
        reset for new video sequence
        """

        self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen, max_batch_size=self.max_batch_size) for _ in range(len(self.mamba_in_dpt_layer))]

    
    def forward_single_frame(self, frame, **kwargs):
        """
        processes a single frame of shape (B, L, d_model) 
        and updates the hidden states in-place so the next call continues from here.
        example:
            sample a video from dataloader -> 
            mamba.start_new_sequence()
            for i in range(num_frames):
                frame = get_frame(i)           # shape (B, L, d_model)
                output = mamba.forward_single_frame(frame)
        """
        downsampled = kwargs['downsample_factor'] != 1.0
        
        if 'in_dpt_layer' not in kwargs and not isinstance(self.blocks[0], nn.ModuleList):
            blocks = self.blocks
            inference_params = self.inference_params
        else:
            blocks = self.blocks[kwargs['in_dpt_layer']]
            inference_params = self.inference_params[kwargs['in_dpt_layer']]
        
        seqlen = frame.shape[1]  # L

        x = frame.clone()

        
        for layer_idx, block in enumerate(blocks):
            x = block(x, inference_params=inference_params)


        if self.mamba_type == 'modulation' and not downsampled:
            x = self.final_layer(x)
            scale, shift = x.chunk(2, dim=-1)  
            x = (1 + scale) * frame + shift

        if self.mamba_type == 'add' and not downsampled:
            x = self.final_layer(x)
            x = x + frame

        inference_params.seqlen_offset += seqlen
        return x  # shape (B, L, d_model)

