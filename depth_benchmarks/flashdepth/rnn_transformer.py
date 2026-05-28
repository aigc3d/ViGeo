import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2_layers.attention import CrossAttention

from einops import rearrange
import logging

class Block(nn.Module):
    def __init__(self, d_model, num_heads=2, mlp_expand=4):
        super().__init__()

        self.norm1 = nn.RMSNorm(d_model)
        self.attention = CrossAttention(dim=d_model, num_heads=num_heads)
        self.context_norm = nn.LayerNorm(d_model)
        
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_expand),
            nn.GELU(),
            nn.Linear(d_model * mlp_expand, d_model)
        )

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

class TransformerRNN(nn.Module):
    def __init__(self, dpt_dim, training_mode=True, **kwargs):
        super().__init__()

        '''
        dpt_dim
        state_size, enc_dim, dec_dim, n_heads, depth
        vocab_size: input and output dimensions 
        embedding_dim: hidden dimensions
        '''

       
        state_size = kwargs['mamba_d_state']
        self.state_size = state_size
        self.init_state = nn.Embedding(state_size, dpt_dim)
        self.state_embedding = nn.Linear(dpt_dim, state_size)
        
        self.embedding = nn.Linear(dpt_dim, state_size)

        self.write_layers = nn.ModuleList([Block(state_size) for i in range(kwargs['num_mamba_layers'])])
        self.read_layers = nn.ModuleList([Block(state_size) for i in range(kwargs['num_mamba_layers'])])

        self.mamba_type = kwargs['mamba_type']
        
        
        # assuming downsample factor is <=0.25
        if kwargs.get('use_upconv', False):
            self.upconv = nn.Sequential(
                nn.Conv2d(dpt_dim, dpt_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(dpt_dim, dpt_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(dpt_dim, dpt_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                # last interpolation is done outside and matches dpt features
            )
        else:
            self.upconv = nn.Identity()

        self.final_layer = nn.Identity()


        # positional encoding 
        self.input_resolution = (518,518)
        self.mult_factor = 2 ** kwargs['mamba_in_dpt_layer'][0]
        self.downsample_factor = kwargs['downsample_mamba'][0]
        num_patches = int((518/14*self.mult_factor*self.downsample_factor)) * int((518/14*self.mult_factor*self.downsample_factor))
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches, state_size))
        nn.init.trunc_normal_(self.img_pos_embed, std=0.02)
        logging.info(f'using pos_embed; shape: {self.img_pos_embed.shape}')
    

    def get_initial_state(self, x):
        B = x.shape[0]

        state = self.init_state(
            torch.arange(self.state_size, device=x.device)
        )

        state = state.expand(B, -1, -1)
        state = self.state_embedding(state)
        # TODO: add positional encoding? not sure if necessary; init_state is already a learnable embedding
        

        return state


    def start_new_sequence(self):
        self.state = None


    def forward_single_frame(self, frame, **kwargs):

        downsampled = kwargs['downsample_factor'] != 1.0

        x = frame.clone()

        B, L, D = x.shape

        x = self.embedding(x)


        x = x + self.interpolate_pos_encoding(x, kwargs['Thw'][1], kwargs['Thw'][2])
        

        if self.state is None:
            self.state = self.get_initial_state(x)


        # TODO: unlike in xlstm or mamba, we don't have a state for each layer
        state = self.state
        for w, r in zip(self.write_layers, self.read_layers):
            # WRITE  : state attends to frame  → update state
            state = w(state, x) 
            # READ   : frame attends to updated state → aligned frame features
            x     = r(x, state) 

        self.state = state

        if self.mamba_type == 'add' and not downsampled:
            out = x + frame

        if self.mamba_type == 'add' and downsampled:
            h,w = int(kwargs['dpt_shape'][0]*self.downsample_factor), int(kwargs['dpt_shape'][1]*self.downsample_factor)
            spatial_out = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            out = self.upconv(spatial_out)
           
        return out


    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype

        if self.input_resolution == (h, w): # only return if h==w==518 (can't rely on npatch=N because could have different dimensions still)
            return self.img_pos_embed

        pos_embed = self.img_pos_embed.float()
        patch_pos_embed = pos_embed
        dim = x.shape[-1]

        orig_h, orig_w  = self.input_resolution
        W_orig = int((orig_w/14*self.mult_factor*self.downsample_factor))
        H_orig = int((orig_h/14*self.mult_factor*self.downsample_factor))
        patch_pos_embed = patch_pos_embed.reshape(1, H_orig, W_orig, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, dim, H_orig, W_orig]

        W_new = int((w/14*self.mult_factor*self.downsample_factor))
        H_new = int((h/14*self.mult_factor*self.downsample_factor))

        patch_pos_embed = F.interpolate(
            patch_pos_embed, 
            size=(H_new, W_new),   
            mode="bicubic",
            align_corners=False
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)     # [1, H_new, W_new, dim]
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)     # [1, H_new*W_new, dim]
        return patch_pos_embed.to(previous_dtype)


