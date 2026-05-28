import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .dinov2 import DINOv2

from .mamba import MambaModel
from .original_dpt import DPTHead
from .helpers import *

class FlashDepth(nn.Module):
    def __init__(
        self, 
        vit_size='vitl', 
        dpt_dim=256, 
        out_channels=[256, 512, 1024, 1024], 
        patch_size=14,
        **kwargs
    ):
        super(FlashDepth, self).__init__()

        encoder = vit_size
        model_configs = {
            'vits': {'encoder': 'vits', 'dpt_dim': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'dpt_dim': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        dpt_dim = model_configs[encoder]['dpt_dim']
        out_channels = model_configs[encoder]['out_channels']

        self.patch_size = patch_size
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23], 
        }


        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder, patch_size=patch_size)

        self.downsample_mamba = [0.1]
        self.mamba_in_dpt_layer = [3]

        self.mamba = MambaModel(dpt_dim, mamba_type="add", num_mamba_layers=4, batch_size=4, mamba_d_conv=4, d_state=256, )
            
        self.depth_head = DPTHead(self.pretrained.embed_dim, dpt_dim=dpt_dim, out_channels=out_channels)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
           
    def dpt_features_to_mamba(self, input_shape, dpt_features, in_dpt_layer):
        # reshape to (B, T*h*w, c) for mamba
        if len(input_shape)==4:
            B, C, H, W = input_shape
            T = 1
        else:
            B, T, C, H, W = input_shape
        BT, c, h, w = dpt_features.shape
        assert BT == B*T, f"Expected batch dimension {B*T}, got {BT}" # sanity check

        downsample_factor = self.downsample_mamba[in_dpt_layer]


        if downsample_factor != 1.0:
            original_dpt_features = dpt_features.clone()
            original_dpt_features = rearrange(original_dpt_features, '(b t) c h w -> b t c h w', b=B, t=T)
            dpt_features = F.adaptive_avg_pool2d(dpt_features, (int(h*downsample_factor), int(w*downsample_factor)))

        
        dpt_features = rearrange(dpt_features, '(b t) c h w -> b t (h w) c', b=B, t=T)
        
        mamba_kwargs = dict(Thw = (1, H, W), dpt_shape=(h,w), downsample_factor=downsample_factor, in_dpt_layer=in_dpt_layer)

        # mamba_out = torch.zeros_like(dpt_features)
        mamba_out = []

        for i in range(T):
            seq_out = self.mamba.forward_single_frame(dpt_features[:,i,...], **mamba_kwargs)
            if downsample_factor != 1.0:
                assert self.mamba.mamba_type == 'add'
                if seq_out.ndim == 3:
                    spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=int(h*downsample_factor), w=int(w*downsample_factor))
                else:
                    spatial_out = seq_out
                spatial_out = F.interpolate(spatial_out, (h,w), mode="bilinear", align_corners=True) 
                
                seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')
                seq_out = self.mamba.final_layer(seq_out)

                spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=h, w=w)
                spatial_out = spatial_out + original_dpt_features[:,i,...]
                seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')
            
            mamba_out.append(seq_out)

        # reshape back to spatial format (B*T, c, h, w)
        mamba_out = torch.stack(mamba_out, dim=1)
        mamba_out = rearrange(mamba_out, 'b t (h w) c -> (b t) c h w', h=h, w=w, b=B)

       
        return mamba_out

    def get_dpt_features(self, x, input_shape=None):

        self.input_resolution = (x.shape[-1], x.shape[-2]) # w,h
       
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        intermediate_features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder])
            
        out = self.depth_head.forward_with_mamba(intermediate_features, patch_h, patch_w, temporal_layer=self.mamba_in_dpt_layer, mamba_fn=self.dpt_features_to_mamba, shape_placeholder=input_shape)

        return out
    
    def final_head(self, x, patch_h, patch_w):
        
        out  = self.depth_head.scratch.output_conv1(x)
        
        bs = out.shape[0]
        target_h = int(patch_h * self.patch_size)
        target_w = int(patch_w * self.patch_size)
        
        # Process in batches of 60 frames
        # out = F.interpolate(out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)), mode="bilinear", align_corners=True)
        # int max is 2147483647; for B,C=128,H=518,W=518, can only handle 60 frames
        # for vit-s using raw 2k resolution, can only handle 30 frames (2147483647/(32*1064*1904)=33)
        outputs = []
        for i in range(0, bs, 30):
            batch = out[i:i+30]  # Take up to 60 frames
            batch_out = F.interpolate(batch, (target_h, target_w), 
                                    mode="bilinear", align_corners=True)
            outputs.append(batch_out)
        
        out = torch.cat(outputs, dim=0)
        out = self.depth_head.scratch.output_conv2(out)
        # if out.max() <=0:
        #     logging.warning("Depth is all zeros")
        depth = F.relu(out).squeeze(1)
    
        return depth
    
    @torch.no_grad()
    def forward(self, video, **kwargs):
        
        preds = []

        self.mamba.start_new_sequence()

        for i in range(video.shape[1]):
            frame = video[:, i, :, :, :].to(torch.cuda.current_device())
            B, C, H, W = frame.shape

       
            patch_h, patch_w = frame.shape[-2] // self.patch_size, frame.shape[-1] // self.patch_size

            # dpt_features = self.get_dpt_features(frame)
            dpt_features = self.get_dpt_features(frame, input_shape=(B,C,H,W)) 

            pred_depth = self.final_head(dpt_features, patch_h, patch_w)
            pred_depth = torch.clip(pred_depth, min=0)

            preds.append(pred_depth)
        

        return depth2disparity(torch.stack(preds).squeeze(0))
    
    def infer(self, video):
        B, T, C, orig_H, orig_W = video.shape
        H, W = round(orig_H / 14) * 14, round(orig_W / 14) * 14
        video = F.interpolate(
                video.flatten(0, 1), size=(H, W), mode='bilinear', align_corners=False
            ).view(B, T, C, H, W)
        video = (video - self.image_mean) / self.image_std

        return self.forward(video)

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity