import torch
import torch.nn as nn
import torch.nn.functional as F
from .helpers import Interpolate
from .dinov2_layers.patch_embed import PatchEmbed


class RefinementHead(nn.Module):
    def __init__(self,
                 dpt_dim: int = 256,
                 ref_dim: int = 1024,
                 resize_factor: int = 4,
                 new_tokenizer: bool = False):
        """
        dpt_features: B, dpt_dim, 288, 288 (assuming a fixed 504 base res)
        ref_features: B, ref_dim, h, w (where h,w = high_res / 14); e.g. 72 for 1008
        """
        super().__init__()

        if new_tokenizer:
            self.patch_embed = PatchEmbed(img_size=[1120,2016], patch_size=14, in_chans=3, embed_dim=ref_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, ref_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)


        self.ref_proj = nn.Conv2d(ref_dim, dpt_dim, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.ref_proj.bias, 0.05)
        
        self.ref_upconv1 = nn.Sequential(
            Interpolate(scale_factor=2),
            nn.SiLU(),
            nn.Conv2d(dpt_dim, dpt_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(dpt_dim, dpt_dim, kernel_size=3, stride=1, padding=1),
            # add
            
        )
        nn.init.constant_(self.ref_upconv1[1].bias, 0.05)
        
        self.ref_upconv2 = nn.Sequential(
            Interpolate(scale_factor=2),
            nn.Conv2d(dpt_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        nn.init.constant_(self.ref_upconv2[1].bias, 0.05)

        self.ref_upconv3 = nn.Sequential(
            Interpolate(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        nn.init.constant_(self.ref_upconv3[1].bias, 0.05)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dpt_dim*2, dpt_dim, kernel_size=1, padding=1),
            nn.SiLU(),
        )
        nn.init.constant_(self.conv1[0].bias, 0.05)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dpt_dim+64, 64, kernel_size=1, padding=1),
            nn.SiLU(),
        )
        nn.init.constant_(self.conv2[0].bias, 0.05)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=1, padding=1),
            nn.SiLU(),
        )
        nn.init.constant_(self.conv3[0].bias, 0.05)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        nn.init.constant_(self.final_conv[0].bias, 0.1)
        nn.init.constant_(self.final_conv[2].bias, 0.2)
        

    def forward(self, dpt_output, ref_output, patch_h, patch_w):
        """
        dpt_features: B, dpt_dim, 160, 288 (assuming a fixed 280x504 base res)
        ref_features: B, ref_dim, h, w (where h,w = high_res / 14); e.g. 80x144 for 1120x2016
        """
        
        assert dpt_output.ndim == 4, "dpt_output should already be spatial format"
        assert ref_output.ndim == 3, "ref output should still be sequence format"


        ref_output = self.reshape_to_spatial(ref_output, patch_h, patch_w)
        
        ref_output = self.ref_proj(ref_output)
        ref_up1 = self.ref_upconv1(ref_output)
        ref_up2 = self.ref_upconv2(ref_up1)
        ref_up3 = self.ref_upconv3(ref_up2)

        combined_features = torch.cat([dpt_output, ref_up1], dim=1)
        combined_features = self.conv1(combined_features)
        combined_features = F.interpolate(combined_features, size=(ref_up2.shape[2], ref_up2.shape[3]), mode='bilinear', align_corners=True)
    
        combined_features = torch.cat([combined_features, ref_up2], dim=1)
        combined_features = self.conv2(combined_features)
        combined_features = F.interpolate(combined_features, size=(ref_up3.shape[2], ref_up3.shape[3]), mode='bilinear', align_corners=True)

        combined_features = torch.cat([combined_features, ref_up3], dim=1)
        combined_features = self.conv3(combined_features)        
        out = F.interpolate(combined_features, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        
        out = self.final_conv(out)

        return out.squeeze(1)


    def reshape_to_spatial(self, x, patch_h, patch_w, spatial_to_sequence=False):
        if spatial_to_sequence:
            x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        else:
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
        return x

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        import ipdb; ipdb.set_trace()
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x


    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
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