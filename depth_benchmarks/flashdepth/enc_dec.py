import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .dinov2_layers.patch_embed import PatchEmbed
import math


class EncoderDecoder(nn.Module):
    def __init__(self, 
                use_encoder,
                use_tokenizer,
                use_skip_connections,
                patch_size,
                input_resolution,
                dpt_dim=256,
                ref_dim=1024,         
                downsample_factor=4.0
                ):
        super().__init__()

        assert not (use_tokenizer and use_encoder), "Cannot use both tokenizer and encoder"

        self.ref_paths = None
        self.input_resolution = input_resolution # h,w

        num_blocks = int(math.log2(downsample_factor)) # one block = 2x downsample
        self.num_blocks = num_blocks
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        enc_block_channels = [3]
        dec_block_channels = [128]

        for i in range(num_blocks):
            if use_skip_connections and use_encoder:
                enc_block_channels.append(64*(i+1))
                self.enc_blocks.append(ResidualConvUnit(enc_block_channels[i], enc_block_channels[i+1]))
            else:
                enc_block_channels.append(0)
        
        for i in range(num_blocks):
            dec_block_channels.append(128)
            self.dec_blocks.append(ResidualConvUnit(dec_block_channels[i]+enc_block_channels[-1-i], dec_block_channels[i+1]))
            


        if use_tokenizer:
            assert not use_skip_connections, "Not implemented skip + tokenizer yet"
            enc_block_channels = [0,0]
            self.patch_size = patch_size
            self.patch_embed = PatchEmbed(img_size=input_resolution, patch_size=patch_size, in_chans=3, embed_dim=ref_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, ref_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.use_skip_connections = use_skip_connections
        self.use_tokenizer = use_tokenizer
        self.use_encoder = use_encoder

        # self.final_conv_500 = nn.Identity() # replace with dpt output_conv2
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        nn.init.constant_(self.final_conv[0].bias, 0.1)
        nn.init.constant_(self.final_conv[2].bias, 0.2)

        logging.info(f"Encoder decoder skip connections: {self.use_skip_connections}; tokenizer: {self.use_tokenizer}; encoder: {self.use_encoder}")

        # init_as_identity_resblock(self.dec_block1)
        # init_as_identity_resblock(self.dec_block2)
        

    def forward(self, x):
        
        pass 



    def encode(self, x):
        """
        x: input tensor of shape (T, C, H, W)
        Returns: list of feature maps processed in batches of size B
        """
        T, C, H, W = x.shape
        B = min(T, 2)
        assert T % B == 0, f"T ({T}) must be divisible by batch size ({B})"
        
        # Lists to store path1 and path2 from all batches
        all_paths = [[] for _ in range(self.num_blocks)]
        
        
        for i in range(0, T, B):
            batch_x = x[i:i+B]
            
            # Process current batch
            for i in range(self.num_blocks):
                batch_x = self.enc_blocks[i](batch_x)
                batch_x = F.avg_pool2d(batch_x, kernel_size=2, stride=2)
                all_paths[i].append(batch_x)
        
        # Combine results from all batches
        self.ref_paths = [
            torch.cat(all_paths[i], dim=0) for i in range(self.num_blocks)
        ]
        return self.ref_paths
    
    def decode(self, x):
        """
        x: dpt_features after outconv1 + interpolate -> (T, C, H//4, W//4)
        """
        T = x.shape[0]
        B = min(T, 2)
        assert T % B == 0, f"T ({T}) must be divisible by batch size ({B})"
        
        outputs = []
        for i in range(0, T, B):
            batch_x = x[i:i+B]
            if self.use_skip_connections:
                batch_ref_paths = [path[i:i+B] for path in self.ref_paths]
            
            combined_features = batch_x
            
            for j in range(self.num_blocks):
                if self.use_skip_connections:
                    # Concatenate with corresponding encoder features
                    # Note: we use -1-j to access encoder features in reverse order
                    combined_features = torch.cat([combined_features, batch_ref_paths[-1-j]], dim=1)
                
                combined_features = self.dec_blocks[j](combined_features)
                combined_features = F.interpolate(combined_features, scale_factor=2, mode="bilinear", align_corners=False)
            
            out = self.final_conv(combined_features)
            outputs.append(out)
        
        self.ref_paths = None
        final_output = torch.cat(outputs, dim=0)
        return final_output.squeeze(1)


    def prepare_tokens(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, h, w)

        return x


    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        # class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed#[:, 1:]
        dim = x.shape[-1]


        orig_h, orig_w  = self.input_resolution
        W_orig = orig_w // self.patch_size  # 136
        H_orig = orig_h // self.patch_size  # 76
        patch_pos_embed = patch_pos_embed.reshape(1, H_orig, W_orig, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # [1, dim, H_orig, W_orig]


        W_new = w // self.patch_size  # e.g. 1008//14=72
        H_new = h // self.patch_size  # e.g. 392//14=28

        patch_pos_embed = F.interpolate(
            patch_pos_embed, 
            size=(H_new, W_new),    # (28,72)
            mode="bicubic",
            align_corners=False
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)     # [1, H_new, W_new, dim]
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)     # [1, H_new*W_new, dim]
        return patch_pos_embed.to(previous_dtype)
        # return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


class ResidualConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = nn.SiLU()

        self.skip_proj = nn.Identity()
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        


    def forward(self, x):
        
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = out + self.skip_proj(x)

        return out

def init_as_identity_resblock(m: nn.Module):
    """Zero out the last conv's weights so the block initially behaves like skip_proj(x)."""
    if isinstance(m, ResidualConvUnit):
        # 1. Normal init for the first conv
        nn.init.kaiming_normal_(m.conv1.weight, nonlinearity='relu')
        if m.conv1.bias is not None:
            # nn.init.zeros_(m.conv1.bias)
            nn.init.constant_(m.conv1.bias, 0.1)

        # 2. Zero out the last conv
        nn.init.zeros_(m.conv2.weight)
        if m.conv2.bias is not None:
            # nn.init.zeros_(m.conv2.bias)
            nn.init.constant_(m.conv2.bias, 0.2)

        # 3. For skip_proj, if it's a conv, optionally make it near-identity
        if isinstance(m.skip_proj, nn.Conv2d):
            # simplest: kaiming init
            nn.init.kaiming_normal_(m.skip_proj.weight, nonlinearity='relu')