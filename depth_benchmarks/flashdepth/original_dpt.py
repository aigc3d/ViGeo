import torch
import torch.nn as nn
import torch.nn.functional as F

from .util.blocks import FeatureFusionBlock, _make_scratch



def _make_fusion_block(dpt_dim, use_bn, size=None):
    return FeatureFusionBlock(
        dpt_dim,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        dpt_dim=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        patch_size=14,
        **kwargs
    ):
        super(DPTHead, self).__init__()
        
        self.patch_size = patch_size
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            dpt_dim,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(dpt_dim, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(dpt_dim, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(dpt_dim, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(dpt_dim, use_bn)
        
        head_dim_1 = dpt_dim
        head_dim_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_dim_1, head_dim_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_dim_1 // 2, head_dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(head_dim_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Identity(),
        )
    
    def forward(self, encoder_features, patch_h, patch_w, 
                return_dpt_features=False, temporal_features=None,
                return_features_only=True, **kwargs
    ):
        '''
        clstoken is False (dinov2 intermediate layers are set to True)
        1. projections with no change in dimensions 
        2. resize layers: if original shape is [32,74] (as in Sintel with patch 14), then the four layers have spatial dimensions x4,x2,x1,x0.5; channel dimensions increase
        3. self.scratch.layer1_rn: conv to same channel dims of 256, so change to spatial
        4. self.scratch.refinenet: second input is passed through a conv unit, then added with the first input; then passed through another conv unit and interpolated to the next layer dimensions
        5. two conv layers to get final output; final channel dimension is 1 (depth)
        '''
        out = []
        for i, x in enumerate(encoder_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        # without upsampling or out_channels, layer1 to 4 are all [B, 256, 36, 36]
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # layeri_rn only changes all channels to 256, no spatial change

    
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)


        if return_features_only:
            return path_1

        if temporal_features is not None:
            path_1 = path_1 + temporal_features
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        if return_dpt_features:
            return out, path_1
        else:
            return out


    def forward_with_mamba(self, encoder_features, patch_h, patch_w, temporal_layer, mamba_fn, **kwargs):
     
        out = []
        for i, x in enumerate(encoder_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        # without upsampling or out_channels, layer1 to 4 are all [B, 256, 36, 36]
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        if kwargs.get('fused_path4') is not None:
            path_4 = kwargs['fused_path4']
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])    
        
        if 0 in temporal_layer:
            path_4 = mamba_fn(kwargs['shape_placeholder'], path_4, in_dpt_layer=temporal_layer.index(0))
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        if 1 in temporal_layer:
            path_3 = mamba_fn(kwargs['shape_placeholder'], path_3, in_dpt_layer=temporal_layer.index(1))
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        if 2 in temporal_layer:
            path_2 = mamba_fn(kwargs['shape_placeholder'], path_2, in_dpt_layer=temporal_layer.index(2))
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        if 3 in temporal_layer:
            path_1 = mamba_fn(kwargs['shape_placeholder'], path_1, in_dpt_layer=temporal_layer.index(3))


        return path_1


    def get_forward_features(self, encoder_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(encoder_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        # without upsampling or out_channels, layer1 to 4 are all [B, 256, 36, 36]
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # layeri_rn only changes all channels to 256, no spatial change
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return [path_4, path_3, path_2, path_1]


    def get_path4(self, encoder_features, patch_h, patch_w):
        # Only process the last two encoder features (index 2 and 3)
        out = []
        for i in range(2, 4):  # Only process indices 2 and 3
            x = encoder_features[i]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        # Unpack only the needed layers
        layer_3, layer_4 = out

        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        return path_4


    # hybrid fusion and mamba in the same function
    # currently have hard coded only path 4 to take features from teacher
    def get_fused_features(self, encoder_features, teacher_dpt_features, patch_h, patch_w, fuse_fn,
                            temporal_layer, mamba_fn, **kwargs
                        ):
        '''
        all teacher_dpt_features are [B, 256, h', w'] (h, w=296 if image size is 518)
        all DPT-S dpt features are [B, 64, h', w']

        fuse_fn is DistillFusion.forward(); passing in from outside
        '''
        out = []
        for i, x in enumerate(encoder_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:]) 

        # path_4.shape, teacher_dpt_features[0].shape = [1, 64, 82, 146], [1, 256, 37, 65]
        path_4 = fuse_fn(path_4, teacher_dpt_features[0], path_idx=0)  

        if 0 in temporal_layer:
            path_4 = mamba_fn(kwargs['shape_placeholder'], path_4, in_dpt_layer=temporal_layer.index(0))   
        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        if 1 in temporal_layer:
            path_3 = mamba_fn(kwargs['shape_placeholder'], path_3, in_dpt_layer=temporal_layer.index(1))

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        if 2 in temporal_layer:
            path_2 = mamba_fn(kwargs['shape_placeholder'], path_2, in_dpt_layer=temporal_layer.index(2))
        
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        if 3 in temporal_layer:
            path_1 = mamba_fn(kwargs['shape_placeholder'], path_1, in_dpt_layer=temporal_layer.index(3))
        
        return path_1

