import torch
import torch.nn as nn
import torch.nn.functional as F
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
from einops import rearrange
import logging

def soft_cap(values: torch.Tensor, cap_value: float | torch.Tensor | None) -> torch.Tensor:
    """
    Soft caps a tensor to a value.

    Performs a tanh operation on the logits and scales the result to the cap value. Common technique in attention
    and output language heads to prevent large logits from dominating the softmax. See for example Gemma2:
    https://arxiv.org/abs/2408.00118

    Args:
        values: The tensor to cap.
        cap_value: The value to cap the values to. If None, no cap is applied.

    Returns:
        The capped values.
    """
    if cap_value is None:
        return values
    return cap_value * torch.tanh(values / cap_value)

class xLSTMModel(nn.Module):
    def __init__(self, vocab_size, training_mode=True, **kwargs):
        super().__init__()

        '''
        vocab_size: input and output dimensions 
        embedding_dim: hidden dimensions
        '''

        self.mode = "inference" if not training_mode else "train"

        xlstm_config = xLSTMLargeConfig(
            embedding_dim=kwargs['mamba_d_state'],
            num_heads=4,
            num_blocks=kwargs['num_mamba_layers'],
            vocab_size=vocab_size,
            return_last_states=True,
            mode=self.mode,
            chunkwise_kernel="chunkwise--triton_xl_chunk", # xl_chunk == TFLA kernels
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            inference_state_dtype= "bfloat16" #The dtype to use for the state tensors in inference mode
        )
        self.xlstm = xLSTMLarge(xlstm_config)

        logging.info(f"xlstm mode: {'inference' if not training_mode else 'train'}")

        self.embedding = nn.Linear(vocab_size, kwargs['mamba_d_state'])

        # init self.xlstm.lm_head to zero
        self.xlstm.lm_head.weight.data.zero_()
        self.state = None
        self.mamba_type = kwargs['mamba_type']
        
        
        # assuming downsample factor is <=0.25
        if kwargs.get('use_upconv', False):
            self.upconv = nn.Sequential(
                nn.Conv2d(vocab_size, vocab_size, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(vocab_size, vocab_size, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(vocab_size, vocab_size, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                # last interpolation is done outside and matches dpt features
            )
        else:
            self.upconv = nn.Identity()

        self.final_layer = nn.Identity()
        
        

    def start_new_sequence(self):
        self.state = None

    def forward(self, x):
        return self.xlstm(x)


    def forward_single_frame(self, frame, **kwargs):

        downsampled = kwargs['downsample_factor'] != 1.0

        x = frame.clone()

        B, L, D = x.shape
        if self.mode == "train":
            # sequence length must be divisible by 64
            if L % 64:                  
                x = F.pad(x, (0, 0, 0, 64 - L % 64))

        x = self.embedding(x)
        x, state = self.xlstm.backbone(x, self.state)
        x = x[:, :L, :]
        logits = self.xlstm.lm_head(x)
        out = soft_cap(logits, self.xlstm.config.output_logit_soft_cap)

        self.state = state

        if self.mamba_type == 'add' and not downsampled:
            out = out + frame

        if self.mamba_type == 'add' and downsampled:
            h,w = int(kwargs['dpt_shape'][0]*kwargs['downsample_factor']), int(kwargs['dpt_shape'][1]*kwargs['downsample_factor'])
            spatial_out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
            out = self.upconv(spatial_out)
           
        return out




if __name__ == "__main__":
    # configure the model with TFLA Triton kernels
    xlstm_config = xLSTMLargeConfig(
        embedding_dim=512,
        num_heads=4,
        num_blocks=6,
        vocab_size=2048,
        return_last_states=True,
        mode="inference",
        chunkwise_kernel="chunkwise--triton_xl_chunk", # xl_chunk == TFLA kernels
        sequence_kernel="native_sequence__triton",
        step_kernel="triton",
        )
    # instantiate the model
    xlstm = xLSTMLarge(xlstm_config)
    xlstm = xlstm.to("cuda")
    # create inputs
    input = torch.randint(0, 2048, (3, 256)).to("cuda")
    # run a forward pass
    out = xlstm(input)
    out, state = out
    import ipdb; ipdb.set_trace()
    out.shape[1:] == (256, 2048)