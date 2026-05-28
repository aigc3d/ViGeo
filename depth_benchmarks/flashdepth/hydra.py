# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.
# Base code from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange, repeat

# try:
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
# except ImportError:
#     RMSNormGated = None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

# from hydra.modules.ops import hydra_split_conv1d_scan_combined


class Hydra(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * (2 * self.ngroups * self.d_state) + 2 * self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * (2 * self.ngroups * self.d_state)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.ones(self.nheads, dtype=torch.float32, device=device)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, inference_params, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        # initial_states = repeat(self.init_states, "... -> b ...", b=2*batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        initial_states, seq_idx = self._get_ssm_states(inference_params, batch*2, seqlen)
        

        if self.use_mem_eff_path:
            return hydra_split_conv1d_scan_combined(
                zxbcdt,
                self.conv1d.weight,
                self.conv1d.bias,
                self.dt_limit,
                self.dt_bias,
                A,
                self.fc_D.weight,
                self.D,
                self.norm.weight,
                self.norm.eps,
                self.out_proj.weight,
                self.out_proj.bias,
                self.chunk_size,
                initial_states,
                seq_idx,
                self.d_inner,
                self.d_state,
                self.headdim,
                self.ngroups,
            )

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * (2 * self.ngroups * self.d_state), 2 * self.nheads],
            dim=-1
        )

        dt = torch.cat((dt[:, :, :self.nheads], torch.flip(dt[:, :, self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt + self.dt_bias)  # (2 * B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, BC = torch.split(xBC, [self.d_inner, 2 * (2 * self.ngroups * self.d_state)], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat(
            (BC[:, :, :2 * self.ngroups * self.d_state],
             torch.flip(BC[:, :, 2 * self.ngroups * self.d_state:], (1,))),
            dim=0
        )
        B, C = torch.split(BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        # import ipdb; ipdb.set_trace()
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=None,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            return_final_states=True,
            **dt_limit_kwargs,
        )

        y, new_states = y
        # initial_states.copy_(new_states)
        inference_params.key_value_memory_dict[self.layer_idx] = new_states


        y = rearrange(y, "b l h p -> b l (h p)")
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))
        y = y_fw + y_bw + x_og * repeat(
            F.linear(x_og, self.fc_D.weight, bias=self.D), "b l h -> b l (h p)", p=self.headdim
        )

        # import ipdb; ipdb.set_trace()

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)

        return out


    def _get_ssm_states(self, inference_params, batch_size, seqlen):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = ssm_state
        else:
            ssm_state = inference_params.key_value_memory_dict[self.layer_idx]

        # if self.layer_idx not in inference_params.seq_idx_dict:
        #     assert inference_params.seqlen_offset == 0
        #     inference_params.seq_idx_dict[self.layer_idx] = torch.arange(seqlen, 
        #                           device=self.in_proj.weight.device, 
        #                           dtype=torch.int32).expand(batch_size, seqlen)
        # else:
        #     assert inference_params.seqlen_offset > 0
        #     inference_params.seq_idx_dict[self.layer_idx] = inference_params.seq_idx_dict[self.layer_idx] + seqlen
        
        return ssm_state, None #inference_params.seq_idx_dict[self.layer_idx]



from mamba_ssm.ops.triton.layernorm_gated import (
    _layer_norm_fwd,
    _layer_norm_bwd,
)
from mamba_ssm.ops.triton.ssd_combined import (
    _mamba_chunk_scan_combined_bwd,
    _mamba_chunk_scan_combined_fwd,
)


def flip(input):  # just for brevity
    return torch.flip(input, dims=(1,))


def chunk_flip_join(input, dim, op):
    input_fwd, input_bwd = input.chunk(2, dim=dim)
    input_bwd = flip(input_bwd)
    if op == "sum":
        return input_fwd + input_bwd
    elif op == "vstack":
        return torch.cat([input_fwd, input_bwd], dim=0)
    elif op == "dstack":
        return torch.cat([input_fwd, input_bwd], dim=-1)
    else:
        raise ValueError()


def dwconv(input, weight, bias):
    return F.conv1d(input.mT, weight, bias, padding="same", groups=input.shape[-1]).mT


def ssm_params(xBC, D_weight, D_bias, d_inner, headdim, ngroups):
    # Split into 3 main branches: X, B, C
    # These correspond to V, K, Q respectively in the SSM/attention duality
    x, BC = xBC.tensor_split([d_inner], dim=-1)
    x_og = x
    x = torch.cat([x, flip(x)], dim=0)
    BC = chunk_flip_join(BC, dim=-1, op="vstack")

    B, C = BC.chunk(2, dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", p=headdim)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    D = repeat(F.linear(x_og, D_weight, D_bias), "b l h -> b l (h p)", p=headdim)

    return x, B, C, D, x_og


class HydraSplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        zxBCdt,
        conv1d_weight,
        conv1d_bias,
        dt_limit,
        dt_bias,
        A,
        D_weight,
        D_bias,
        rmsnorm_weight,
        rmsnorm_eps,
        outproj_weight,
        outproj_bias,
        chunk_size,
        initial_states,
        seq_idx,
        d_inner,
        d_state,
        headdim,
        ngroups,
    ):
        # Infer some other dimensions
        batch, seqlen, _ = zxBCdt.shape
        assert d_inner % headdim == 0
        nheads = d_inner // headdim
        assert nheads % ngroups == 0

        # Check some shapes
        d_xBC = d_inner + 2 * (2 * ngroups * d_state)
        d_conv = conv1d_weight.shape[2]
        assert d_conv % 2 == 1

        assert zxBCdt.shape == (batch, seqlen, d_inner + d_xBC + 2 * nheads)
        assert conv1d_weight.shape == (d_xBC, 1, d_conv)
        assert conv1d_bias.shape == (conv1d_weight.shape[0],)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        assert D_weight.shape == (nheads, d_inner)
        assert D_bias.shape == (D_weight.shape[0],)
        assert rmsnorm_weight.shape == (d_inner,)
        assert outproj_weight.ndim == 2 and outproj_weight.shape[1] == d_inner
        if outproj_bias is not None:
            assert outproj_bias.shape == (outproj_weight.shape[0],)
        if initial_states is not None:
            assert initial_states.shape == (2 * batch, nheads, headdim, d_state)

        # Make contiguous()
        rmsnorm_weight = rmsnorm_weight.contiguous()
        if seq_idx is not None:
            seq_idx = seq_idx.contiguous()

        # Split
        z, xBC_og, dt = torch.split(zxBCdt, [d_inner, d_xBC, 2 * nheads], dim=-1)

        # 1D Convolution
        xBC = F.silu(dwconv(xBC_og, conv1d_weight, conv1d_bias))

        # Flip and rearrange
        x, B, C, D, x_og = ssm_params(
            xBC=xBC,
            D_weight=D_weight,
            D_bias=D_bias,
            d_inner=d_inner,
            headdim=headdim,
            ngroups=ngroups,
        )
        dt = chunk_flip_join(dt, dim=-1, op="vstack")

        scan = _mamba_chunk_scan_combined_fwd(
            x=x, dt=dt, A=A, B=B, C=C,
            chunk_size=chunk_size,
            D=None,
            z=None,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            dt_softplus=True,
            dt_limit=dt_limit,
        )[0]
        scan = rearrange(scan, "b l h p -> b l (h p)")

        y = torch.roll(scan, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y = chunk_flip_join(y, dim=0, op="sum") + (D * x_og)

        # RMSNorm and gate
        u, _, rstd = _layer_norm_fwd(
            x=rearrange(y, "b s d -> (b s) d"),
            z=rearrange(z, "b s d -> (b s) d"),
            weight=rmsnorm_weight,
            bias=None,
            eps=rmsnorm_eps,
            out=None,
            norm_before_gate=True,
            is_rms_norm=True,
        )
        u = rearrange(u, "(b s) d -> b s d", b=batch)

        # Out projection
        ctx.outproj_weight_dtype = outproj_weight.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            u = u.to(dtype)
            outproj_weight = outproj_weight.to(dtype)
            if outproj_bias is not None:
                outproj_bias = outproj_bias.to(dtype)
        out = F.linear(u, outproj_weight, outproj_bias)

        ctx.save_for_backward(
            z, xBC_og, dt,
            scan,
            conv1d_weight,
            conv1d_bias,
            A,
            D_weight,
            D_bias,
            dt_bias,
            initial_states,
            seq_idx,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
        )

        ctx.dt_limit = dt_limit
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.chunk_size = chunk_size
        ctx.d_inner = d_inner
        ctx.d_state = d_state
        ctx.headdim = headdim
        ctx.ngroups = ngroups

        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        (
            z, xBC_og, dt,
            scan,
            conv1d_weight,
            conv1d_bias,
            A,
            D_weight,
            D_bias,
            dt_bias,
            initial_states,
            seq_idx,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
        ) = ctx.saved_tensors

        # Recompute everything except scan
        with torch.enable_grad():
            xBC_og.requires_grad_(True)
            xBC_og_conv = dwconv(xBC_og, conv1d_weight, conv1d_bias)
            xBC = F.silu(xBC_og_conv)

        x, B, C, D, x_og = ssm_params(
            xBC=xBC.detach(),
            D_weight=D_weight,
            D_bias=D_bias,
            d_inner=ctx.d_inner,
            headdim=ctx.headdim,
            ngroups=ctx.ngroups,
        )

        y = torch.roll(scan, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y = chunk_flip_join(y, dim=0, op="sum") + (D * x_og)

        # Compute gradients
        du = F.linear(dout, outproj_weight.T)

        dy, drmsnorm_weight, _, dz, u = _layer_norm_bwd(
            dy=rearrange(du, "b s d -> (b s) d"),
            x=rearrange(y, "b s d -> (b s) d"),
            z=rearrange(z, "b s d -> (b s) d"),
            weight=rmsnorm_weight,
            bias=None,
            eps=ctx.rmsnorm_eps,
            mean=None,
            rstd=rstd,
            norm_before_gate=True,
            is_rms_norm=True,
            recompute_output=True,
        )
        batch = dout.shape[0]
        dy = rearrange(dy, "(b s) d -> b s d", b=batch)
        dz = rearrange(dz, "(b s) d -> b s d", b=batch)
        u = rearrange(u, "(b s) d -> b s d", b=batch)

        doutproj_weight = einsum(dout, u, "b s o, b s d -> o d")
        doutproj_bias = None if (outproj_bias is None) else einsum(dout, "b s d -> d")

        dy_x_og = rearrange(dy * x_og, "b s (h p) -> b s h p", p=ctx.headdim)
        dD_weight = einsum(dy_x_og, x_og, "b s h p, b s d -> h d")
        dD_bias = einsum(dy_x_og, "b s h p -> h")
        dx_og = (D * dy) + einsum(dy_x_og, D_weight, "b s h p, h d -> b s d")

        dy = torch.cat([dy, flip(dy)], dim=0)
        dy[:, 0, :] = 0.0
        dscan = torch.roll(dy, shifts=-1, dims=1)

        scan = rearrange(scan, "b s (h p) -> b s h p", p=ctx.headdim)
        dscan = rearrange(dscan, "b s (h p) -> b s h p", p=ctx.headdim)
        dx, ddt, dA, dB, dC, _, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
            dout=dscan,
            x=x, dt=dt, A=A, B=B, C=C,
            out=scan,
            chunk_size=ctx.chunk_size,
            D=None,
            z=None,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dfinal_states=None,
            seq_idx=seq_idx,
            dt_softplus=True,
            dt_limit=ctx.dt_limit,
        )
        dx, dB, dC = [rearrange(grad, "b l h p -> b l (h p)") for grad in (dx, dB, dC)]

        dx_og = dx_og + chunk_flip_join(dx, dim=0, op="sum")
        dBC = torch.cat([dB, dC], dim=-1)
        dBC = chunk_flip_join(dBC, dim=0, op="dstack")
        ddt = chunk_flip_join(ddt, dim=0, op="dstack")
        dxBC = torch.cat([dx_og, dBC], dim=-1)

        # Autograd seems to be significantly faster than manual differentiation
        dxBC, dconv1d_weight, dconv1d_bias = torch.autograd.grad(
            outputs=[xBC],
            inputs=[xBC_og, conv1d_weight, conv1d_bias],
            grad_outputs=[dxBC],
        )

        dzxBCdt = torch.cat([dz, dxBC, ddt], dim=-1)

        return (
            dzxBCdt,
            dconv1d_weight,
            dconv1d_bias,
            None,
            ddt_bias,
            dA,
            dD_weight,
            dD_bias,
            drmsnorm_weight,
            None,
            doutproj_weight,
            doutproj_bias,
            None,
            dinitial_states,
            None,
            None,
            None,
            None,
            None,
        )


hydra_split_conv1d_scan_combined = HydraSplitConv1dScanCombinedFn.apply