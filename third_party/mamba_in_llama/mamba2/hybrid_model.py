from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from transformers.activations import ACT2FN

from mamba2.hybrid_mamba_config import MambaConfig
from mamba2.hybrid_mamba_layer import Mamba2
from mamba_ssm.utils.generation import InferenceParams

import torch
import torch.nn as nn

from rich.console import Console

console = Console()


class MLP(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        rms_norm=True,
        residual_in_fp32=True,
        repeat_kv_before_conv=False,
    ):
        super(MambaDecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mamba = Mamba2(
            d_model=config.d_model,
            d_xb=config.d_xb,
            d_inner=config.d_inner,
            layer_idx=layer_idx,
            repeat_kv_before_conv=repeat_kv_before_conv,
            **config.ssm_cfg,
            **factory_kwargs,
        )
        self.mlp = MLP(config=config)
        if rms_norm:
            self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()
        self.residual_in_fp32 = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        hidden_states: Tensor,
        inference_params: InferenceParams = None,
        *args,
        **kwargs
    ):
        # print(inference_params)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if inference_params is None:
            hidden_states = self.mamba(hidden_states)
        else:
            hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # so here is just to be compatible with Transformer
        if kwargs is None:
            console.log("No kwargs passed to MambaDecoderLayer")
            return (hidden_states, inference_params, None)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            use_cache = kwargs.get("use_cache", False)
            # console.log(f"Use cache: {use_cache}")
            # return (hidden_states, None, None)
            if use_cache is True:
                dummy_keys = torch.ones(
                    1,
                    1,
                    hidden_states.size(1),
                    1,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                dummy_values = torch.ones(
                    1,
                    1,
                    hidden_states.size(1),
                    1,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return (hidden_states, inference_params, past_key_value)
