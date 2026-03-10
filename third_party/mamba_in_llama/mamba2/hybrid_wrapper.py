# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM

from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.utils.hub import cached_file

from mamba2.hybrid_model import MambaDecoderLayer
from mamba2.hybrid_mamba_config import MambaConfig

from mamba2.util import load_safetensors_to_dict

from rich.console import Console

console = Console()

MAMBA_CONFIG_NAME = "mamba_config.json"


def load_config(path, filename=MAMBA_CONFIG_NAME):
    # get the parent directory of the path
    path = os.path.dirname(path)
    config_path = os.path.join(path, filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config = json.load(open(config_path))
    mamba_config = MambaConfig(**config)
    return mamba_config


class MambaTransformerHybridModelWrapper(nn.Module):

    def __init__(
        self,
        checkpoint_path,
        transformer_model,
        mamba_config,
        attn_layers,
        dtype,
        init_with_kqvo,
        weight_init_method="reused",
        load_from_hub=False,
        attn_implementation="default",
        **kwargs,
    ):
        super(MambaTransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.model = transformer_model
        self.config = self.model.config
        self.embed_tokens = self.model.embed_tokens

        for layer_idx in range(mamba_config.n_layer):
            if layer_idx not in attn_layers:
                mamba_encoder = MambaDecoderLayer(
                    mamba_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )

                self.model.layers[layer_idx] = mamba_encoder

        # self.layers = self.model.layers
        self.model = self.model.to(dtype).cuda()

        model_dtype = torch.float32  # Default to float32

        def _convert_dtype(module):
            for param in module.parameters(recurse=False):
                if param.dtype != model_dtype:
                    param.data = param.data.to(model_dtype)
            for buffer in module.buffers(recurse=False):
                if buffer.dtype != model_dtype:
                    buffer.data = buffer.data.to(model_dtype)
            for child in module.children():
                _convert_dtype(child)  # Recurse

        _convert_dtype(self.model)

    def allocate_mamba_inference_cache(
        self, batch_size, max_seqlen, dtype=None, **kwargs
    ):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.model.layers)
            if isinstance(layer, MambaDecoderLayer)
        }

    def forward(
        self,
        inputs_embeds,
        **kwargs,
    ):

        # Especially for CosyVoice (Qwen)l
        # TODO: this part might effect the training process. Keep an eye on it.

        model_dtype = torch.float32  # Default to float32

        # if inputs_embeds is not None:
        # model_dtype = next(self.model.parameters()).dtype  # Or self.model.dtype
        inputs_embeds = inputs_embeds.to(model_dtype)

        # check if use_cache in kwargs
        is_inference = kwargs.get("use_cache", False)
        if not is_inference:
            if "use_cache" in kwargs:
                kwargs["use_cache"] = False

        self.model.to(model_dtype)  # Final top-level cast for any missed items
        return self.model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(
        self,
        inputs_embeds,
        max_length=2048,
        eos_token_id=None,
        do_sample=True,
        top_p=1,
        temperature=0.8,
        **kwargs,
    ):
        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=False,
            **kwargs,
        )
        # print(output)
        return output

    @staticmethod
    def init_distillation(
        checkpoint_path,
        transformer_path,  # Here, i
        mamba_config,
        attn_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_kqvo=True,
        **kwargs,
    ):
        # transformer_model = AutoModelForCausalLM.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)
        transformer_model = AutoModelForCausalLM.from_pretrained(
            transformer_path, torch_dtype=dtype, attn_implementation=attn_implementation
        )
        transformer_model.train()
        return MambaTransformerHybridModelWrapper(
            checkpoint_path,
            transformer_model,
            mamba_config,
            attn_layers,
            dtype,
            init_with_kqvo,
        )

    @staticmethod
    def init_distillation_from_model(
        checkpoint_path,
        transformer_model,
        mamba_config,
        attn_layers,
        weight_init_method="reused",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_kqvo=True,
        **kwargs,
    ):
        transformer_model.train()
        return MambaTransformerHybridModelWrapper(
            checkpoint_path,
            transformer_model,
            mamba_config,
            attn_layers,
            dtype,
            init_with_kqvo,
            attn_implementation=attn_implementation,
            weight_init_method=weight_init_method,
        )

    @staticmethod
    def from_pretrained_local(
        pretrained_model_name,
        teacher_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):

        # config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
        # get parent dir of pretrained_model_name
        mamba_config_dir = os.path.dirname(pretrained_model_name)
        with open(f"{mamba_config_dir}/{MAMBA_CONFIG_NAME}", "r") as json_file:
            config_dict = json.load(json_file)
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(
            pretrained_model_name,
            transformer_model,
            mamba_config,
            mamba_config.attn_layers,
            torch_dtype,
            init_with_kqvo=False,
            load_from_hub=True,
        )

    @staticmethod
    def from_pretrained_hub(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(
            config_data["_name_or_path"],
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        resolved_archive_file = cached_file(
            pretrained_model_name,
            MAMBA_CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        config_dict = json.load(open(resolved_archive_file))
        mamba_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(
            pretrained_model_name,
            transformer_model,
            mamba_config,
            mamba_config.attn_layers,
            torch_dtype,
            init_with_kqvo=False,
            load_from_hub=True,
        )

    @staticmethod
    def from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ):
        if os.path.exists(pretrained_model_name):
            return MambaTransformerHybridModelWrapper.from_pretrained_local(
                pretrained_model_name, torch_dtype, attn_implementation
            )
        else:
            return MambaTransformerHybridModelWrapper.from_pretrained_hub(
                pretrained_model_name, torch_dtype, attn_implementation
            )

    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "mamba_config.json")
        with open(config_path, "w") as f:
            json.dump(self.mamba_config.__dict__, f, indent=4)
