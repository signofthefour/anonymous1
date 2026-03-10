# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import os
import copy
import torch
import torch.distributed as dist
from copy import deepcopy
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from rich.table import Table
from rich.console import Console

import sys

sys.path.append("./third_party/mamba_in_llama")
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig

console = Console()


# Compute total parameters before and after pruning
def count_parameters_per_child(module, name=""):
    """
    Counts number of parameters directly in each immediate child of the module
    without recursing further, returning a list of tuples (name, num_params).
    """
    param_counts = []
    for child_name, child_module in module.named_children():
        # Construct full name
        full_name = f"{name}.{child_name}" if name else child_name
        # Count all parameters in this child and all of its submodules
        num_params = sum(p.numel() for p in child_module.parameters())
        # convert to billions
        num_params = num_params / 1e9  # Convert to billions
        param_counts.append((full_name, num_params))
    return param_counts


from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter,
    save_model,
    wrap_cuda_model,
    check_modify_and_save_config,
)

from rich.logging import RichHandler
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_args():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument(
        "--train_engine",
        default="torch_ddp",
        choices=["torch_ddp", "deepspeed"],
        help="Engine for paralleled training",
    )
    parser.add_argument("--model", required=True, help="model which will be trained")
    parser.add_argument("--ref_model", required=False, help="ref model used in dpo")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--train_data", required=True, help="train data file")
    parser.add_argument("--cv_data", required=True, help="cv data file")
    parser.add_argument(
        "--qwen_pretrain_path", required=False, help="qwen pretrain path"
    )
    parser.add_argument("--checkpoint", help="checkpoint model")
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument(
        "--tensorboard_dir", default="tensorboard", help="tensorboard log dir"
    )
    parser.add_argument(
        "--ddp.dist_backend",
        dest="dist_backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="distributed backend",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="num of subprocess workers for reading",
    )
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Use pinned memory buffers used for reading",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--dpo",
        action="store_true",
        default=False,
        help="Use Direct Preference Optimization",
    )
    parser.add_argument(
        "--deepspeed.save_states",
        dest="save_states",
        default="model_only",
        choices=["model_only", "model+optimizer"],
        help="save model/optimizer states",
    )
    parser.add_argument(
        "--timeout",
        default=60 * 5,
        type=int,
        help="timeout (in seconds) of cosyvoice_join.",
    )

    # list of layers to be removed
    parser.add_argument(
        "--layers_to_remove",
        type=str,
        default="",
        help="Comma-separated list of layer indices to remove from the model. "
        'For example, "0,1,2" will remove the first three layers. '
        "If empty, no layers will be removed.",
    )
    # list of layers to be distilled
    parser.add_argument(
        "--distill_teacher_layers",
        type=str,
        default="",
        help="Comma-separated list of layer indices to distill from the teacher model. "
        'For example, "0,1,2" will distill the first three layers. '
        "If empty, no layers will be distilled.",
    )

    parser.add_argument(
        "--attn_layers",
        type=str,
        default="",
        help="Comma-separated list of attention layers to be kept",
    )

    parser.add_argument(
        "--weight_init_method",
        type=str,
        default="reused",
        help="Weight initialization method for Mamba layers (kaiming or xavier or reused)",
    )

    parser.add_argument(
        "--hybrid_model",
        default=False,
        action="store_true",
        help="Whether to use hybrid model",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def prepare_student_model(
    transformer_model, config, attn_layers, weight_init_method="reused"
):
    # CosyVoice stuck with head size of 14, which is not supported by Mamba
    # console.log(f"[bold green]{config}")
    # console.log(f"[bold green] weight init method: {weight_init_method}")
    if "head_dim" not in config.to_dict():
        config.head_dim = config.hidden_size // config.num_attention_heads
    d_xb = config.num_key_value_heads * config.head_dim
    d_inner = config.num_attention_heads * config.head_dim
    d_state = config.head_dim

    console.log(f"[bold green]Attention layers to be maintained: {attn_layers}")

    mamba_config = MambaConfig(
        config.hidden_size,
        {"expand": 1, "ngroups": config.num_attention_heads, "d_state": d_state},
        config.rms_norm_eps,
        d_inner=d_inner,
        d_xb=d_xb,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        n_layer=config.num_hidden_layers,
        attn_layers=attn_layers,
    )
    # log mamba config
    # console.print(f"[bold green]Mamba Config: {mamba_config}")
    student = MambaTransformerHybridModelWrapper.init_distillation_from_model(
        None,
        transformer_model,
        mamba_config,
        attn_layers=attn_layers,
        init_with_kqvo=weight_init_method == "reused",
        attn_implementation="flash_attention_2",
        weight_init_method=weight_init_method,
    )
    for name, param in student.named_parameters():
        if "mamba" not in name:
            param.requires_grad = False
    return student
