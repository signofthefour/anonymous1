# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import torch
import json
import re
import datetime
import yaml

import deepspeed
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live,
)

from cosyvoice.dataset.dataset import Dataset
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR
import torch.nn.functional as F


def skewed_reverse_kl(logits, teacher_logits, mask, lam=0.1):
    # borrowed from https://github.com/jongwooko/distillm/blob/master/distillm/losses.py
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    logging.info(
        "training on multiple gpus, this gpu {}".format(local_rank)
        + ", rank {}, world_size {}".format(rank, world_size)
    )
    if args.train_engine == "torch_ddp":
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    else:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan, dpo):
    data_pipeline = (
        configs["data_pipeline_gan"] if gan is True else configs["data_pipeline"]
    )
    train_dataset = Dataset(
        args.train_data,
        data_pipeline=data_pipeline,
        mode="train",
        gan=gan,
        dpo=dpo,
        shuffle=True,
        partition=True,
    )
    cv_dataset = Dataset(
        args.cv_data,
        data_pipeline=data_pipeline,
        mode="train",
        gan=gan,
        dpo=dpo,
        shuffle=False,
        partition=False,
    )

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
    )
    cv_data_loader = DataLoader(
        cv_dataset,
        batch_size=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
    )
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        configs["train_conf"]["dtype"] = "fp32"
    else:
        with open(args.deepspeed_config, "r") as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs["train_conf"]["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs["train_conf"]["dtype"] = "bf16"
        else:
            configs["train_conf"]["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        # if use deepspeed, override ddp config
        configs["train_conf"]["save_per_step"] = int(
            configs["train_conf"]["save_per_step"]
            * configs["train_conf"]["accum_grad"]
            / ds_configs["gradient_accumulation_steps"]
        )
        configs["train_conf"]["accum_grad"] = ds_configs["gradient_accumulation_steps"]
        configs["train_conf"]["grad_clip"] = ds_configs["gradient_clipping"]
        configs["train_conf"]["log_interval"] = ds_configs["steps_per_print"]
    return configs


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert torch.cuda.is_available()
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )
    else:
        if int(os.environ.get("RANK", 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size,
            )
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):
    if gan is False:
        if configs["train_conf"]["optim"] == "adam":
            optimizer = optim.Adam(
                model.parameters(), **configs["train_conf"]["optim_conf"]
            )
        elif configs["train_conf"]["optim"] == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), **configs["train_conf"]["optim_conf"]
            )
        else:
            raise ValueError("unknown optimizer: " + configs["train_conf"])

        if configs["train_conf"]["scheduler"] == "warmuplr":
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs["train_conf"]["scheduler_conf"])
        elif configs["train_conf"]["scheduler"] == "NoamHoldAnnealing":
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(
                optimizer, **configs["train_conf"]["scheduler_conf"]
            )
        elif configs["train_conf"]["scheduler"] == "constantlr":
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs["train_conf"])

        # use deepspeed optimizer for speedup
        if args.train_engine == "deepspeed":

            def scheduler(opt):
                return scheduler_type(opt, **configs["train_conf"]["scheduler_conf"])

            model, optimizer, _, scheduler = deepspeed.initialize(
                args=args,
                model=model,
                optimizer=None,
                lr_scheduler=scheduler,
                model_parameters=model.parameters(),
            )

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model, so we cannot use deepspeed
        if configs["train_conf"]["optim"] == "adam":
            optimizer = optim.Adam(
                model.module.generator.parameters(),
                **configs["train_conf"]["optim_conf"]
            )
        elif configs["train_conf"]["optim"] == "adamw":
            optimizer = optim.AdamW(
                model.module.generator.parameters(),
                **configs["train_conf"]["optim_conf"]
            )
        else:
            raise ValueError("unknown optimizer: " + configs["train_conf"])

        if configs["train_conf"]["scheduler"] == "warmuplr":
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs["train_conf"]["scheduler_conf"])
        elif configs["train_conf"]["scheduler"] == "NoamHoldAnnealing":
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(
                optimizer, **configs["train_conf"]["scheduler_conf"]
            )
        elif configs["train_conf"]["scheduler"] == "constantlr":
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs["train_conf"])

        if configs["train_conf"]["optim_d"] == "adam":
            optimizer_d = optim.Adam(
                model.module.discriminator.parameters(),
                **configs["train_conf"]["optim_conf"]
            )
        elif configs["train_conf"]["optim_d"] == "adamw":
            optimizer_d = optim.AdamW(
                model.module.discriminator.parameters(),
                **configs["train_conf"]["optim_conf"]
            )
        else:
            raise ValueError("unknown optimizer: " + configs["train_conf"])

        if configs["train_conf"]["scheduler_d"] == "warmuplr":
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(
                optimizer_d, **configs["train_conf"]["scheduler_conf"]
            )
        elif configs["train_conf"]["scheduler_d"] == "NoamHoldAnnealing":
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(
                optimizer_d, **configs["train_conf"]["scheduler_conf"]
            )
        elif configs["train_conf"]["scheduler"] == "constantlr":
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs["train_conf"])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args):
    writer = None
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer


def save_model(model, model_name, info_dict):
    rank = int(os.environ.get("RANK", 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, "{}.pt".format(model_name))

    if info_dict["train_engine"] == "torch_ddp":
        if rank == 0:
            torch.save(
                {
                    **model.module.state_dict(),
                    "epoch": info_dict["epoch"],
                    "step": info_dict["step"],
                },
                save_model_path,
            )
    else:
        with torch.no_grad():
            model.save_checkpoint(
                save_dir=model_dir, tag=model_name, client_state=info_dict
            )
    if rank == 0:
        info_path = re.sub(".pt$", ".yaml", save_model_path)
        info_dict["save_time"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(info_path, "w") as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info(
            "[Rank {}] Checkpoint: save to checkpoint {}".format(rank, save_model_path)
        )


def cosyvoice_join(group_join, info_dict):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    if info_dict["batch_idx"] != 0:
        # we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
        try:
            dist.monitored_barrier(
                group=group_join, timeout=group_join.options._timeout
            )
            return False
        except RuntimeError as e:
            logging.info(
                "Detected uneven workload distribution: {}\n".format(e)
                + "Break current worker to manually join all workers, "
                + "world_size {}, current rank {}, current local_rank {}\n".format(
                    world_size, rank, local_rank
                )
            )
            return True
    else:
        return False


def batch_forward_with_distill(
    student_model,
    teacher_model,
    batch,
    scaler,
    info_dict,
    temperature=2.0,
    alpha=0.5,
    distill_teacher_layers=None,
    distill_attention=True,
):
    device = int(os.environ.get("LOCAL_RANK", 0))
    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    # Enable mixed precision if applicable
    autocast = torch.amp.autocast(
        enabled=scaler is not None, device_type="cuda", dtype=dtype, cache_enabled=False
    )
    with autocast:
        # Run teacher forward pass without gradient
        with torch.no_grad():
            teacher_outputs = teacher_model(
                batch,
                device,
                output_hidden_states=True,
                output_attentions=distill_attention,
            )
            teacher_hidden_states = teacher_outputs["hidden_states"]
            # teacher_attentions = teacher_outputs["attentions"]
            teacher_attentions = teacher_outputs.get(
                "attentions", None
            )  # attentions might not be available
            # Extract teacher embeddings and layer outputs
            teacher_embeddings = teacher_hidden_states[0]  # embedding output
            teacher_layer_outputs = teacher_hidden_states[1:]  # layers outputs

        # Run student forward pass
        with torch.amp.autocast(device_type="cuda", dtype=dtype, cache_enabled=False):
            student_outputs = student_model(
                batch,
                device,
                output_hidden_states=True,
                output_attentions=distill_attention,
            )
        student_hidden_states = student_outputs["hidden_states"]
        student_attentions = student_outputs.get(
            "attentions", None
        )  # attentions might not be available
        # student_attentions = student_outputs.attentions
        student_embeddings = student_hidden_states[0]
        student_layer_outputs = student_hidden_states[1:]
        reverse_kl_mask = student_outputs["lm_output_mask"]
        # Compute cross-entropy loss as primary supervision
        ce_loss = (
            student_outputs["loss"]
            if student_outputs["loss"] is not None
            else torch.tensor(0.0, device=device)
        )

        # Compute embedding-level distillation loss
        embedding_loss = F.mse_loss(
            student_embeddings, teacher_embeddings, reduction="mean"
        )

        # embedding_loss = 0.0

        # Compute hidden-layer distillation loss (mean over layers)
        # layer_losses = []
        # # min_layers = min(len(teacher_layer_outputs), len(student_layer_outputs))
        # assert len(distill_teacher_layers) == len(
        #     student_layer_outputs
        # ), "Teacher and student layer outputs must have the same length"
        # for student_layer_idx, teacher_layer_idex in enumerate(distill_teacher_layers):
        #     s_feat = student_layer_outputs[student_layer_idx]
        #     t_feat = teacher_layer_outputs[teacher_layer_idex]
        #     layer_losses.append(F.mse_loss(s_feat, t_feat, reduction="mean"))

        # Compute attention distillation loss if applicable
        if student_attentions is not None and teacher_attentions is not None:
            attention_losses = []
            for student_layer_idx, teacher_layer_idex in enumerate(
                distill_teacher_layers
            ):
                s_attn = student_attentions[student_layer_idx]
                t_attn = teacher_attentions[teacher_layer_idex]
                attention_losses.append(F.mse_loss(s_attn, t_attn, reduction="mean"))
            # Average attention losses if any
            attention_loss = (
                torch.stack(attention_losses).mean()
                if attention_losses
                else torch.tensor(0.0, device=device)
            )
        else:
            attention_loss = torch.tensor(0.0, device=device)
        # attention_loss = torch.tensor(0.0, device=device)  # Placeholder for attention loss
        # Average layer losses if any
        # layer_loss = (
        #     torch.stack(layer_losses).mean()
        #     if layer_losses
        #     else torch.tensor(0.0, device=device)
        # )
        layer_loss = 0.0

        # Compute logits distillation loss
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        # logits_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
        # logits_loss = skewed_reverse_kl(
        #     student_logits, teacher_logits, reverse_kl_mask, lam=0.1
        # )
        logits_loss = 0.0

        # Combine losses
        total_loss = alpha * ce_loss + (1 - alpha) * (
            0.25 * embedding_loss
            + 0.25 * layer_loss
            + 0.25 * logits_loss
            + 0.25 * attention_loss
        )
        # print("Type of total_loss:", total_loss.dtype, flush=True)
        # Convert total_loss to bfloat16 if using bf16 training
        # if dtype == torch.bfloat16:
        # total_loss = total_loss.to(torch.bfloat16)

        # print("Type of total_loss after bf16 conversion:", total_loss.dtype, flush=True)

        # Package results into info_dict for logging
        # print("CE Loss: {:.4f}, Embedding Loss: {:.4f}, Layer Loss: {:.4f}, Logits Loss: {:.4f}, Attention Loss: {:.4f}, Total Loss: {:.4f}".format(
        #     ce_loss.item(),
        #     embedding_loss.item(),
        #     layer_loss.item(),
        #     logits_loss.item(),
        #     attention_loss.item(),
        #     total_loss.item()
        # ))
        info_dict["loss_dict"] = {
            "loss": total_loss,
            "acc": (
                student_outputs["acc"]
                if "acc" in student_outputs
                else torch.tensor(0.0, device=device)
            ),
            "ce_loss": ce_loss,
            "embedding_loss": embedding_loss,
            "layer_loss": layer_loss,
            "logits_loss": logits_loss,
            "attention_loss": attention_loss,
        }

    return info_dict


def batch_forward(
    model, batch, scaler, info_dict, ref_model=None, dpo_loss=None, finetune=False
):
    device = int(os.environ.get("LOCAL_RANK", 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    if info_dict["train_engine"] == "torch_ddp":
        autocast = torch.cuda.amp.autocast(enabled=scaler is not None)
    else:
        autocast = torch.cuda.amp.autocast(
            enabled=True, dtype=dtype, cache_enabled=False
        )

    with autocast:
        info_dict["loss_dict"] = {}
        model_outputs = model(batch, device)
        info_dict["loss_dict"]["loss"] = (
            model_outputs["loss"]
            if model_outputs["loss"] is not None
            else torch.tensor(0.0, device=device)
        )
        info_dict["loss_dict"]["acc"] = (
            model_outputs["acc"]
            if "acc" in model_outputs
            else torch.tensor(0.0, device=device)
        )
        if ref_model is not None and dpo_loss is not None:
            chosen_logps = info_dict["loss_dict"]["chosen_logps"]
            rejected_logps = info_dict["loss_dict"]["rejected_logps"]
            sft_loss = info_dict["loss_dict"]["loss"]
            with torch.no_grad():
                ref_loss_dict = ref_model(batch, device)
            reference_chosen_logps = ref_loss_dict["chosen_logps"]
            reference_rejected_logps = ref_loss_dict["rejected_logps"]
            preference_loss, chosen_reward, reject_reward = dpo_loss(
                chosen_logps,
                rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            dpo_acc = (chosen_reward > reject_reward).float().mean()
            info_dict["loss_dict"]["loss"] = preference_loss + sft_loss
            info_dict["loss_dict"]["sft_loss"] = sft_loss
            info_dict["loss_dict"]["dpo_loss"] = preference_loss
            info_dict["loss_dict"]["dpo_acc"] = dpo_acc
            info_dict["loss_dict"]["chosen_reward"] = chosen_reward.mean()
            info_dict["loss_dict"]["reject_reward"] = reject_reward.mean()

        if finetune:
            # For finetune, we only return the loss_dict
            pass
    return info_dict


def batch_backward(model, scaler, info_dict):
    if info_dict["train_engine"] == "deepspeed":
        scaled_loss = model.backward(info_dict["loss_dict"]["loss"])
    else:
        scaled_loss = info_dict["loss_dict"]["loss"] / info_dict["accum_grad"]
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    info_dict["loss_dict"]["loss"] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):

    grad_norm = 0.0
    if info_dict["train_engine"] == "deepspeed":
        info_dict["is_gradient_accumulation_boundary"] = (
            model.is_gradient_accumulation_boundary()
        )
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (info_dict["batch_idx"] + 1) % info_dict["accum_grad"] == 0:
        # Use mixed precision training
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), info_dict["grad_clip"])
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                logging.warning(
                    "get infinite grad_norm, check your code/data if it appears frequently"
                )
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), info_dict["grad_clip"])
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logging.warning(
                    "get infinite grad_norm, check your code/data if it appears frequently"
                )
        optimizer.zero_grad()
        scheduler.step()
    info_dict["lr"] = optimizer.param_groups[0]["lr"]
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get("epoch", 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict["loss_dict"]
    rank = int(os.environ.get("RANK", 0))

    # only rank 0 write to tensorboard to avoid multi-process write
    if writer is not None:
        if (
            info_dict["train_engine"] == "deepspeed"
            and info_dict["is_gradient_accumulation_boundary"] is True
        ) or (
            info_dict["train_engine"] == "torch_ddp"
            and (info_dict["batch_idx"] + 1) % info_dict["accum_grad"] == 0
        ):
            for k in ["epoch", "lr", "grad_norm"]:
                writer.add_scalar("{}/{}".format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar("{}/{}".format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict["batch_idx"] + 1) % info_dict["log_interval"] == 0:
        log_str = "{} Batch {}/{} ".format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += "{} {:.6f} ".format(name, value)
        if tag == "TRAIN":
            log_str += "lr {:.8f} grad_norm {:.6f}".format(
                info_dict["lr"], info_dict["grad_norm"]
            )
        log_str += " rank {}".format(rank)
        logging.debug(log_str)


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict["lr"]
    rank = int(os.environ.get("RANK", 0))
    logging.info(
        "Epoch {} Step {} CV info lr {} {} rank {}".format(
            epoch,
            step + 1,
            lr,
            rank,
            " ".join(["{} {}".format(k, v) for k, v in loss_dict.items()]),
        )
    )

    if writer is not None:
        for k in ["epoch", "lr"]:
            writer.add_scalar("{}/{}".format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar("{}/{}".format(tag, k), v, step + 1)
