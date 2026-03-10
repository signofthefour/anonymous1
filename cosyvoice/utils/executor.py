None  # Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import logging
from contextlib import nullcontext
import os
import math

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import (
    update_parameter_and_lr,
    log_per_step,
    log_per_save,
    batch_forward,
    batch_backward,
    save_model,
    cosyvoice_join,
    batch_forward_with_distill,
)


class Executor:

    def __init__(
        self,
        gan: bool = False,
        ref_model: torch.nn.Module = None,
        dpo_loss: torch.nn.Module = None,
    ):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.device = torch.device("cuda:{}".format(self.rank))

    def train_one_epoc(
        self,
        model,
        optimizer,
        scheduler,
        train_data_loader,
        cv_data_loader,
        writer,
        info_dict,
        scaler,
        group_join,
        ref_model=None,
    ):
        """Train one epoch"""

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch {} TRAIN info lr {} rank {}".format(self.epoch, lr, self.rank)
        )
        logging.info(
            "using accumulate grad, new batch size is {} times"
            " larger than before".format(info_dict["accum_grad"])
        )
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        model_context = (
            model.join if info_dict["train_engine"] == "torch_ddp" else nullcontext
        )
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if (
                    info_dict["train_engine"] == "torch_ddp"
                    and (batch_idx + 1) % info_dict["accum_grad"] != 0
                ):
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(
                        model,
                        batch_dict,
                        scaler,
                        info_dict,
                        ref_model=self.ref_model,
                        dpo_loss=self.dpo_loss,
                    )
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(
                    model, optimizer, scheduler, scaler, info_dict
                )
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if (
                    info_dict["save_per_step"] > 0
                    and (self.step + 1) % info_dict["save_per_step"] == 0
                    and (batch_idx + 1) % info_dict["accum_grad"] == 0
                ):
                    dist.barrier()
                    self.cv(
                        model, cv_data_loader, writer, info_dict, on_batch_end=False
                    )
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def heal_one_epoch(
        self,
        model,
        optimizer,
        scheduler,
        train_data_loader,
        cv_data_loader,
        writer,
        info_dict,
        scaler,
        group_join,
        ref_model=None,
        teacher_model=None,
        distill_teacher_layers=None,
        distill_attention=True,
    ):
        """Train one epoch, some as above, but for heal model with more loss functions"""

        # sanity check the validation
        # print("Sanity check the validation before healing epoch {}".format(self.epoch))
        # self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch {} TRAIN info lr {} rank {}".format(self.epoch, lr, self.rank)
        )
        logging.info(
            "using accumulate grad, new batch size is {} times"
            " larger than before".format(info_dict["accum_grad"])
        )
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        model_context = (
            model.join if info_dict["train_engine"] == "torch_ddp" else nullcontext
        )
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if (
                    info_dict["train_engine"] == "torch_ddp"
                    and (batch_idx + 1) % info_dict["accum_grad"] != 0
                ):
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    # info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                    info_dict = batch_forward_with_distill(
                        student_model=model,
                        teacher_model=teacher_model,
                        batch=batch_dict,
                        scaler=scaler,
                        info_dict=info_dict,
                        temperature=info_dict.get("temperature", 2.0),
                        alpha=info_dict.get("alpha", 0.5),
                        distill_teacher_layers=distill_teacher_layers,
                        distill_attention=distill_attention,
                    )
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(
                    model, optimizer, scheduler, scaler, info_dict
                )
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if (
                    info_dict["save_per_step"] > 0
                    and (self.step + 1) % info_dict["save_per_step"] == 0
                    and (batch_idx + 1) % info_dict["accum_grad"] == 0
                ):
                    dist.barrier()
                    self.cv(
                        model, cv_data_loader, writer, info_dict, on_batch_end=False
                    )
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoch_gan(
        self,
        model,
        optimizer,
        scheduler,
        optimizer_d,
        scheduler_d,
        train_data_loader,
        cv_data_loader,
        writer,
        info_dict,
        scaler,
        group_join,
    ):
        """Train one epoch"""

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch {} TRAIN info lr {} rank {}".format(self.epoch, lr, self.rank)
        )
        logging.info(
            "using accumulate grad, new batch size is {} times"
            " larger than before".format(info_dict["accum_grad"])
        )
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = (
            model.join if info_dict["train_engine"] == "torch_ddp" else nullcontext
        )
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if (
                    info_dict["train_engine"] == "torch_ddp"
                    and (batch_idx + 1) % info_dict["accum_grad"] != 0
                ):
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict["turn"] = "discriminator"
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(
                    model, optimizer_d, scheduler_d, scaler, info_dict
                )
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict["turn"] = "generator"
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(
                    model, optimizer, scheduler, scaler, info_dict
                )
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if (
                    info_dict["save_per_step"] > 0
                    and (self.step + 1) % info_dict["save_per_step"] == 0
                    and (batch_idx + 1) % info_dict["accum_grad"] == 0
                ):
                    dist.barrier()
                    self.cv(
                        model, cv_data_loader, writer, info_dict, on_batch_end=False
                    )
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(
        self, model, cv_data_loader, writer, info_dict, on_batch_end=True, heal=False
    ):
        """Cross validation on"""
        logging.info(
            "Epoch {} Step {} on_batch_end {} CV rank {}".format(
                self.epoch, self.step + 1, on_batch_end, self.rank
            )
        )
        model.eval()
        total_num_utts = 0
        # Use per-key sums and per-key utterance counters so we can ignore
        # non-finite (NaN/Inf) loss values when computing averages.
        total_loss_sums = {}
        total_loss_utts = {}
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict["turn"] = "generator"
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict["loss_dict"].items():
                # get numeric value (supports torch scalar tensors and floats)
                try:
                    val = float(v.item()) if hasattr(v, "item") else float(v)
                except Exception:
                    # fallback: skip non-convertible entries
                    logging.warning(
                        "CV: could not convert loss value for key %s, skipping.", k
                    )
                    continue
                if not math.isfinite(val):
                    logging.warning(
                        "CV: non-finite loss for key %s (value=%s) at batch %d - skipping",
                        k,
                        val,
                        batch_idx,
                    )
                    continue
                if k not in total_loss_sums:
                    total_loss_sums[k] = 0.0
                    total_loss_utts[k] = 0
                total_loss_sums[k] += val * num_utts
                total_loss_utts[k] += num_utts
            log_per_step(None, info_dict)

        # Compute per-key averages using only finite entries. If a key had no
        # finite values, set its average to 0.0 and log a warning.
        total_loss_dict = {}
        for k, s in total_loss_sums.items():
            ut = total_loss_utts.get(k, 0)
            if ut > 0:
                total_loss_dict[k] = s / ut
            else:
                logging.warning(
                    "CV: all values for loss key %s were non-finite; setting to 0.0", k
                )
                total_loss_dict[k] = 0.0
        info_dict["loss_dict"] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = (
            "epoch_{}_whole".format(self.epoch)
            if on_batch_end
            else "epoch_{}_step_{}".format(self.epoch, self.step + 1)
        )
        save_model(model, model_name, info_dict)
