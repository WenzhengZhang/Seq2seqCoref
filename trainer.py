import time
from tqdm.auto import tqdm
import torch.distributed as dist
from transformers.trainer_utils import HPSearchBackend, speed_metrics, \
    TrainOutput
from pathlib import Path
import sys
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_callback import TrainerState
from transformers.trainer import TRAINER_STATE_NAME, OptimizerNames

from transformers.utils import is_apex_available
from transformers.integrations import hp_params
from transformers import Seq2SeqTrainer
import numpy as np
import os
import json
import re
from packaging import version
import torch.nn as nn
from collections import defaultdict
from metrics import CorefAllMetrics
from typing import Dict, Union, Any, Optional, Tuple, List
import torch
import shutil
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import math
from transformers.pytorch_utils import is_torch_less_than_1_11
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput, has_length, \
    denumpify_detensorize, ShardedDDPOption
from data import get_document_predicts, parse_int_output_tokens, \
    parse_short_target_tokens, parse_nonint_output_tokens
from constants import SPECIAL_IDS, MARK_SPECIAL_IDS, NON_INT_SPECIAL_IDS
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import find_batch_size, nested_concat, \
    nested_numpify, IterableDatasetShard, nested_truncate, get_parameter_names
from transformers.modeling_utils import PreTrainedModel, unwrap_model, \
    load_sharded_checkpoint

from transformers.utils import logging, is_torch_tpu_available, \
    is_sagemaker_mp_enabled, is_safetensors_available, SAFE_WEIGHTS_NAME, \
    WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from transformers.integrations import is_fairscale_available
from transformers.dependency_versions_check import dep_version_check

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
if is_fairscale_available():
    dep_version_check("fairscale")
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse(
        "1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    import safetensors.torch
if is_apex_available():
    from apex import amp

from transformers import LogitsProcessorList
from logits_processor import ShortSeqProcessor, IntProcessor, NonIntProcessor
from transformers.trainer_seq2seq import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class CorefTrainer(Seq2SeqTrainer):

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime,
                                                      output_dir=output_dir)
        if self.args.val_after_train and self.args.eval_delay < \
                self.state.global_step:
            for checkpoint in checkpoints_sorted[:-1]:
                states_dir = [str(x) for x in Path(
                    checkpoint).glob(f'global_step*') if os.path.isdir(x)]
                for state_dir in states_dir:
                    logger.info(f"Deleting optimizer states of saved "
                                f"checkpoint {checkpoint}")
                    if os.path.exists(state_dir) and os.path.isdir(
                            state_dir):
                        shutil.rmtree(state_dir)
        else:
            if len(checkpoints_sorted) <= self.args.save_total_limit:
                return

            # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
            # we don't do to allow resuming.
            save_total_limit = self.args.save_total_limit
            if (
                    self.state.best_model_checkpoint is not None
                    and self.args.save_total_limit == 1
                    and checkpoints_sorted[
                -1] != self.state.best_model_checkpoint
            ):
                save_total_limit = 2

            number_of_checkpoints_to_delete = max(0, len(
                checkpoints_sorted) - save_total_limit)
            checkpoints_to_be_deleted = checkpoints_sorted[
                                        :number_of_checkpoints_to_delete]
            for checkpoint in checkpoints_to_be_deleted:
                logger.info(
                    f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                shutil.rmtree(checkpoint)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel) and not hasattr(
                self.model, 'save_pretrained'):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict,
                    # safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                # if self.args.save_safetensors:
                #     safetensors.torch.save_file(state_dict,
                #                                 os.path.join(output_dir,
                #                                              SAFE_WEIGHTS_NAME))
                # else:
                torch.save(state_dict, os.path.join(output_dir,
                                                    WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict,
                # safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None,
            trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps,
                resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state,
                                                            self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader,
                                            "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    if self.args.joint_train:
                        train_dataloader.dataset.set_samples(epoch)
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)
        if args.manual_empty_cache:
            torch.cuda.empty_cache()
        for epoch in range(epochs_trained, num_train_epochs):
            if self.args.joint_train:
                train_dataloader.dataset.set_samples(epoch)
            if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(
                    train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [
                    args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args,
                                                                self.state,
                                                                self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            if args.manual_empty_cache:
                torch.cuda.empty_cache()
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if args.manual_empty_cache:
                    torch.cuda.empty_cache()
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args,
                                                                       self.state,
                                                                       self.control)
                # if args.manual_empty_cache:
                #     torch.cuda.empty_cache()
                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (
                        torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                            1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    if args.manual_empty_cache:
                        torch.cuda.empty_cache()
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients,
                                              scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(
                                    self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    if args.manual_empty_cache:
                        torch.cuda.empty_cache()
                    self.control = self.callback_handler.on_step_end(args,
                                                                     self.state,
                                                                     self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch,
                                                  ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args,
                                                                        self.state,
                                                                        self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state,
                                                              self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch,
                                          ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time,
                                num_samples=num_train_samples,
                                num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False,
                                                      output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and \
                self.args.save_total_limit == 1 and self.is_world_process_zero():
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state,
                                                          self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def my_compute_metrics(self,
                           doc_labels: Dict[str, List[List]],
                           predicts: Any,
                           samples: List,
                           split: str,
                           id_to_name: Dict = None
                           ) -> Dict:
        if self.args.joint_train:
            data_names = self.args.joint_data_names.split(',')
            joint_threds = [
                int(t) for t in self.args.joint_min_num_mentions.split(',')]
            name_to_threds = {n: t for n, t in zip(data_names, joint_threds)}
        documents_to_chunk_data = defaultdict(list)
        documents_to_chunk_gold = defaultdict(list)
        predictions = {}
        golds = {}
        assert len(samples) == len(predicts)
        out_sents = []
        last_doc_id = re.sub(r'_\d+$', '', samples[0]['doc_key'])
        for sample, predict in zip(samples, predicts):
            doc_key = sample['doc_key']
            doc_id = re.sub(r'_\d+$', '', doc_key)
            # require convert to ids first
            input_ids = sample['sentence']
            subtoken_map = sample['subtoken_map']
            offset = sample['offset']
            # remove bos
            predict_ids = predict[1:].tolist()
            gold_data = sample['seg_clusters']
            if self.args.joint_train:
                thred = name_to_threds[id_to_name[doc_id]]
            else:
                thred = self.args.min_num_mentions
            if self.args.seq2seq_type == "short_seq":
                special_ids = MARK_SPECIAL_IDS if self.args.mark_sentence \
                    else SPECIAL_IDS
                pred_data, aligned_input_ids, aligned_pred_ids = \
                    parse_short_target_tokens(input_ids, predict_ids,
                                              special_ids, subtoken_map,
                                              self.tokenizer,
                                              self.args.align_mode,
                                              thred,
                                              self.args.mark_sentence
                                              )
                pred_tokens = self.tokenizer.convert_ids_to_tokens(
                    predict_ids)
                out_predict = {
                    'doc_key': doc_key,
                    'pred_tokens': pred_tokens,
                    'pred_text': self.tokenizer.convert_tokens_to_string(
                        pred_tokens),
                    'pred_aligned_text': self.tokenizer.convert_ids_to_tokens(
                        aligned_pred_ids
                    ),
                    'input_aligned_text': self.tokenizer.convert_ids_to_tokens(
                        aligned_input_ids
                    )
                }
            else:
                is_tagging = (self.args.seq2seq_type == 'tagging')
                if self.args.action_type == 'integer':
                    pred_data, pred_token_mentions, predict_ids = \
                        parse_int_output_tokens(
                            input_ids,
                            predict_ids,
                            SPECIAL_IDS,
                            subtoken_map,
                            self.tokenizer,
                            thred, is_tagging)
                else:
                    pred_data, pred_token_mentions, predict_ids = \
                        parse_nonint_output_tokens(
                            input_ids,
                            predict_ids,
                            NON_INT_SPECIAL_IDS,
                            subtoken_map,
                            self.tokenizer, self.args.add_mention_end,
                            thred)
                pred_token_mentions = [(m[0] + offset, m[1] + offset) for m in
                                       pred_token_mentions]
                pred_tokens = self.tokenizer.convert_ids_to_tokens(
                    predict_ids)
                out_predict = {'doc_key': doc_key,
                               'pred_tokens': pred_tokens,
                               'pred_text':
                                   self.tokenizer.convert_tokens_to_string(
                                       pred_tokens),
                               'predict_clusters': pred_data,
                               'gold_clusters': gold_data,
                               'predict_token_mentions': pred_token_mentions
                               }
            # list of (m1,m2)

            documents_to_chunk_data[doc_id].extend(pred_data)
            documents_to_chunk_gold[doc_id].extend(gold_data)

            out_sents.append(out_predict)
            if doc_id != last_doc_id:
                predictions[last_doc_id] = get_document_predicts(
                    documents_to_chunk_data[
                        last_doc_id])
                golds[last_doc_id] = get_document_predicts(
                    documents_to_chunk_gold[
                        last_doc_id])
            last_doc_id = doc_id
        # final one
        predictions[last_doc_id] = get_document_predicts(
            documents_to_chunk_data[last_doc_id]
        )
        golds[last_doc_id] = get_document_predicts(
            documents_to_chunk_gold[last_doc_id]
        )
        # print(predictions)
        if self.args.joint_train:
            predictions_list = defaultdict(list)
            labels_list = defaultdict(list)
            golds_list = defaultdict(list)
        else:
            predictions_list = []
            labels_list = []
            golds_list = []
        for document_id, doc_label in doc_labels.items():
            if self.args.joint_train:
                predictions_list[id_to_name[document_id]].append(
                    predictions[document_id])
                labels_list[id_to_name[document_id]].append(doc_label)
                golds_list[id_to_name[document_id]].append(golds[document_id])
            else:
                predictions_list.append(predictions[document_id])
                labels_list.append(doc_label)
                golds_list.append(golds[document_id])
        if self.args.joint_train:
            label_results = {}
            gold_results = {}
            for dn in predictions_list.keys():
                metrics = CorefAllMetrics().get_all_metrics(
                    labels_list[dn],
                    predictions_list[dn])
                metrics_golds = CorefAllMetrics().get_all_metrics(
                    golds_list[dn],
                    predictions_list[dn])
                single_label_results = {
                    f'{dn}_{metric_name}_{x}': v
                    for metric_name, metric_values in metrics['micro'].items()
                    for x, v in metric_values.items()
                }
                single_gold_results = {
                    f'{dn}_gold_{metric_name}_{x}': v
                    for metric_name, metric_values in
                    metrics_golds['micro'].items()
                    for x, v in metric_values.items()
                }
                label_results.update(single_label_results)
                gold_results.update(single_gold_results)

        else:
            metrics = CorefAllMetrics().get_all_metrics(labels_list,
                                                        predictions_list)
            metrics_golds = CorefAllMetrics().get_all_metrics(golds_list,
                                                              predictions_list)
            label_results = {
                f'{metric_name}_{x}': v
                for metric_name, metric_values in metrics['micro'].items()
                for x, v in metric_values.items()
            }
            gold_results = {
                f'gold_{metric_name}_{x}': v
                for metric_name, metric_values in metrics_golds['micro'].items()
                for x, v in metric_values.items()
            }
        results = {**label_results, **gold_results}
        if self.args.joint_train:
            avg_f1s = [results[f"{dname}_average_f1"] for dname in
                       data_names]
            results["average_f1"] = sum(avg_f1s) / len(avg_f1s)
        if self.is_world_process_zero() and self.args.save_predicts:
            os.makedirs(self.args.save_dir, exist_ok=True)
            save_path = os.path.join(self.args.save_dir,
                                     f'{split}-predicts.txt')
            results_path = os.path.join(self.args.save_dir,
                                        f'{split}-results.json')
            with open(save_path, 'w') as f:
                for p in out_sents:
                    f.write('%s\n' % json.dumps(p))
            with open(results_path, 'w') as f:
                json.dump(results, f)

        return results

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = False,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
                Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
                Works both with or without labels.
                """
        args = self.args

        prediction_loss_only = False

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None,
                inference=is_deepspeed_zero3_enabled()
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
        if self.args.gradient_checkpointing:
            self.model.config.use_cache = True
        model = self._wrap_model(self.model, training=False,
                                 dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader,
                                           [args.device]).per_device_loader(
                args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs,
                                                        prediction_loss_only,
                                                        ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs[
                                                    "input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode,
                                       padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args,
                                                                    self.state,
                                                                    self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (
                    step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode,
                                           padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(
                    all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(
                eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        doc_labels = eval_dataset.doc_labels
        eval_samples = eval_dataset.samples
        split = eval_dataset.split
        if self.args.joint_train:
            doc_id_to_name = eval_dataset.id_to_name
        else:
            doc_id_to_name = None
        # allow_singletons = eval_dataset.data_args.allow_singletons
        assert all_preds is not None
        metrics = self.my_compute_metrics(doc_labels, all_preds,
                                          eval_samples, split,
                                          doc_id_to_name)
        # if all_preds is not None and doc_labels is not None:
        #     metrics = self.get_eval_metrics(doc_labels, all_preds,
        #                                     eval_samples, split)
        # else:
        #     metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        if self.args.gradient_checkpointing:
            self.model.config.use_cache = False
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels,
                              metrics=metrics, num_samples=num_samples)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys:
                list of ignore keys

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get(
                "max_length") is not None else self.model.config.max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get(
                "num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get(
                "synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model,
                   "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        #  add our logits_processor here
        if self.args.seq2seq_type != 'short_seq':
            if self.args.action_type == 'non_integer':
                gen_kwargs['logits_processor'] = LogitsProcessorList(
                    [NonIntProcessor(generation_inputs, NON_INT_SPECIAL_IDS,
                                     self.args.add_mention_end)])
            else:
                gen_kwargs['logits_processor'] = LogitsProcessorList(
                    [IntProcessor(generation_inputs, SPECIAL_IDS,
                                  self.args.seq2seq_type)])
        elif self.args.mark_sentence:
            gen_kwargs['logits_processor'] = LogitsProcessorList(
                [ShortSeqProcessor(generation_inputs, MARK_SPECIAL_IDS)])
        if self.args.use_peft:
            gen_kwargs["input_ids"] = generation_inputs
            gen_kwargs["use_cache"] = True
            generated_tokens = self.model.generate(
                **gen_kwargs,
            )
        else:
            generated_tokens = self.model.generate(
                generation_inputs,
                **gen_kwargs,
            )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens,
                                                            gen_kwargs[
                                                                "max_length"])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs,
                                               inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else
                            outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels,
                                                      gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
