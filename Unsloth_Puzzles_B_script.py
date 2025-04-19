# Helpful functions used through the entire notebook
import torch
import torch.nn as nn
from transformers import set_seed
import time
import inspect
import os

major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = major_version >= 8
from inspect import currentframe as _C, getframeinfo

_F = lambda c: getframeinfo(c).lineno  # Gets line number
WARN = lambda x: print(f"\033[31m{x}\033[0m")  # Red colored warnings


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# %%
# HELPFUL functions to undo Unsloth patches:
import sys


def remove_patched_module(package_name):
    modules_to_delete = [
        name
        for name in sys.modules
        if name == package_name or name.startswith(package_name + ".")
    ]
    for name in modules_to_delete:
        del sys.modules[name]


remove_patched_module("trl")
remove_patched_module("transformers")
remove_patched_module("peft")
remove_patched_module("bitsandbytes")

# %%
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True," "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

max_seq_length = 2048
torch.set_default_dtype(torch.float16)
model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Get dataset
from datasets import load_dataset

url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")

# %%
torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": True,
    "triton.cudagraphs": False,
}

import os

os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import logging

torch._inductor.config.debug = True
torch._logging.set_logs(
    dynamo=logging.WARN,
    inductor=logging.WARN,
    graph_breaks=True,
    # recompiles=True,
    # recompiles_verbose=True,
    # compiled_autograd_verbose=True,
)
torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = False


# %% [markdown]
# ## Patches

print("Patching modules")

import contextlib
import functools
import shutil
import torch
import torch.distributed as dist
from packaging import version

from transformers.trainer_utils import (
    TrainOutput,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import (
    WEIGHTS_NAME,
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logging,
)
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_model_param_count,
)
from torch.utils.data import (
    RandomSampler,
)


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        DistributedType,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]


if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


def patched_inner_training_loop(
    self,
    batch_size=None,
    args=None,
    resume_from_checkpoint=None,
    trial=None,
    ignore_keys_for_eval=None,
):
    self.accelerator.free_memory()
    self._train_batch_size = batch_size
    if self.args.auto_find_batch_size:
        if self.state.train_batch_size != self._train_batch_size:
            from accelerate.utils import release_memory

            (self.model_wrapped,) = release_memory(self.model_wrapped)
            self.model_wrapped = self.model

            # Check for DeepSpeed *after* the initial pass and modify the config
            if self.is_deepspeed_enabled:
                # Temporarily unset `self.args.train_batch_size`
                original_bs = self.args.per_device_train_batch_size
                self.args.per_device_train_batch_size = self._train_batch_size // max(
                    1, self.args.n_gpu
                )
                self.propagate_args_to_deepspeed(True)
                self.args.per_device_train_batch_size = original_bs
        self.state.train_batch_size = self._train_batch_size
    logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    # Data loader and number of training steps
    train_dataloader = self.get_train_dataloader()
    if self.is_fsdp_xla_v2_enabled:
        train_dataloader = tpu_spmd_dataloader(train_dataloader)

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = (
        self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    )
    logger.info("total_train_batch_size", total_train_batch_size)
    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    num_train_tokens = None
    if self.args.include_tokens_per_second:
        num_train_tokens = self.num_tokens(
            train_dataloader, None if epoch_based else max_steps
        )
        # If going by epochs, multiply tokens linearly
        if len_dataloader is not None and epoch_based:
            num_train_tokens *= args.num_train_epochs
        # Otherwise since its steps, we just multiply by grad accum
        else:
            num_train_tokens *= args.gradient_accumulation_steps

    # if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #     if self.args.n_gpu > 1:
    #         # nn.DataParallel(model) replicates the model, creating new variables and module
    #         # references registered here no longer work on other gpus, breaking the module
    #         raise ValueError(
    #             "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #             " (torchrun or torch.distributed.launch (deprecated))."
    #         )
    #     else:
    #         debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    delay_optimizer_creation = (
        is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
    )

    # We need to reset the scheduler, as its parameters may be different on subsequent calls
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False

    if self.is_deepspeed_enabled:
        self.optimizer, self.lr_scheduler = deepspeed_init(
            self, num_training_steps=max_steps
        )

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState(
        stateful_callbacks=[
            cb
            for cb in self.callback_handler.callbacks + [self.control]
            if isinstance(cb, ExportableState)
        ]
    )
    self.state.is_hyper_param_search = trial is not None
    self.state.train_batch_size = self._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    self.state.compute_steps(args, max_steps)

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
        )

    model = self._wrap_model(self.model_wrapped)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = True if model is self.model else False

    if use_accelerator_prepare and self.is_fsdp_enabled:
        # In case of auto_find_batch_size=True
        # Remove FSDP wrapping from sub-models.
        self.model = unwrap_model(self.model, recursive=True)

    if delay_optimizer_creation:
        if use_accelerator_prepare:
            # configure fsdp plugin for qlora if any
            self._fsdp_qlora_plugin_updates()
            # if self.accelerator.mixed_precision != "fp8":
            # self.model = self.accelerator.prepare(self.model)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            # if self.use_apex:
            #     model = self.accelerator.prepare(self.model)
            # else:
            #     model, self.optimizer = self.accelerator.prepare(
            #         self.model, self.optimizer
            #     )
            self.optimizer = self.accelerator.prepare(self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.optimizer, self.lr_scheduler
            )

    elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        # In this case we are in DDP + LOMO, which should be supported
        self.optimizer = self.accelerator.prepare(self.optimizer)

    if self.is_fsdp_enabled:
        self.model = self.model_wrapped = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
        self.model_wrapped = model

    # backward compatibility
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped

    # ckpt loading
    if resume_from_checkpoint is not None:
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped,
                resume_from_checkpoint,
                load_module_strict=not _is_peft_model(self.model),
            )
        elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    # Check if saved optimizer or scheduler states exist
    self._load_optimizer_and_scheduler(resume_from_checkpoint)
    self._load_scaler(resume_from_checkpoint)

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(
        f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
    )
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(
            f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
        )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(
        f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
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
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        )
        self.compare_trainer_and_checkpoint_args(self.args, self.state)
        self._load_callback_state()
        epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = self.state.global_step % (
                num_update_steps_per_epoch
            )
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {self.state.global_step}")
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )

    # Update the references
    for attr in ("model", "optimizer", "lr_scheduler"):
        setattr(self.callback_handler, attr, getattr(self, attr))
    self.callback_handler.train_dataloader = train_dataloader

    self.state.init_training_references(self, max_steps, num_train_epochs, trial)

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0).to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()
    grad_norm: Optional[float] = None
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    if args.eval_on_start:
        self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    for epoch in range(epochs_trained, num_train_epochs):
        epoch_dataloader = train_dataloader
        if hasattr(epoch_dataloader, "set_epoch"):
            epoch_dataloader.set_epoch(epoch)

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None

        steps_in_epoch = (
            len(epoch_dataloader)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        self.control = self.callback_handler.on_epoch_begin(
            args, self.state, self.control
        )

        if (
            epoch == epochs_trained
            and resume_from_checkpoint is not None
            and steps_trained_in_current_epoch == 0
        ):
            self._load_rng_state(resume_from_checkpoint)

        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_dataloader = skip_first_batches(
                epoch_dataloader, steps_trained_in_current_epoch
            )
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        step = -1
        epoch_iterator = iter(epoch_dataloader)
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = num_examples % args.gradient_accumulation_steps
        if remainder == 0:
            remainder = args.gradient_accumulation_steps
        update_step = -1
        total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
        if args.gradient_accumulation_steps == 1:
            total_updates -= 1
        for _ in range(total_updates):
            update_step += 1
            num_batches = (
                args.gradient_accumulation_steps
                if update_step != (total_updates - 1)
                else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(
                epoch_iterator, num_batches, args.device
            )
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(
                        self.model, "main_input_name", "input_ids"
                    )
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_tokens = inputs[main_input_name].numel()
                        input_tokens = torch.tensor(
                            input_tokens, device=self.args.device, dtype=torch.int64
                        )
                        self.state.num_input_tokens_seen += (
                            self.accelerator.gather(input_tokens).sum().cpu().item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
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
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i != len(batch_samples) - 1
                    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    else contextlib.nullcontext
                )
                with context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss = tr_loss + tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss = tr_loss + tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if do_sync_step:
                    # Since we perform prefetching, we need to manually set sync_gradients to True
                    self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(
                                args.max_grad_norm
                            )
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type
                            == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.control = self.callback_handler.on_pre_optimizer_step(
                        args, self.state, self.control
                    )

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(
                        args, self.state, self.control
                    )

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )
                    self._maybe_log_save_evaluate(
                        tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                # PyTorch/XLA relies on the data loader to insert the mark_step for
                # each step. Since we are breaking the loop early, we need to manually
                # insert the mark_step here.
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_xla_available():
                    xm.mark_step()
                break
        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(
            args, self.state, self.control
        )
        self._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
        )

        if self.control.should_training_stop:
            break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    logger.info(
        "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
    )
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        # Wait for everyone to get here so we are sure the model has been saved by process 0.
        if is_torch_xla_available():
            xm.rendezvous("load_best_model_at_end")
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()

        self._load_best_model()

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    effective_global_step = max(
        self.state.global_step, 0.001
    )  # Avoid ZeroDivisionError
    train_loss = self._total_loss_scalar / effective_global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_train_samples,
        num_steps=self.state.max_steps,
        num_tokens=num_train_tokens,
    )
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    run_dir = self._get_output_dir(trial)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if (
        self.args.should_save
        and self.state.best_model_checkpoint is not None
        and self.args.save_total_limit == 1
    ):
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(
                    f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                )
                shutil.rmtree(checkpoint, ignore_errors=True)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    # Wait for the checkpoint to be uploaded.
    self._finish_current_push()

    # After training we make sure to retrieve back the original forward pass method
    # for the embedding layer by removing the forward post hook.
    if self.neftune_noise_alpha is not None:
        self._deactivate_neftune(self.model)

    return TrainOutput(self.state.global_step, train_loss, metrics)


import transformers.trainer as trans_trainer

trans_trainer.Trainer._inner_training_loop = patched_inner_training_loop

print("patched trainer")

# %%
from typing import Optional, Tuple
from transformers.integrations.sdpa_attention import repeat_kv


def patched_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:

    # stole sdpa implementation from source
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # we can just set is_casual to true since we're training
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_output, attn_weights = patched_sdpa_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


import transformers.models.llama.modeling_llama

transformers.models.llama.modeling_llama.LlamaAttention.forward = (
    patched_llama_attention_forward
)

print("Patched Compilable Attention")

# %% [markdown]
# torch compile will atempt to bin Linear4bit as a weight-only quantized (WOQ) linear operation, make a check and fail to match.
# However this check has expectations on the params of `Linear4bitCompilable`, which will fail. We can choose to modify our implementation or the check. In this case, we change the check (which should honestly have been more flexible)


# %%
def patch_is_valid_woq_optimization_pattern():
    def fn(match):
        assert all(k in match.kwargs for k in ("x", "weight", "scales"))
        try:
            x = match.kwargs["x"].meta["val"]
            weight = match.kwargs["weight"].meta["val"]
            scales = match.kwargs["scales"].meta["val"]
            return (
                # For now, we only support woq mm kernels
                # with x.type=bfloat16 and w.type=int8
                x.dtype == torch.float16
                and weight.dtype == torch.int8
                and scales.dtype == torch.float16
                # _weight_int8pack_mm kernel only supports cpu now
                # TODO: add cuda kernel support instead of calling mul+sum
                and x.device.type == "cpu"
                and x.device == weight.device
                and x.device == scales.device
            )
        except AttributeError:
            # match does not conform to expected param structure
            return False

    return fn


import torch._inductor.fx_passes.quantization as inductor_quantization

inductor_quantization._is_valid_woq_optimization_pattern = (
    patch_is_valid_woq_optimization_pattern
)

print("Patched Torch Compile WOQ Pattern")

from torch.distributed.tensor import DTensor
import collections
import copy
import os
import warnings
from typing import Any, Optional, Union

from safetensors.torch import save_file as safe_save_file
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    get_peft_model_state_dict,
    id_tensor_storage,
)


def patched_save_pretrained(
    self,
    save_directory: str,
    safe_serialization: bool = True,
    selected_adapters: Optional[list[str]] = None,
    save_embedding_layers: Union[str, bool] = "auto",
    is_main_process: bool = True,
    path_initial_model_for_weight_conversion: Optional[str] = None,
    **kwargs: Any,
) -> None:

    if os.path.isfile(save_directory):
        raise ValueError(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )

    if selected_adapters is None:
        selected_adapters = list(self.peft_config.keys())
    else:
        if any(
            selected_adapter_name not in list(self.peft_config.keys())
            for selected_adapter_name in selected_adapters
        ):
            raise ValueError(
                f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                f" {list(self.peft_config.keys())} - got {selected_adapters}."
            )

    def save_mutated_as_lora(
        peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
    ):
        if peft_config.use_rslora and (
            peft_config.rank_pattern or peft_config.alpha_pattern
        ):
            msg = (
                "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not supported when "
                "using `rank_pattern` or `alpha_pattern` at the same time as `use_rslora=True`."
            )
            raise ValueError(msg)

        if not any(
            str(peft_config.init_lora_weights).lower().startswith(prefix)
            for prefix in ["pissa", "corda", "olora", "true"]
        ):
            warnings.warn(
                "`path_initial_model_for_weight_conversion` only works for converting a PiSSA/CorDA/OLoRA adapter to "
                "a LoRA adapter"
            )
        initial_adapter_name = os.path.basename(
            path_initial_model_for_weight_conversion
        )
        try:
            self.load_adapter(
                os.path.dirname(path_initial_model_for_weight_conversion),
                subfolder=initial_adapter_name,
                adapter_name=initial_adapter_name,
            )
            is_pissa = (
                str(self.peft_config[initial_adapter_name].init_lora_weights)
                .lower()
                .startswith("pissa")
            )
            is_corda = (
                str(self.peft_config[initial_adapter_name].init_lora_weights).lower()
                == "corda"
            )
            is_olora = (
                str(self.peft_config[initial_adapter_name].init_lora_weights).lower()
                == "olora"
            )
            if is_pissa or is_corda or is_olora:
                raise ValueError(
                    "The `init_lora_weights` parameter of the initial adapter should be set to `True`. "
                    "Otherwise, `self.load_adapter` will subtract the decomposed values again based on the "
                    "residual model."
                )
            output_state_dict = self.base_model.subtract_mutated_init(
                output_state_dict, initial_adapter_name, kwargs
            )
        finally:
            self.delete_adapter(initial_adapter_name)
        return output_state_dict

    if is_main_process:
        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)

    for adapter_name in selected_adapters:
        peft_config = self.peft_config[adapter_name]
        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
            adapter_name=adapter_name,
            save_embedding_layers=save_embedding_layers,
        )
        output_dir = (
            os.path.join(save_directory, adapter_name)
            if adapter_name != "default"
            else save_directory
        )
        os.makedirs(output_dir, exist_ok=True)

        if is_main_process and safe_serialization:

            # we need to localize all DTensors before they can be saved
            for name, tensor in output_state_dict.items():
                if isinstance(tensor, DTensor):
                    # If the tensor is a DTensor, we need to get the underlying tensor
                    tensor = tensor.to_local()
                    if not isinstance(tensor, torch.Tensor):
                        tensor.wait()
                    output_state_dict[name] = tensor

            # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in output_state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            for _, names in shared_ptrs.items():
                # Here we just clone the shared tensors to avoid tensor aliasing which is
                # not supported in safetensors.
                for shared_tensor_name in names[1:]:
                    output_state_dict[shared_tensor_name] = output_state_dict[
                        shared_tensor_name
                    ].clone()
            if path_initial_model_for_weight_conversion is not None:
                peft_config = copy.deepcopy(peft_config)
                peft_config.init_lora_weights = True
                peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                output_state_dict = save_mutated_as_lora(
                    peft_config,
                    path_initial_model_for_weight_conversion,
                    output_state_dict,
                    kwargs,
                )
            safe_save_file(
                output_state_dict,
                os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        elif is_main_process:
            if path_initial_model_for_weight_conversion is not None:
                peft_config = copy.deepcopy(peft_config)
                peft_config.init_lora_weights = True
                peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                output_state_dict = save_mutated_as_lora(
                    peft_config,
                    path_initial_model_for_weight_conversion,
                    output_state_dict,
                    kwargs,
                )
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if peft_config.is_prompt_learning
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = self._get_base_model_class(
                is_prompt_tuning=peft_config.is_prompt_learning,
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        if is_main_process:
            if path_initial_model_for_weight_conversion is not None:
                peft_config.init_lora_weights = True
                peft_config.r *= 2
                if not peft_config.use_rslora:
                    peft_config.lora_alpha *= 2
                else:
                    # with rslora, we have scaling = alpha / sqrt(r), we thus adjust alpha to keep the same scaling
                    peft_config.lora_alpha *= 2**0.5

                if peft_config.rank_pattern:
                    peft_config.rank_pattern = {
                        key: 2 * val for key, val in peft_config.rank_pattern.items()
                    }
                if peft_config.alpha_pattern:
                    peft_config.alpha_pattern = {
                        key: 2 * val for key, val in peft_config.alpha_pattern.items()
                    }

            peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
        peft_config.inference_mode = inference_mode


import peft.peft_model as peft_model

peft_model.PeftModel.save_pretrained = patched_save_pretrained

print("Patched model save")

# %% [markdown]
# Add our kernel from A:

# %%
import triton
import triton.language as tl


@triton.jit
def gather_nf4_code(code_index, code_ptr):
    """In newer versions of triton, we attempt to use gather here on the code values.
    But since that's not available, we use this instead.
    """
    return tl.load(
        code_ptr + code_index, mask=code_index < 16, eviction_policy="evict_last"
    )


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 32}),
        # triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        # triton.Config({'BLOCK_SIZE': 256}),
    ],
    key=["total_elements"],
)
@triton.jit()
def _your_dequantize_nf4_kernel(
    weight_ptr,  # [uint8]  Quantized weights, 1 byte => 2 NF4 elements
    absmax_ptr,  # [int]  One int "index" per element
    absmax2_ptr,  # [float]  One absmax2 per block
    code_ptr,  # [float]  NF4 code lookup
    code2_ptr,  # [float]  Absmax code lookup
    offset_ptr,  # [float]  Offset to add after absmax is decoded
    out_ptr,  # [float]  Final dequantized output
    weight_blocksize: tl.constexpr,  # [int]   Number of elements in a weight block
    absmax_blocksize: tl.constexpr,  # [int]   Number of elements in a absmax block
    total_elements,  # [int]   Number of actual float elements to reconstruct
    BLOCK_SIZE: tl.constexpr,
):
    # context; we adopt the pov of a quantized byte, which corresponds to 2 elems
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    index = start + tl.arange(0, BLOCK_SIZE)

    offset = tl.load(offset_ptr, eviction_policy="evict_last")

    # the block index of the byte
    # weight_block_index = index // (weight_blocksize // 2) # naive approach
    # now we end up with a vector like (0, 0, ... 0, 0, 1, 1, ...)
    # we can shrink this by applying a stride of weight_blocksize // 2 so that our vector looks like (0, 1, 2, ...)
    weight_block_index = start // (weight_blocksize // 2) + tl.arange(
        0, BLOCK_SIZE // (weight_blocksize // 2)
    )
    absmax_block_index = (
        weight_block_index // absmax_blocksize
    )  # the block index of the absmax

    # mask calculations
    byte_index_mask = index < (total_elements // 2)
    weight_block_mask = weight_block_index < (total_elements // weight_blocksize)
    abs_block_mask = absmax_block_index < (
        total_elements // weight_blocksize // absmax_blocksize
    )

    # compute absmax for the block these weights belong to:
    absmax1_code = tl.load(
        absmax_ptr + weight_block_index,
        mask=weight_block_mask,
        eviction_policy="evict_last",
    )
    absmax1_code = absmax1_code.to(tl.int16)  # this is necessary

    # reading values from code is likely a very slow process due to lack of contiguity
    absmax1_val = tl.load(code2_ptr + absmax1_code, mask=(absmax1_code < 256))
    absmax2_val = tl.load(
        absmax2_ptr + absmax_block_index,
        mask=abs_block_mask,
        eviction_policy="evict_last",
    )

    absmax = tl.fma(absmax1_val, absmax2_val, offset)

    # lookup each weight bit
    byte = tl.load(
        weight_ptr + index, mask=byte_index_mask, eviction_policy="evict_first"
    )

    code_index2 = byte & 0x0F  # Lower 4 bits
    code_index1 = (byte >> 4) & 0x0F  # Upper 4 bits

    code_index_range = tl.arange(0, 16)  # 16 possible values
    _ = tl.load(
        code_ptr + code_index_range, eviction_policy="evict_last"
    )  # pull code contiguously into cache

    # interleaving the values here, prior to matmul, causes a slight decrease in average perf on T4 (poorer lower bound)
    # weight = dequantize_nf4_code(tl.interleave(code_index1, code_index2)).reshape(BLOCK_SIZE // (weight_blocksize // 2), weight_blocksize) * absmax[:, None]

    # Dequantize both 4-bit values
    weight1 = (
        gather_nf4_code(code_index1, code_ptr).reshape(
            BLOCK_SIZE // (weight_blocksize // 2), weight_blocksize // 2
        )
        * absmax[:, None]
    )
    weight2 = (
        gather_nf4_code(code_index2, code_ptr).reshape(
            BLOCK_SIZE // (weight_blocksize // 2), weight_blocksize // 2
        )
        * absmax[:, None]
    )

    # combine to prevent a non-contiguous write because of writing to even/odd indexes
    weight_index = start * 2 + tl.arange(0, BLOCK_SIZE * 2)
    weight = tl.interleave(weight1, weight2)
    tl.store(
        out_ptr + weight_index, tl.ravel(weight), mask=weight_index < total_elements
    )


NF4_WEIGHT_BLOCKSIZE = 64
NF4_ABSMAX_BLOCKSIZE = 256


def triton_dequantize_nf4(weight, quant_state):
    absmax = quant_state.absmax
    shape = quant_state.shape
    dtype = quant_state.dtype
    offset = quant_state.offset
    state2 = quant_state.state2
    absmax2 = state2.absmax
    code2 = state2.code

    out = torch.empty(shape, dtype=torch.float16, device="cuda")
    # cdiv: (x + y - 1) // y
    weight_grid = lambda meta: (
        (out.numel() // 2 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # taken from bitsandbytes get_4bit_type to ensure we use the same values
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/bitsandbytes/functional.py#L1075
    NF4_CODE_VALUES = torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=torch.float32,
        device="cuda",
    )

    _your_dequantize_nf4_kernel[weight_grid](
        weight,
        absmax,
        absmax2,
        NF4_CODE_VALUES,
        code2,
        offset,
        out,
        weight_blocksize=NF4_WEIGHT_BLOCKSIZE,
        absmax_blocksize=NF4_ABSMAX_BLOCKSIZE,
        total_elements=out.numel(),
    )

    # the fact that it could need a transpose is already handled, no need to replicate that
    return out


# %%
class Linear4bitCompilable(nn.Module):
    def __init__(self, bnb_linear):
        super().__init__()
        # Copy over quantized weight and metadata from the bitsandbytes layer
        self.in_features = bnb_linear.in_features
        self.out_features = bnb_linear.out_features
        self.quant_type = bnb_linear.weight.quant_type
        self.blocksize = bnb_linear.weight.blocksize
        self.weight = (
            bnb_linear.weight.data
        )  # don't trace param 4 bit, access its vars here
        self.quant_state = bnb_linear.weight.quant_state
        self.bias = bnb_linear.bias

        # Freeze weight and metadata as buffers (no grad needed)
        self.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mp policy means that quant_state.dtype may not match dtype expected by x
        # two different ways to handle the dequantization & typing
        # 1. dequantize in the dtype defined by quant state, then cast to x.dtype
        # 2. dequantize in the dtype of x
        # we're going with (1), which altho less performant is less complicated

        # Use the triton kernel for dequantization
        if self.quant_type == "nf4":
            # Dequantize the weights using the triton kernel
            # cast the weights to match the type of the input
            W = triton_dequantize_nf4(self.weight, self.quant_state).to(x.dtype)
        else:
            # For other quantization types (like fp4), we would need a different implementation
            raise NotImplementedError(
                f"Quantization type {self.quant_type} not supported yet"
            )

        # Compute linear output: x @ W^T + bias
        out = torch.matmul(x, W.t())
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


# %% [markdown]
# ## Building and training the model

from torch.distributed import DeviceMesh, init_process_group, get_rank
from datetime import timedelta
import argparse
import json


def parse_sft_args(sft_args_list):
    """Parses a list of 'key=value' strings into a dictionary."""
    overrides = {}
    if not sft_args_list:
        return overrides
    for arg in sft_args_list:
        try:
            key, value = arg.split("=", 1)
            overrides[key.strip()] = value.strip()
        except ValueError:
            print(
                f"Warning: Could not parse SFT arg '{arg}'. Expected format 'key=value'. Skipping."
            )
    return overrides


def convert_arg_types(overrides, base_config_obj):
    """
    Converts string values in overrides dict to match types in base_config_obj attributes.
    Uses json.loads for dict and list types for safety.
    """
    converted_overrides = {}
    for key, value_str in overrides.items():
        if not hasattr(base_config_obj, key):
            # Allow passing arbitrary kwargs not present in default SFTConfig if needed
            print(
                f"Warning: SFT arg '{key}' not found in base SFTConfig attributes. Passing as int/string."
            )
            converted_overrides[key] = (
                value_str if not value_str.isdigit() else int(value_str)
            )  # Or skip if strict matching is desired
            # print(
            #     f"Warning: SFT arg '{key}' not found in base SFTConfig attributes. Skipping."
            # )
            continue

        target_attr = getattr(base_config_obj, key)
        target_type = type(target_attr)

        try:
            if target_type == bool:
                if value_str.lower() in ("true", "1", "yes", "y"):
                    converted_value = True
                elif value_str.lower() in ("false", "0", "no", "n"):
                    converted_value = False
                else:
                    raise ValueError(f"Invalid boolean value: {value_str}")
            elif target_type == dict:
                try:
                    converted_value = json.loads(value_str)
                    if not isinstance(converted_value, dict):
                        raise ValueError("Parsed JSON is not a dictionary.")
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON for dict value: {value_str}")
            elif target_type == list:
                try:
                    converted_value = json.loads(value_str)
                    if not isinstance(converted_value, list):
                        raise ValueError("Parsed JSON is not a list.")
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON for list value: {value_str}")
            elif target_type == type(None) and value_str.lower() == "none":
                converted_value = None
            elif value_str.lower() == "none" and key in [
                "report_to"
            ]:  # Handle specific 'none' strings
                converted_value = "none"
            else:
                # General type conversion (int, float, str)
                converted_value = target_type(value_str)

            converted_overrides[key] = converted_value
        except (ValueError, TypeError) as e:
            print(
                f"Warning: Could not convert SFT arg '{key}={value_str}' to type {target_type}. Skipping. Error: {e}"
            )

    return converted_overrides


def main(args):
    # These are set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"--- Starting Rank {global_rank}/{world_size} (Local Rank {local_rank}) ---")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"Rank {global_rank} assigned to device: {device}")

    # init the process group for not just cuda (nccl) but also the cpu (because we want to use CPU offload)
    # https://github.com/pytorch/pytorch/issues/148532
    # https://github.com/pytorch/torchtitan/blob/aed946c8ef36e6b3d424d6bfd25fff5ca4a97e02/torchtitan/distributed/utils.py#L181-L182
    init_process_group("cuda:nccl,cpu:gloo", timeout=timedelta(seconds=1_800))
    rank = get_rank()

    # # 1d mesh
    mesh = DeviceMesh(
        device_type="cuda", mesh=torch.arange(int(os.environ["WORLD_SIZE"]))
    )

    # %%
    # similarly, let's set configs for FSDP2
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )

    # This mp policy is already enough to mess with sharding Linear4Bit
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16, reduce_dtype=torch.float16, cast_forward_inputs=True
    )

    if rank == 1:
        print("Building model")

    # %%
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map="auto",
        attn_implementation="sdpa",
        quantization_config=bnb_config,
    )

    # Get LoRA and setup model
    model = get_peft_model(model, lora_config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # if rank == 1:
    #     print("base qlora model built")

    # %%
    # patch modules for torch compile

    modules_to_replace = []
    for name, module in model.named_modules():
        module_type = str(type(module))
        # we'll handle lora modules later
        if "lora" in module_type.lower():
            continue
        if "bnb" in module_type or "bitsandbytes" in module_type:
            # print(name, type(module))
            modules_to_replace.append(name)

    for name in modules_to_replace:
        module = model.get_submodule(name)
        # create a new patched module from the old one
        new_module = Linear4bitCompilable(module)
        # shard before compiling
        fully_shard(
            new_module,
            reshard_after_forward=False,
            mp_policy=mp_policy,
            mesh=mesh,
            offload_policy=CPUOffloadPolicy(pin_memory=False),
        )
        new_module.forward = torch.compile(
            new_module.forward,
            fullgraph=False,
            dynamic=True,
            options=torch_compile_options,
        )
        # Find the direct parent module
        parts = name.split(".")
        parent = model
        for i in range(len(parts) - 1):
            parent = parent.get_submodule(parts[i])

        module_name = parts[-1]  # The attribute name within the parent
        setattr(parent, module_name, new_module)

    if rank == 1:
        print("Patched, sharded, and compiled Linear4bit base layer")

    # %% [markdown]
    # As noted in FSDP2 docs, `fully_shard` should be called bottom up and generally not on top level modules.
    # >  Calling fully_shard() on module constructs one group that includes the parameters in module.parameters() except those already assigned to a group from an earlier call on a submodule. This means that fully_shard() should be called bottom-up on your model. Each group’s parameters are all-gathered in one collective, and its gradients are reduce-scattered in one collective. Partitioning the model into multiple groups (“layer by layer”) allows for peak memory savings and communication/computation overlap. Users generally should not call fully_shard() only on the topmost root module.
    #
    # In following this, we next fully shard the Lora module, which needs to be handled a bit different from the Linear4bit "base module" which we just patched out. This module actually has gradients!
    for name, module in model.named_modules():
        module_type = str(type(module))

        if "peft.tuners.lora.bnb.Linear4bit".lower() in module_type.lower():
            # fully shard these modules
            fully_shard(
                module,
                reshard_after_forward=True,
                mp_policy=mp_policy,
                mesh=mesh,
                offload_policy=CPUOffloadPolicy(pin_memory=False),
            )
    if rank == 1:
        print("Sharded lora")

    # We compile the remaining top level modules. Layernorm is also sharded before compiling because it is outside of the "lora tree" that we have already sharded and compiled. This extends our coverage of sharding and compilation to all levels of the model.
    module_names = set()
    for index, child in model.base_model.model.model.layers.named_children():
        for name, module in child.named_children():
            if any(k in name for k in ["mlp", "self_atten"]):
                module = torch.compile(
                    module, fullgraph=False, dynamic=True, options=torch_compile_options
                )
                setattr(child, name, module)

            if any(k in name for k in ["layernorm"]):
                fully_shard(
                    module,
                    reshard_after_forward=False,
                    mp_policy=mp_policy,
                    mesh=mesh,
                    offload_policy=CPUOffloadPolicy(pin_memory=False),
                )
                module = torch.compile(
                    module, fullgraph=False, dynamic=True, options=torch_compile_options
                )
                setattr(child, name, module)

            module_names.add(name)

    if rank == 1:
        print("Compiled top level modules")

    # %%
    # model.model

    # %%
    from trl import SFTTrainer, SFTConfig
    import types

    # we want total train batch size to be effectively the same for both 1 and 2 gpus
    # total_train_batch_size = (
    #     self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    # )

    BASE_SFT_CONFIG_DEFAULTS = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 1,
        "max_steps": 10,
        "logging_steps": 1,
        "output_dir": "outputs",
        "seed": 3407,
        "max_seq_length": max_seq_length,
        "fp16": model.get_input_embeddings().weight.dtype == torch.float16,
        "bf16": model.get_input_embeddings().weight.dtype == torch.bfloat16,
        "report_to": "none",
        "dataset_num_proc": 4,
    }

    sft_configs = BASE_SFT_CONFIG_DEFAULTS.copy()
    sft_configs.update(
        convert_arg_types(
            parse_sft_args(args.sftargs), SFTConfig(**BASE_SFT_CONFIG_DEFAULTS)
        )
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            **sft_configs,
        ),
    )

    trainer._inner_training_loop = types.MethodType(
        patched_inner_training_loop, trainer
    )

    trainer.train()

    # %% [markdown]
    # The graph breaks are caused by the explicit `torch._dynamo.disable` calls in the `_pre_forward` and `post_forward` hooks in FSDP2. These are the communication methods. It did sound like pytorch already implemented compilable communication (https://www.youtube.com/watch?v=1XuibaVRewc) but I guess not. Not sure enabling this is in scope of the task.

    # %%
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSDP2 QLoRA Training Script with Flexible SFT Args"
    )
    # Generic argument for SFTConfig overrides
    parser.add_argument(
        "--sftargs",
        nargs="*",  # Accept 0 or more 'key=value' pairs
        help='Override SFTConfig parameters with key=value pairs (e.g., "max_steps=50" "logging_steps=5" "fsdp_config={\'limit_all_gathers\':False}"). Use quotes for args with spaces or special chars.',
        default=[],
    )
    try:
        main(parser.parse_args())
    except Exception as e:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise e
