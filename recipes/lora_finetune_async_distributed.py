# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
    MixedPrecision,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
    validate_state_dict_for_lora_async,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.utils import DummyProfiler, PROFILER_KEY

from tqdm import tqdm

from queue import Queue

import asyncio

import math

import torch.distributed as dist

log = utils.get_logger("DEBUG")


class QueueObject:
    def __init__(self, batch_idx, layer_idx, source, layer_input, lora_idx, labels=None):
        self.batch_idx = batch_idx
        self.layer_idx = layer_idx
        self.source = source
        self.input = layer_input
        self.lora_idx = lora_idx
        self.labels = labels

    def __repr__(self):
        return f"QueueObject(batch_number={self.batch_number}, layer_num={self.layer_num}, source='{self.source}')"



class LoRAFinetuneRecipeAsyncDistributed(FTRecipeInterface):
    """
    Distributed LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. DDP is currently not supported. Traning on CPU is not
            supported.

        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        ValueError: If world_size is 1
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        _, rank = utils.get_world_size_and_rank()

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        self._is_rank_zero = rank == 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # training attributes
        self._enable_activation_checkpointing = cfg.enable_activation_checkpointing

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self.num_adapters = len(cfg.model.lora_rank)

        # These are the queues for forward / backward / softmax passes that are yet to be executed
        self.fwd_queue = Queue()
        self.bwd_queue = Queue()
        self.softmax_queue = Queue()

        self.gather_handles = []

        # self.num_layers = num_layers

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        # When resuming from checkpoint for LoRA, the recipe expects the adapter weights
        # and recipe state to be present. The keys should match up with what ``save_checkpoint``
        # used to create these intermediate checkpoints
        if self._resume_from_checkpoint:
            if utils.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[utils.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[utils.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[utils.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[utils.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        # print(checkpoint_dict)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[utils.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer1 = self._setup_optimizer_async(
            cfg_optimizer=cfg.optimizer,
            idx = 0
        )

        self._optimizer2 = self._setup_optimizer_async(
            cfg_optimizer=cfg.optimizer,
            idx = 1
        )

        # self._optimizer = self._setup_optimizer(
        #     cfg_optimizer=cfg.optimizer,
        #     opt_state_dict=checkpoint_dict[utils.OPT_KEY]
        #     if self._resume_from_checkpoint
        #     else None,
        # )

        self._loss_fn = config.instantiate(cfg.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Total number of data points in the dataset
        total_data_points = len(self._dataloader.dataset)

        # Batch size
        batch_size = self._dataloader.batch_size

        # Calculate number of batches
        self.num_batches = math.ceil(total_data_points / batch_size)

        self.processed_batches = [0, 0]

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        # self._lr_scheduler = self._setup_lr_scheduler(
        #     cfg_lr_scheduler=cfg.lr_scheduler,
        #     num_training_steps=self.total_epochs * self._steps_per_epoch,
        #     last_epoch=self.global_step - 1,
        # )

        self._lr_scheduler1 = self._setup_lr_scheduler_async(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
            optimizer=self._optimizer1,
        )

        self._lr_scheduler2 = self._setup_lr_scheduler_async(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
            optimizer=self._optimizer2,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_profiler(
        self, cfg_profiler: DictConfig
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler: DictConfig - `profiler` section of the top-level `cfg` (the main config passed to `recipe.main`)

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.utils.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.utils.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.utils.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        if self._is_rank_zero:
            log.info(f" Profiler config after instantiation: {profiler_cfg}")

        return profiler

    def all_gather_next_layer_hook(self, module, input):
        """
        Hook that initiates AllGather for the next layer's weights during the forward pass.
        """
        fsdp_model = module.fsdp_model
        current_layer_index = module.layer_index

        if current_layer_index < len(fsdp_model.module.layers) - 1:
            next_layer = fsdp_model.module.layers[current_layer_index + 1]
            handles = []
            for param in next_layer.parameters():
                if hasattr(param, "_local_shard"):
                    handle = dist.all_gather(
                        tensors=[torch.zeros_like(param.data) for _ in range(dist.get_world_size())],
                        tensor=param.data,
                        async_op=True
                    )
                    handles.append(handle)
            fsdp_model.module.gather_handles = handles

    @staticmethod
    def custom_comm_hook(state, bucket, process_group=None):
        """
        Custom communication hook to initiate AllGather of weights for the next layer
        during the backward pass computation of the current layer.
        """
        fsdp_module = state.fsdp_module
        current_layer_index = state.current_layer_index

         # Ensure the output tensor is of the correct size and type
        output = torch.zeros(bucket.size(0) // dist.get_world_size(), device=bucket.device, dtype=bucket.dtype)

        # Ensure input tensors are all of the same type as the output
        input_list = [tensor.to(dtype=bucket.dtype) for tensor in bucket.chunk(dist.get_world_size())]

        # # Debugging: Check process_group and tensor shapes
        # print(f"Rank {dist.get_rank()}: process_group type: {type(process_group)}")
        # print(f"Rank {dist.get_rank()}: process_group: {process_group}")
        # print(f"Rank {dist.get_rank()}: Output tensor shape: {output.shape}, device: {output.device}")
        # for idx, tensor in enumerate(input_list):
        #     print(f"Rank {dist.get_rank()}: Input tensor {idx} shape: {tensor.shape}, device: {tensor.device}")

        if not isinstance(process_group, dist.ProcessGroup):
            process_group = dist.group.WORLD

        try:
            torch.cuda.synchronize()  # Ensure all prior CUDA work is complete
            future = dist.reduce_scatter(
                output,
                input_list=input_list,
                group=process_group,
                async_op=True
            )
        except RuntimeError as e:
            print(f"Rank {dist.get_rank()}: Error during reduce_scatter: {e}")
            torch.cuda.synchronize()
            raise e

        # Initiate AllGather for the next layer's weights if we are not at the last layer
        if current_layer_index > 0:  # Because we go backward, next layer in backprop is the previous index
            # print(f"current_layer_index is {current_layer_index}")
            next_layer = fsdp_module.module.layers[current_layer_index - 1]
            handles = []
            for param in next_layer.parameters():
                if hasattr(param, "_local_shard"):
                    # print(f"param is {param}")
                    gathered_tensors = [torch.zeros_like(param.data) for _ in range(dist.get_world_size())]
                    handle = dist.all_gather(
                        gathered_tensors,
                        param.data,
                        async_op=True
                    )
                    handles.append(handle)
            fsdp_module.module.gather_handles = handles
        
        return future


    def setup_hooks(self, fsdp_model):
        """
        Sets up custom communication hooks on the FSDP model.
        """
        # Store a reference to the fsdp model in the state
        state = type('', (), {})()  # Create a simple object to store state
        state.fsdp_module = fsdp_model
        state.current_layer_index = len(fsdp_model.module.layers)  # Start from the last layer
        
        # Register the custom communication hook
        fsdp_model.register_comm_hook(state=state, hook=self.custom_comm_hook)
        
        # Register pre-backward hooks to update the current layer index during backward pass
        for layer_index, layer in enumerate(fsdp_model.module.layers):
            def update_layer_index(module, input, layer_index=layer_index, state=state):
                setattr(state, 'current_layer_index', layer_index)
                
            layer.register_forward_pre_hook(update_layer_index)


    def wait_for_all_gather(self):
        for handle in self.gather_handles:
            handle.wait()
        self.gather_handles = []

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we load the model on CPU with the right
              dtype. To ensure that we don't instantiate ``world_size`` number of models,
              we initialize on meta_device for all ranks other than rank 0.
           b. Rank 0 is also responsible for calling ``load_state_dict`` and loading the
              model weights from checkpoint.
           c. While wrapping the model with FSDP, we set ``sync_module_states``
              to TRUE and broadcast module params and buffers from rank 0.
           d. The ``device_id`` param ensures that the FSDP initialization happens on
              the correct device.
        """

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        if self._is_rank_zero:
            log.info("FSDP is enabled. Instantiating Model on CPU for Rank 0 ...")
            init_start = time.perf_counter()

            with utils.set_default_dtype(self._dtype):
                model = config.instantiate(cfg_model)

            log.info(
                f"Model instantiation took {time.perf_counter() - init_start:.2f} secs"
            )

            # The model contains LoRA params which won't have any matching keys in
            # the state dict. As a result, we need to load with strict=False.
            # Before loading the state dict, ensure the state dict keys for the base
            # model and adapters (if available) match the keys in the full LoRA model
            # This is a good sanity check to prevent silent errors
            validate_state_dict_for_lora_async(
                lora_attn_modules=cfg_model.lora_attn_modules,
                apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
                apply_lora_to_output=getattr(cfg_model, "apply_lora_to_output", False),
                full_model_state_dict_keys=model.state_dict().keys(),
                lora_state_dict_keys=(
                    lora_weights_state_dict.keys()
                    if lora_weights_state_dict is not None
                    else None
                ),
                base_model_state_dict_keys=base_model_state_dict.keys(),
            )

            # Load both the base model weights and (if available) the adapter weights. Both
            # of this should happen only on Rank 0
            model.load_state_dict(base_model_state_dict, strict=False)
            if lora_weights_state_dict:
                model.load_state_dict(lora_weights_state_dict, strict=False)

        else:
            # For non-zero ranks, load the model on meta device
            with utils.set_default_dtype(self._dtype), torch.device("meta"):
                model = config.instantiate(cfg_model)

        if self._dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        self.num_layers = len(model.layers)

        # LoRA hyper-params needed for merging weights while saving checkpoints
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha

        # Note: this needs to be set before wrapping with FSDP
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        # print(self.adapter_params)
        # for k, v in model.named_parameters():
        #     if v.requires_grad:
        #         print(k)

        # Define a mixed precision configuration
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,   # Use FP16 for parameters
            reduce_dtype=torch.float16,  # Use FP16 for gradient reduction
            buffer_dtype=torch.float16   # Use FP16 for buffers
        )


        model = FSDP(
            module=model,
            auto_wrap_policy=utils.lora_fsdp_wrap_policy(
                modules_to_wrap={modules.TransformerDecoderLayer}
            ),
            sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            device_id=self._device,
            # this recipe does not currently support mixed precision training
            mixed_precision=mixed_precision,
            # Ensure we broadcast params and buffers from rank 0
            sync_module_states=True,
            # Initialize empty modules on all non-zero ranks
            param_init_fn=(
                lambda module: module.to_empty(
                    device=torch.device("cuda"), recurse=False
                )
                if not self._is_rank_zero
                else None
            ),
            backward_prefetch=torch.distributed.fsdp.BackwardPrefetch.BACKWARD_POST, #alternative: backward_pre
            forward_prefetch=True,
            use_orig_params=True,
        )

        # Ensure no params and buffers are on meta device
        utils.validate_no_params_on_meta_device(model)

        # self.setup_hooks(model)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )
        if self._is_rank_zero:
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        # This is inefficient since there is not need to instantiate the optimizer with all params as based model params are frozen.
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            # Note: technically we should check _contains_fsdp for
            # just the state dict of the adapter cfg, but should be equivalent
            opt_state_dict = FSDP.optim_state_dict_to_load(
                self._model, optimizer, opt_state_dict
            )
            optimizer.load_state_dict(opt_state_dict)

        if self._is_rank_zero:
            log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_optimizer_async(
        self, cfg_optimizer: DictConfig, idx: int
    ) -> Optimizer:
        # This is inefficient since there is not need to instantiate the optimizer with all params as based model params are frozen.
        active_params = [param for name, param in self._model.named_parameters() if f'lora_a_{idx}' in name or f'lora_a_{idx}' in name]
        optimizer = config.instantiate(cfg_optimizer, active_params)

        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")
        return lr_scheduler
    
    def _setup_lr_scheduler_async(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
        optimizer: Optimizer
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
        )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            )
            if not packed
            else None,
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        return sampler, dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        with FSDP.state_dict_type(
            self._model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state_dict = self._model.state_dict()
            if intermediate_checkpoint:
                opt_state_dict = FSDP.optim_state_dict(self._model, self._optimizer)
            else:
                opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

            # Filter out the adapter keys and weights from the model state dict. These will
            # be saved separately
            adapter_key_filter = lambda x: x in self.adapter_params
            adapter_state_dict = {
                k: v for k, v in cpu_state_dict.items() if adapter_key_filter(k)
            }
            checkpoint_dict.update({utils.ADAPTER_KEY: adapter_state_dict})

            # merge the adapter weights and base weights to create the model checkpoint
            merged_state_dict = get_merged_lora_ckpt(
                cpu_state_dict,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
            )
            checkpoint_dict.update({utils.MODEL_KEY: merged_state_dict})

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        utils.OPT_KEY: opt_state_dict,
                        utils.SEED_KEY: self.seed,
                        utils.EPOCHS_KEY: self.epochs_run,
                        utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                        utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )

            adapter_config = {
                "r": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "target_modules": get_lora_module_names(
                    self._lora_attn_modules,
                    self._apply_lora_to_mlp,
                    self._apply_lora_to_output,
                ),
                "peft_type": "LORA",
            }
            checkpoint_dict.update({utils.ADAPTER_CONFIG: adapter_config})

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        utils.cleanup_before_training()

        _, rank = utils.get_world_size_and_rank()

        # zero out the gradients before starting training
        self._optimizer1.zero_grad(set_to_none=True)
        # self._optimizer2.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        self._profiler.start()

        if self._is_rank_zero:
            self.print_memory_usage()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                # Both are shape [b, s]
                tokens, labels = batch["tokens"], batch["labels"]
                # Get the attention mask and position ids from the dataset if they
                # exist. Currently, only sample packing in PackedDataset returns these
                mask = batch.get("mask", None)  # shape [b, s, s]
                input_pos = batch.get("input_pos", None)  # shape [b, s]

                tokens = tokens.to(self._device)
                num_tokens += tokens.numel()
                labels = labels.to(self._device)
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                if self._is_rank_zero:
                    print("Finished processing inputs, now ready for forward pass")
                    self.print_memory_usage()
                logits = self._model(tokens, mask=mask, input_pos=input_pos)

                # self._model._free_full_params()  

                if self._is_rank_zero:
                    print("Finished forward pass")
                    self.print_memory_usage()
                # Shift so that tokens < n predict n
                # logits = logits[..., :-1, :].contiguous()
                # labels = labels[..., 1:].contiguous()
                # logits = logits.transpose(1, 2)

                # _, _, idx1, idx2 = logits.shape
                logits = logits[:, :, :-1, :]
                logits = logits.reshape(-1, logits.size(2), logits.size(3)).contiguous()
                # if self._is_rank_zero:
                #     print(logits.shape)
                labels = labels[..., 1:].contiguous()
                logits = logits.transpose(1, 2)
                
                # if self._is_rank_zero:
                #     print(f"label shape {labels.shape}, logits shape {logits.shape}")
                labels = labels.repeat(self.num_adapters, 1)
                # if self._is_rank_zero:
                #     print(f"label shape {labels.shape}, logits shape {logits.shape}")
                torch.cuda.empty_cache()
                if self._is_rank_zero:
                    print("Finished processing logits")
                    self.print_memory_usage()
                # print(torch.cuda.memory_summary())

                # Compute loss
                loss = self._loss_fn(logits, labels)
                # free logits otherwise it peaks backward memory
                del logits
                del labels

                loss = loss / self._gradient_accumulation_steps
                running_loss += loss
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    self._optimizer1.step()
                    self._optimizer1.zero_grad(set_to_none=True)
                    self._lr_scheduler1.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss.item()
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch+1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer1.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(utils.get_memory_stats(device=self._device))
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def print_memory_usage(self):
        """
        Utility function to print memory usage for the current device.
        """
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
        print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")


    def peek_queue(self, q):
        if not q.queue:
            return None
        return [q.queue[0]]

    async def safe_dispatch_iteration(self, task):
        try:
            await self.dispatch_iteration(task)
        except Exception as e:
            # Log the error with the relevant task information and the actual error message
            log.error(
                f"Error in task with batch_idx={task.batch_idx}, source={task.source}: {str(e)}",
                exc_info=True  # This will include the traceback in the log
            )
        
    async def train_by_step(self) -> None:
        """
        The core training loop, step by step.
        """
        # clean up before training begins
        utils.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer1.zero_grad()
        self._optimizer2.zero_grad()

        curr_epoch = self.epochs_run
        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        self._sampler.set_epoch(curr_epoch)
        
        self.processed_batches = [0, 0]

        self.fwd_queue.put(QueueObject(0, 0, "fwd", None, 0))
        self.fwd_queue.put(QueueObject(1, 0, "fwd", None, 0))
        # self.fwd_queue.put(QueueObject(0, 0, "fwd", None, 1))
        # self.fwd_queue.put(QueueObject(1, 0, "fwd", None, 1))

        tasks = set()
        priority_map = {'fwd': 1, 'softmax': 2, 'bwd': 3}

        while self.fwd_queue or self.bwd_queue or self.softmax_queue:
            fwd_ptr = self.peek_queue(self.fwd_queue) or []
            bwd_ptr = self.peek_queue(self.bwd_queue) or []
            softmax_ptr = self.peek_queue(self.softmax_queue) or []

            combined_queue = fwd_ptr + bwd_ptr + softmax_ptr
            sorted_list = sorted(combined_queue, key=lambda x: (x.batch_idx, priority_map[x.source]))

            if len(sorted_list) > 0:
                if len(tasks) < 2:
                    tasks.add(asyncio.create_task(self.safe_dispatch_iteration(sorted_list[0])))

                if len(sorted_list) > 1 and len(tasks) < 2:
                    tasks.add(asyncio.create_task(self.safe_dispatch_iteration(sorted_list[1])))

            if tasks:
                # Wait for the first completed task
                # done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for done_task in asyncio.as_completed(tasks):
                    await done_task
                    tasks.remove(done_task)


    def retrieve_data(self, batch_idx):
        # Calculate the start and end indices for the batch
        start_index = batch_idx * self._dataloader.batch_size
        end_index = start_index + self._dataloader.batch_size

        if self._is_rank_zero:
            print(f"Start index: {start_index}, End index: {end_index}")

        # Create a list of indices for the batch
        subset_indices = list(range(start_index, end_index))
        
        # Create a Subset from the dataset using the calculated indices
        subset = torch.utils.data.Subset(self._dataloader.dataset, subset_indices)
        
        # Create a DataLoader for this subset
        subset_loader = torch.utils.data.DataLoader(subset, batch_size=self._dataloader.batch_size, num_workers=0, 
                shuffle=False, collate_fn=partial(
                                            utils.padded_collate,
                                            padding_idx=self._tokenizer.pad_id,
                                            ignore_idx=self._loss_fn.ignore_index,
                                        ))
        
        # Retrieve the batch from the subset DataLoader
        batch = next(iter(subset_loader))

        # Process the batch as usual
        tokens = batch["tokens"]
        labels = batch["labels"]
        mask = batch.get("mask", None)
        input_pos = batch.get("input_pos", None)

        # if self._is_rank_zero:
        #     print(f"tokens are: {tokens}")
        #     print(f"labels are: {labels}")
        #     print(f"mask are: {mask}")

        # Convert to tensors if they are not already
        tokens = tokens.to(self._device)
        labels = labels.to(self._device)
        mask = mask.to(self._device) if mask is not None else None
        input_pos = input_pos.to(self._device) if input_pos is not None else None

        # if self._is_rank_zero:
        #     print(f"tokens shapes are: {tokens.shape}")
        #     print(f"labels shapes are: {labels.shape}")

        return tokens, mask, input_pos, labels


    async def dispatch_layer(self, item):
        if item.source == "fwd":
            new_input = self.layers[item.layer_idx](layer_input)

            if layer_idx == self.num_layers - 1:
                # TODO: handle the output projection here

                self.softmax_queue.put(QueueObject(item.batch_idx, -1, "softmax", new_input))
            else:
                self.fwd_queue.put(QueueObject(item.batch_idx, item.layer_idx+1, "fwd", new_input))
        elif item.source == "bwd":
            if layer_idx == self.num_layers - 1:
                #TODO: handle 
                loss = criterion(activation, target)
                loss.backward(retain_graph=True)


            if layer_idx == 0:
                # TODO: handle the output projection backprop here

                # Get new input
                self.fwd_queue.put(QueueObject(item.batch_idx+1, 0, "fwd", new_input))
            else:
                self.bwd_queue.put(QueueObject(item.batch_idx, item.layer_idx-1, "bwd", new_input))
        elif item.source == "softmax":

            logits = self.layer_input[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            logits = logits.transpose(1, 2)

            # print(f"label shape {labels.shape}, logits shape {logits.shape}")
            labels = labels.repeat(self.num_adapters, 1)
            # Compute loss
            loss = self._loss_fn(logits, labels)
            # free logits otherwise it peaks backward memory
            del logits

            loss = loss / self._gradient_accumulation_steps
            
            self.bwd_queue.put(QueueObject(item.batch_idx, self.num_layers - 1, "bwd", loss))

    async def dispatch_iteration(self, item):
        if item.source == "fwd":
            if self._is_rank_zero:
                print(f"start fwd batch {item.batch_idx}")
            self.fwd_queue.get()
            tokens, mask, input_pos, labels = self.retrieve_data(item.batch_idx)
            logits = self._model(tokens, mask=mask, input_pos=input_pos, activated=item.lora_idx)
            self.softmax_queue.put(QueueObject(item.batch_idx, -1, "softmax", logits, item.lora_idx, labels=labels))
            # self.fwd_queue.put(QueueObject(item.batch_idx+1, 0, "fwd", None, item.lora_idx))
            if self._is_rank_zero:
                print(f"end fwd batch {item.batch_idx}")
        
        elif item.source == "bwd":
            if self._is_rank_zero:
                print(f"start bwd batch {item.batch_idx}")
            self.bwd_queue.get()
            # item.input.backward()

            # Step with optimizer
            if (item.batch_idx + 1) % self._gradient_accumulation_steps == 0:
                getattr(self, f"_optimizer{item.lora_idx+1}").step()
                getattr(self, f"_optimizer{item.lora_idx+1}").zero_grad(set_to_none=True)
                getattr(self, f"_lr_scheduler{item.lora_idx+1}").step()

                # Update the number of steps when the weights are updated
                self.global_step += 1
            
            self.processed_batches[item.lora_idx] += 1
            # Get new input
            if self.processed_batches[item.lora_idx] < self.num_batches * (self.total_epochs - self.epochs_run):
                self.fwd_queue.put(QueueObject(item.batch_idx+1, 0, "fwd", None, item.lora_idx))
            if self._is_rank_zero:
                print(f"end bwd batch {item.batch_idx}")

        elif item.source == "softmax":
            if self._is_rank_zero:
                print(f"start softmax batch {item.batch_idx}")
            self.softmax_queue.get()
            
            logits = item.input[:, :, :-1, :]
            logits = logits.reshape(-1, logits.size(2), logits.size(3))
            labels = item.labels[..., 1:].contiguous()
            logits = logits.transpose(1, 2).contiguous()
            
            labels = labels.repeat(self.num_adapters, 1)
            
            # Compute loss
            loss = self._loss_fn(logits, labels)
            if self._is_rank_zero:
                print(f"softmax batch {item.batch_idx} compute loss")
            # free logits otherwise it peaks backward memory
            del logits
            del labels
            
            loss = loss / self._gradient_accumulation_steps

            # print(f"do loss here {item.batch_idx}")

            # loss.backward(retain_graph=True)
            # self.bwd_queue.put(QueueObject(item.batch_idx, self.num_layers - 1, "bwd", loss, item.lora_idx))
            self.fwd_queue.put(QueueObject(item.batch_idx+1, 0, "fwd", None, item.lora_idx))
            
            if self._is_rank_zero:
                print(f"end softmax batch {item.batch_idx}")
    

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not utils.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")

    config.log_config(recipe_name="LoRAFinetuneRecipeAsyncDistributed", cfg=cfg)

    recipe = LoRAFinetuneRecipeAsyncDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    # asyncio.run(recipe.train_by_step())
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
