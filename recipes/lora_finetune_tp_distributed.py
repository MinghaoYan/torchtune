import os
import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed.tensor.parallel as tp
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module, SequenceParallel, ParallelStyle, PrepareModuleInput
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor import Replicate, Shard, DTensor, distribute_tensor
from torch.distributed._tensor import DeviceMesh, distribute_module
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

import math

import torch.distributed as dist

log = utils.get_logger("DEBUG")


# class InterleavedLoRALinearParallel(ParallelStyle):
#     def __init__(self, input_shard_dim=1, output_shard_dim=0):
#         super().__init__()
#         self.input_shard_dim = input_shard_dim
#         self.output_shard_dim = output_shard_dim

#     def _apply(self, module, device_mesh):
#         # Shard the base weight across devices
#         if hasattr(module, 'weight') and module.weight is not None:
#             # Split and shard the weight tensor
#             shards = module.weight.chunk(device_mesh.size(0), dim=self.output_shard_dim)
#             local_shard = shards[device_mesh.get_rank()]
#             module.weight = Parameter(
#                 DTensor.from_local(
#                     local_shard.contiguous(),  # Ensure the tensor is contiguous
#                     device_mesh=device_mesh,
#                     placements=[Shard(self.output_shard_dim)]
#                 )
#             )

#         # Shard lora_a weights
#         for lora_a in module.lora_a:
#             if lora_a.weight is not None:
#                 shards = lora_a.weight.chunk(device_mesh.size(0), dim=self.input_shard_dim)
#                 local_shard = shards[device_mesh.get_rank()]
#                 lora_a.weight = Parameter(
#                     DTensor.from_local(
#                         local_shard.contiguous(),
#                         device_mesh=device_mesh,
#                         placements=[Shard(self.input_shard_dim)]
#                     )
#                 )

#         # Shard lora_b weights
#         for lora_b in module.lora_b:
#             if lora_b.weight is not None:
#                 shards = lora_b.weight.chunk(device_mesh.size(0), dim=self.output_shard_dim)
#                 local_shard = shards[device_mesh.get_rank()]
#                 lora_b.weight = Parameter(
#                     DTensor.from_local(
#                         local_shard.contiguous(),
#                         device_mesh=device_mesh,
#                         placements=[Shard(self.output_shard_dim)]
#                     )
#                 )


# class LoRALinearColColParallel(ParallelStyle):
#     def __init__(self, input_shard_dim=1, output_shard_dim=0):
#         super().__init__()
#         self.input_shard_dim = input_shard_dim
#         self.output_shard_dim = output_shard_dim

#     def _partition_lora_fn(self, module, device_mesh):
#         """Manually partition LoRA weights in a Colwise manner."""
#         rank = device_mesh.get_rank()
#         world_size = device_mesh.size(0)
        
#         if hasattr(module, 'weight') and module.weight is not None:
#             shards = module.weight.chunk(world_size, dim=self.output_shard_dim)
#             local_weight_shard = shards[rank]
#             dist_weight = nn.Parameter(distribute_tensor(local_weight_shard, device_mesh, [Shard(self.output_shard_dim)]))
#             module.register_parameter("weight", dist_weight)
#             # print(f"[Rank {rank}] Assigned shard shape for {name} after distribute_tensor: {dist_param.shape}")

#     def _apply(self, module, device_mesh):
#         rank = device_mesh.get_rank()
#         world_size = device_mesh.size(0)

#         # Distribute the main weight matrix if needed
#         if hasattr(module, 'weight') and module.weight is not None:
#             shards = module.weight.chunk(world_size, dim=self.output_shard_dim)
#             local_weight_shard = shards[rank]
#             dist_weight = nn.Parameter(distribute_tensor(local_weight_shard, device_mesh, [Shard(self.output_shard_dim)]))
#             module.register_parameter("weight", dist_weight)
#             # print(f"[Rank {rank}] Main weight shard shape: {dist_weight.shape}")

#         # Apply column-wise partitioning to each LoRA module (lora_a and lora_b)
#         for lora_a in module.lora_a:
#             ColwiseParallel._apply(ColwiseParallel, lora_a, device_mesh)

#         for lora_b in module.lora_b:
#             ColwiseParallel._apply(ColwiseParallel, lora_b, device_mesh)


# class LoRALinearRowColParallel(ParallelStyle):
#     def __init__(self, input_shard_dim=1, output_shard_dim=0):
#         super().__init__()
#         self.input_shard_dim = input_shard_dim
#         self.output_shard_dim = output_shard_dim

#     def _partition_lora_fn(self, lora_module, device_mesh, shard_dim):
#         """Manually partition LoRA weights in a specified manner."""
#         rank = device_mesh.get_rank()
#         world_size = device_mesh.size(0)
        
#         for name, param in lora_module.named_parameters():
#             # Manually chunk the tensor along the specified dimension
#             shards = param.chunk(world_size, dim=shard_dim)
#             local_shard = shards[rank].detach()  # Detach to ensure it's a leaf tensor
            
#             # Apply distribute_tensor to the local shard only
#             dist_param = nn.Parameter(distribute_tensor(local_shard, device_mesh, [Shard(shard_dim)]))
#             lora_module.register_parameter(name, dist_param)

#     def _apply(self, module, device_mesh):
#         rank = device_mesh.get_rank()
#         world_size = device_mesh.size(0)

#         # Apply row-wise partitioning to the main weight matrix
#         if hasattr(module, 'weight') and module.weight is not None:
#             weight_shards = module.weight.chunk(world_size, dim=self.input_shard_dim)
#             local_weight_shard = weight_shards[rank]
#             dist_weight = nn.Parameter(distribute_tensor(local_weight_shard, device_mesh, [Shard(self.input_shard_dim)]))
#             module.register_parameter("weight", dist_weight)

#         # Apply row-wise partitioning to each LoRA module's `lora_a`
#         for lora_a in module.lora_a:
#             RowwiseParallel._apply(RowwiseParallel, lora_a, device_mesh)

#         for lora_b in module.lora_b:
#             ColwiseParallel._apply(ColwiseParallel, lora_b, device_mesh)


class LoRALinearColColParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a column-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it together with RowwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Colwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w1" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w1" Linear will be converted to Replicated DTensor
        >>> # and the output of "w1" will return :class:`torch.Tensor` that shards on the last dim.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel()})
        >>> ...

    .. note:: By default ``ColwiseParallel`` output is sharded on the last dimension if the ``output_layouts`` not
        specified, if there're operators that require specific tensor shape (i.e. before the paired ``RowwiseParallel``),
        keep in mind that if the output is sharded the operator might need to be adjusted to the sharded size.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        # if isinstance(module, nn.Dropout) or isinstance(module, nn.ModuleList):
        #     return
        # else:
        #     module.register_parameter("weight", nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(0)])))
        # for name, param in module.named_parameters():
        if hasattr(module, 'weight') and module.weight is not None:
            # print(module)
            dist_param = nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(0)]))
            module.register_parameter("weight", dist_param)


    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:

        return distribute_module(
            module,
            device_mesh,
            self._partition_linear_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class LoRALinearRowColParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w2" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w2" Linear will be converted to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w2": RowwiseParallel()}),
        >>> ...
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        if isinstance(module, nn.Dropout) or isinstance(module, nn.ModuleList):
            return
        else:
            module.register_parameter(
                "weight",
                nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(1)])),
            )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        partition_fn = self._partition_linear_fn
        # rowwise linear runtime sharding requires input tensor shard on last dim
        self.desired_input_layouts: Tuple[Placement, ...] = (Shard(-1),)
        
        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class LoRAFinetuneRecipeTPDistributed(FTRecipeInterface):
    """
    Distributed LoRA finetuning recipe for dense transformer-based LLMs such as Llama2.
    This recipe supports distributed training and can be run on a single node (1 to 8 GPUs).
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
            # _update_recipe_state will throw an exception if the recipe state is not correctly loaded
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

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=checkpoint_dict[utils.OPT_KEY]
            if self._resume_from_checkpoint
            else None,
        )

        self._loss_fn = config.instantiate(cfg.loss)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._dataloader = self._setup_data(
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
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_profiler(
        self, cfg_profiler: DictConfig
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
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

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Model initialization for tensor parallelism.
        """
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        # Initialize model on CPU for all ranks
        if self._is_rank_zero:
            log.info("Initializing Model on CPU for Rank 0 ...")

        init_start = time.perf_counter()

        # Define the device mesh for tensor parallelism
        world_size = torch.distributed.get_world_size()
        device_ids = list(range(world_size))
        self.device_mesh = DeviceMesh('cuda', device_ids)

        cfg_model.device_ids = device_ids
        with utils.set_default_dtype(self._dtype):
            model = config.instantiate(cfg_model)
        
        log.info(f"device mesh size 0 is {self.device_mesh.size(0)}")

        log.info(f"Model instantiation took {time.perf_counter() - init_start:.2f} secs")

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

        # Load base model weights
        log.info("Start loading model state dict")
        model.load_state_dict(base_model_state_dict, strict=False)
        if lora_weights_state_dict:
            model.load_state_dict(lora_weights_state_dict, strict=False)
        log.info("Finish loading model state dict")

        if self._dtype == torch.bfloat16 or self._dtype == torch.float16:
            model = model.to(self._dtype)

        self.num_layers = len(model.layers)

        # LoRA hyper-params
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha

        # Set trainable parameters
        self.adapter_params = get_adapter_params(model)

        # Check if each parameter is a leaf
        for name, param in model.named_parameters():
            if not param.is_leaf:
                print(f"{param} is not a leaf")

        set_trainable_params(model, self.adapter_params)
        for name, param in model.named_parameters():
            if not param.is_leaf:
                print(f"after setting trainable params, {param} is not leaf")

        # Combined Tensor Parallelism plan
        # Base Tensor Parallelism plan
        full_tp_plan = {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),  # Ensure weights are replicated for embedding
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                use_local_output=False,
            ),
        }

        # Loop over each layer to add to the plan
        for idx in range(len(model.layers)):
            layer_prefix = f"layers.{idx}"
            
            # Add layer-specific parallelization strategies
            full_tp_plan[f"{layer_prefix}.sa_norm"] = SequenceParallel(sequence_dim=1)
            full_tp_plan[f"{layer_prefix}.attn.q_proj"] = LoRALinearColColParallel()
            # Include other attention projections if needed
            full_tp_plan[f"{layer_prefix}.attn.k_proj"] = LoRALinearColColParallel()
            full_tp_plan[f"{layer_prefix}.attn.v_proj"] = LoRALinearColColParallel()
            full_tp_plan[f"{layer_prefix}.attn.output_proj"] = LoRALinearColColParallel()
            full_tp_plan[f"{layer_prefix}.mlp_norm"] = SequenceParallel(sequence_dim=1)
            full_tp_plan[f"{layer_prefix}.mlp.w1"] = LoRALinearColColParallel()
            full_tp_plan[f"{layer_prefix}.mlp.w2"] = LoRALinearColColParallel()
            full_tp_plan[f"{layer_prefix}.mlp.w3"] = LoRALinearColColParallel()
        
            for idx in range(self.num_adapters):
                full_tp_plan[f"{layer_prefix}.attn.q_proj.lora_a.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.attn.q_proj.lora_b.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.attn.k_proj.lora_a.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.attn.k_proj.lora_b.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.attn.v_proj.lora_a.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.attn.v_proj.lora_b.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.attn.output_proj.lora_a.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.attn.output_proj.lora_b.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.mlp.w1.lora_a.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.mlp.w1.lora_b.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.mlp.w2.lora_a.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.mlp.w2.lora_b.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.mlp.w3.lora_a.{idx}"] = LoRALinearColColParallel()
                full_tp_plan[f"{layer_prefix}.mlp.w3.lora_b.{idx}"] = LoRALinearColColParallel()


        log.info("Start parallelizing layers")
        # print(f"{model}")
        # Apply parallelization using the `parallelize_module` function
        model = parallelize_module(
            module=model,
            device_mesh=self.device_mesh,
            parallelize_plan=full_tp_plan,
        )

        for i, lora_a in enumerate(model.layers[0].attn.q_proj.lora_a):
            print(f"[Rank {self.device_mesh.get_rank()}] LoRA A1[{i}] weight shape after parallelize_module: {lora_a.weight.shape}")

        log.info("Finish parallelizing module")

        # set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        if self._is_rank_zero:
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        torch.distributed.barrier()

        return model


    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        # Assuming `self.adapter_params` is a list of parameters
        # for idx, param in enumerate(self.adapter_params):
        #     if not isinstance(param, (torch.Tensor, nn.Parameter)):
        #         print(f"Non-tensor item found at index {idx}: {param} (type: {type(param)})")

        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)
        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
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

        # torch.seed()

        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
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

        return dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_dict = {}

        # Collect the adapter state dict
        adapter_state_dict = {
            k: v.cpu() for k, v in self._model.state_dict().items() if k in self.adapter_params
        }
        checkpoint_dict.update({utils.ADAPTER_KEY: adapter_state_dict})

        # Save optimizer state if intermediate checkpoint
        intermediate_checkpoint = epoch + 1 < self.total_epochs
        if intermediate_checkpoint:
            opt_state_dict = self._optimizer.state_dict()
            checkpoint_dict.update({utils.OPT_KEY: opt_state_dict})
            checkpoint_dict.update({
                utils.SEED_KEY: self.seed,
                utils.EPOCHS_KEY: self.epochs_run,
                utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
            })

        if self._is_rank_zero:
            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )

    def train(self) -> None:
        """
        The core training loop.
        """
        # Clean up before training begins
        utils.cleanup_before_training()

        _, rank = utils.get_world_size_and_rank()

        # Zero out the gradients before starting training
        self._optimizer.zero_grad(set_to_none=True)

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # Collect all LoRA modules in the model
        lora_modules = []
        for module in self._model.modules():
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b'):
                lora_modules.append(module)

        self._profiler.start()

        for curr_epoch in range(self.epochs_run, self.total_epochs):

            pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                tokens, labels = batch["tokens"], batch["labels"]
                mask = batch.get("mask", None)
                input_pos = batch.get("input_pos", None)

                tokens = tokens.to(self._device)
                num_tokens += tokens.numel()
                labels = labels.to(self._device)
                mask = mask.to(self._device) if mask is not None else None
                input_pos = (
                    input_pos.to(self._device) if input_pos is not None else None
                )

                # Repeat inputs for all adapters
                tokens_repeated = tokens.repeat(self.num_adapters, 1).to(self.device_mesh.device_type)
                labels_repeated = labels.repeat(self.num_adapters, 1).to(self.device_mesh.device_type)
                if mask is not None:
                    mask_repeated = mask.repeat(self.num_adapters, 1).to(self.device_mesh.device_type)
                else:
                    mask_repeated = None
                if input_pos is not None:
                    input_pos_repeated = input_pos.repeat(self.num_adapters, 1).to(self.device_mesh.device_type)
                else:
                    input_pos_repeated = None

                max_seq_len = tokens_repeated.shape[1]
                if max_seq_len % self.device_mesh.size(0) != 0:
                    # Calculate padding to make sequence length divisible by the number of ranks
                    pad_len = self.device_mesh.size(0) - (max_seq_len % self.device_mesh.size(0))
                    tokens_repeated = F.pad(tokens_repeated, (0, pad_len))  # pad on sequence dimension
                print(f"tokens_repeated shape is {tokens_repeated.shape}")
                # Ensure tokens_repeated is fully replicated before embedding layer
                tokens_repeated = DTensor.from_local(
                    tokens_repeated.contiguous(),
                    device_mesh=self.device_mesh,
                    placements=[Replicate()]  # Ensure full replication across ranks
                )

                # # Move to the correct device
                # tokens_repeated = tokens_repeated.to(self.device_mesh.device_type)

                # Repeat for labels, mask, and input_pos
                shards = labels_repeated.chunk(self.device_mesh.size(0), dim=0)
                local_shard = shards[self.device_mesh.get_rank()]
                labels_repeated = DTensor.from_local(
                    local_shard.contiguous(),
                    device_mesh=self.device_mesh,
                    placements=[Shard(0)]
                )
                labels_repeated = labels_repeated.to(self.device_mesh.device_type)


                # Perform one forward pass
                log.info("start forward pass")
                log.info(f"token repeated shape is {tokens_repeated.shape}")
                logits = self._model(tokens_repeated, mask=mask_repeated, input_pos=input_pos_repeated)
                log.info("finish forward pass")
                logits = logits[..., :-1, :].contiguous()
                logits = logits.transpose(1, 2)
                labels_shifted = labels_repeated[..., 1:].contiguous()

                bsz = tokens.size(0)

                # Initialize accumulators for gradients
                accum_lora_a_grads = {module: torch.zeros_like(module.lora_a.weight) for module in lora_modules}
                accum_lora_b_grads = {module: torch.zeros_like(module.lora_b.weight) for module in lora_modules}

                # Perform separate backward passes for each adapter
                for i in range(self.num_adapters):
                    lora_output = logits[i * bsz:(i + 1) * bsz, ...]
                    lora_labels = labels_shifted[i * bsz:(i + 1) * bsz, ...]

                    # Compute loss
                    log.info("start computing loss")
                    loss = self._loss_fn(lora_output, lora_labels)
                    loss = loss / self._gradient_accumulation_steps
                    running_loss += loss.item()
                    log.info("finish computing loss")

                    # Zero out gradients of LoRA parameters
                    for module in lora_modules:
                        module.lora_a.weight.grad = None
                        module.lora_b.weight.grad = None

                    # Compute gradients w.r.t. LoRA parameters
                    log.info("start backward pass")
                    loss.backward(retain_graph=True)
                    log.info("finish backward pass")

                    # Accumulate gradients
                    for module in lora_modules:
                        if module.lora_a.weight.grad is not None:
                            accum_lora_a_grads[module] += module.lora_a.weight.grad.clone()
                        if module.lora_b.weight.grad is not None:
                            accum_lora_b_grads[module] += module.lora_b.weight.grad.clone()

                # After processing all adapters, assign accumulated gradients
                for module in lora_modules:
                    module.lora_a.weight.grad = accum_lora_a_grads[module]
                    module.lora_b.weight.grad = accum_lora_b_grads[module]

                # Clear intermediate variables to free memory
                del logits, labels_shifted, tokens_repeated, labels_repeated
                torch.cuda.empty_cache()

                # Gradient accumulation and optimizer step
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    # Clip gradients if necessary
                    torch.nn.utils.clip_grad_norm_(self.adapter_params, max_norm=1.0)

                    # Optimizer step
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()

                    self.global_step += 1

                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch+1}|{self.global_step}|Loss: {running_loss:.4f}"
                    )

                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": running_loss,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(utils.get_memory_stats(device=self._device))
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    self._profiler.step()

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()


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

    config.log_config(recipe_name="LoRAFinetuneRecipeTPDistributed", cfg=cfg)

    recipe = LoRAFinetuneRecipeTPDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()

    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
