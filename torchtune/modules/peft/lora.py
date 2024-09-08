# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import List

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune.modules.low_precision import _register_nf4_dispatch_ops  # noqa: F401
from torchtune.modules.peft.peft_utils import AdapterModule
from torchtune import utils
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import torch.distributed as dist

class LoRALinear(nn.Module, AdapterModule):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: [int],
        alpha: [float],
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
        bsz: int = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        self.bsz = bsz
        weight, bias = self._create_weight_and_bias()
        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout)
        # self.lora_a = nn.Linear(in_features=in_dim, out_features=sum(self.rank), bias=False)
        # self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        for idx in range(len(self.rank)):
            setattr(self, f'lora_a_{idx}', nn.Linear(in_features=in_dim, out_features=self.rank[idx], bias=False))
            setattr(self, f'lora_b_{idx}', nn.Linear(in_features=self.rank[idx], out_features=out_dim, bias=False))

        self.merged = False
        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        # _lora_a_init_params(self.lora_a)
        for idx in range(len(self.rank)):
            _lora_a_init_params(getattr(self, f"lora_a_{idx}"))
            _lora_b_init_params(getattr(self, f"lora_b_{idx}"))

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not self._quantize_base else to_nf4(linear.weight)
        bias = None
        if self.use_bias:
            if self._quantize_base:
                raise NotImplementedError(
                    "Quantized LoRALinear does not support bias at the moment."
                )
            bias = linear.bias
        return weight, bias

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = []
        for idx in range(len(self.rank)):
            adapter_params.append(f"lora_a_{idx}.weight")
            adapter_params.append(f"lora_b_{idx}.weight")
        return adapter_params

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, self.bias)
        if self.disabled:
            return out

        # Handle first layer
        # total_dim = len(self.rank) * 
        # if x.shape[0] == 1:
        # print(f"Lora input dim is {x.shape}")
        lora_out = []
        num_adapters = len(self.rank)
        # print(x.shape[0], self.bsz, x.shape[0] == self.bsz)
        if x.shape[0] == self.bsz and num_adapters > 1:
            x = x.repeat(num_adapters, 1, 1)
        # print(f"Lora input dim after repeat is {x.shape}")
        after_dropout = self.dropout(x)
        bsz = x.shape[0] // num_adapters
        for idx in range(num_adapters):
            lora_a_slice = after_dropout[idx * bsz: (idx + 1) * bsz, :, :]
            # print(f"lora_a_slice dim is {lora_a_slice.shape}")
            lora_after_a = getattr(self, f'lora_a_{idx}')(lora_a_slice)
            # print(f"Lora after a dim is {lora_after_a.shape}")
            # print(f"out dim is {out.shape}")
            lora_after_b = getattr(self, f'lora_b_{idx}')(lora_after_a)
            # print(f"Lora after b dim is {lora_after_b.shape}")
            scaled_results = (self.alpha[idx] / self.rank[idx]) * lora_after_b
            # print(f"scaled_results dim is {scaled_results.shape}")
            if out.shape[0] > self.bsz:
                final_results = scaled_results + out[idx * bsz: (idx + 1) * bsz, :, :]
            else:
                final_results = scaled_results + out
            # print(f"final_results dim is {final_results.shape}")
            lora_out.append(final_results)

        
        # return lora_out
        # lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        # for out in lora_out:
        #     print(out.shape)
        total_out = torch.stack(lora_out, dim=0)
        # print(total_out.shape)
        
        return total_out


def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)



class InterleavedLoRALinear(nn.Module, AdapterModule):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: [int],
        alpha: [float],
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
        bsz: int = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        self.bsz = bsz
        weight, bias = self._create_weight_and_bias()
        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=sum(self.rank), bias=False)
        self.lora_b = nn.Linear(in_features=sum(self.rank), out_features=out_dim, bias=False)

        self.world_size, self.device_rank = utils.get_world_size_and_rank()

        assert(len(self.rank) // self.world_size == 0, "Must evenly divide num lora adapters and world size")
        # Create a rank-specific mask for the weight matrix
        self.start_row = sum(self.rank[:self.device_rank])
        self.end_row = sum(self.rank[:self.device_rank]) + self.rank[self.device_rank]

        # for idx in range(len(self.rank)):
        #     setattr(self, f'lora_a_0_{idx}', nn.Linear(in_features=in_dim, out_features=self.rank[idx], bias=False))
        #     setattr(self, f'lora_b_0_{idx}', nn.Linear(in_features=self.rank[idx], out_features=out_dim, bias=False))
        #     setattr(self, f'lora_a_1_{idx}', nn.Linear(in_features=in_dim, out_features=self.rank[idx], bias=False))
        #     setattr(self, f'lora_b_1_{idx}', nn.Linear(in_features=self.rank[idx], out_features=out_dim, bias=False))

        self.merged = False
        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)
        # for idx in range(len(self.rank)):
        #     _lora_a_init_params(getattr(self, f"lora_a_0_{idx}"))
        #     _lora_b_init_params(getattr(self, f"lora_b_0_{idx}"))
        #     _lora_a_init_params(getattr(self, f"lora_a_1_{idx}"))
        #     _lora_b_init_params(getattr(self, f"lora_b_1_{idx}"))


    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not self._quantize_base else to_nf4(linear.weight)
        bias = None
        if self.use_bias:
            if self._quantize_base:
                raise NotImplementedError(
                    "Quantized LoRALinear does not support bias at the moment."
                )
            bias = linear.bias
        return weight, bias

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        # for idx in range(len(self.rank)):
        #     adapter_params.append(f"lora_a_0_{idx}.weight")
        #     adapter_params.append(f"lora_b_0_{idx}.weight")
        #     adapter_params.append(f"lora_a_1_{idx}.weight")
        #     adapter_params.append(f"lora_b_1_{idx}.weight")
        return adapter_params

    
    def all_gather_lora_b_weight(self):
        # Ensure all the workers are synchronized
        dist.barrier()

        # Get the local weight (sharded part)
        local_weight = self.lora_b.weight

        # Create an empty list of tensors for gathering the weights from all ranks
        gathered_weights = [torch.zeros_like(local_weight) for _ in range(dist.get_world_size())]

        # Perform the all-gather operation across all ranks
        dist.all_gather(gathered_weights, local_weight)

        # Concatenate the gathered shards into the full weight
        full_weight = torch.cat(gathered_weights, dim=0)

        return full_weight


    def forward(self, x: Tensor, activated: int = 0) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, None)
        if self.disabled:
            return out

        # print(f"x.shape: {x.shape}")
        # print(f"self.mask_a.shape: {self.mask_a.shape}")

        lora_a_out = self.lora_a(self.dropout(x)) 
        
        # Initialize output accumulator for merging multiple LoRA paths
        lora_out_total = torch.zeros(out.shape, device=x.device, dtype=x.dtype)

        # Manually gather the full weight matrix across all ranks
        gathered_weight = self.all_gather_lora_b_weight()

        # Ensure that `gathered_weight` has the correct shape (out_dim, in_dim)
        gathered_weight = gathered_weight.view(self.lora_b.out_features, self.lora_b.in_features).to(lora_a_out.dtype)

        # Apply the LoRA process separately for each rank in self.rank
        start_idx = 0
        for i, rank_i in enumerate(self.rank):
            # Determine the range for the current rank
            end_idx = start_idx + rank_i

            # Create a mask for this specific rank
            mask = torch.zeros_like(lora_a_out)
            mask[:, :, start_idx:end_idx].fill_(1.0)

            # Mask and process lora_a_out for this specific rank
            lora_a_out_masked = lora_a_out * mask
            # print(f"lora_a_out shape is {lora_a_out.shape}, mask shape is {mask.shape}")
            # print(f"lora_a_out_masked shape is {lora_a_out_masked.shape}")
            # print(f"gathered weights shape is {gathered_weight.shape}")
            # Use the all-gathered weight for this rank segment
            lora_out = F.linear(lora_a_out_masked, gathered_weight, bias=None)

            # Scale and accumulate the output for this rank
            lora_out_total = lora_out_total + (self.alpha[i] / self.rank[i]) * lora_out

            # Move to the next rank segment
            start_idx = end_idx

        # Return the final output: base + accumulated LoRA output
        return out + lora_out_total