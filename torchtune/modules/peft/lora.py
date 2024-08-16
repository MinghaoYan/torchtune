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
        # self.lora_a = nn.Linear(in_features=in_dim, out_features=sum(self.rank), bias=False)
        # self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        for idx in range(len(self.rank)):
            setattr(self, f'lora_a_0_{idx}', nn.Linear(in_features=in_dim, out_features=self.rank[idx], bias=False))
            setattr(self, f'lora_b_0_{idx}', nn.Linear(in_features=self.rank[idx], out_features=out_dim, bias=False))
            setattr(self, f'lora_a_1_{idx}', nn.Linear(in_features=in_dim, out_features=self.rank[idx], bias=False))
            setattr(self, f'lora_b_1_{idx}', nn.Linear(in_features=self.rank[idx], out_features=out_dim, bias=False))

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
            _lora_a_init_params(getattr(self, f"lora_a_0_{idx}"))
            _lora_b_init_params(getattr(self, f"lora_b_0_{idx}"))
            _lora_a_init_params(getattr(self, f"lora_a_1_{idx}"))
            _lora_b_init_params(getattr(self, f"lora_b_1_{idx}"))


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
            adapter_params.append(f"lora_a_0_{idx}.weight")
            adapter_params.append(f"lora_b_0_{idx}.weight")
            adapter_params.append(f"lora_a_1_{idx}.weight")
            adapter_params.append(f"lora_b_1_{idx}.weight")
        return adapter_params

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

        # print(f"Lora input dim is {x.shape}")
        lora_out = []
        num_adapters = len(self.rank)
        # print(x.shape[0], self.bsz, x.shape[0] == self.bsz)
        if x.dim() == 3 and num_adapters > 1:
            x = x.repeat(num_adapters, 1, 1)
            x = x.view(num_adapters, -1, x.shape[1], x.shape[2])
        # print(f"Lora input dim after repeat is {x.shape}")
        after_dropout = self.dropout(x)
        bsz = x.shape[0] // num_adapters

        # Disable gradients for inactive LoRA parameters
        for idx in range(num_adapters):
            lora_a_active = getattr(self, f'lora_a_{activated}_{idx}')
            lora_b_active = getattr(self, f'lora_b_{activated}_{idx}')
            for param in lora_a_active.parameters():
                param.requires_grad = True
            for param in lora_b_active.parameters():
                param.requires_grad = True

        for idx in range(num_adapters):
            # print(f"after dropout shape is {after_dropout.shape}")
            lora_a_slice = after_dropout[idx, :, :, :]
            # print(f"lora a slice shape is {lora_a_slice.shape}")
            
            lora_after_a = getattr(self, f'lora_a_{activated}_{idx}')(lora_a_slice)
            lora_after_b = getattr(self, f'lora_b_{activated}_{idx}')(lora_after_a)

            scaled_results = (self.alpha[idx] / self.rank[idx]) * lora_after_b

            if out.dim() == 4:
                final_results = scaled_results + out[idx, :, :, :]
            elif out.dim() == 3:
                final_results = scaled_results + out

            # print(f"final_results shape is {final_results.shape}")

            lora_out.append(final_results)
        
        # Disable gradients for inactive LoRA parameters
        for idx in range(num_adapters):
            lora_a_inactive = getattr(self, f'lora_a_{1-activated}_{idx}')
            lora_b_inactive = getattr(self, f'lora_b_{1-activated}_{idx}')
            for param in lora_a_inactive.parameters():
                param.requires_grad = False
            for param in lora_b_inactive.parameters():
                param.requires_grad = False
        
        if lora_out[0].dim() == 3:
            total_out = torch.stack(lora_out, dim=0)
        elif lora_out[0].dim() == 4:
            total_out = torch.cat(lora_out, dim=0)
        # print(f"lora out len is {len(lora_out)} total out shape is {total_out.shape}")
        return total_out