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
from torchtune.modules.peft.peft_utils import AdapterModule
from torchtune import utils
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import DTensor, Shard, DeviceMesh, distribute_tensor, Replicate

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
        lora_out = []
        num_adapters = len(self.rank)
        if x.shape[0] == self.bsz and num_adapters > 1:
            x = x.repeat(num_adapters, 1, 1)
        # print(f"Lora input dim after repeat is {x.shape}")
        after_dropout = self.dropout(x)
        bsz = x.shape[0] // num_adapters
        # print(f"enter adapter loop")
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
        print(total_out.shape)
        
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

def _lora_a_list_init_params(x: nn.ModuleList) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    for lora in x:
        nn.init.kaiming_uniform_(lora.weight, a=math.sqrt(5))


def _lora_b_list_init_params(x: nn.ModuleList) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    for lora in x:
        nn.init.zeros_(lora.weight)



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
        # self.lora_b = nn.Linear(in_features=sum(self.rank), out_features=out_dim, bias=False)

        # Initialize ModuleLists for lora_a and lora_b
        self.lora_a = nn.ModuleList()
        self.lora_b = nn.ModuleList()
        for r in self.rank:
            self.lora_a.append(nn.Linear(in_features=self.in_dim, out_features=r, bias=False))
            self.lora_b.append(nn.Linear(in_features=r, out_features=self.out_dim, bias=False))


        self.world_size, self.device_rank = utils.get_world_size_and_rank()

        assert len(self.rank) % self.world_size == 0, "Must evenly divide num lora adapters and world size"
        # Create a rank-specific mask for the weight matrix
        self.start_row = sum(self.rank[:self.device_rank])
        self.end_row = sum(self.rank[:self.device_rank]) + self.rank[self.device_rank]


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
        _lora_a_list_init_params(self.lora_a)
        _lora_b_list_init_params(self.lora_b)


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
        # adapter_params = ["lora_a.weight", "lora_b.weight"]

        adapter_params = []

        for i, (lora_a, lora_b) in enumerate(zip(self.lora_a, self.lora_b)):
            # Access the weights of each LoRA adapter in the ModuleList
            adapter_params.append(f"lora_a.{i}.weight")
            adapter_params.append(f"lora_b.{i}.weight")

            # If the LoRA layers have bias (if added in the future), include them as well
            if lora_a.bias is not None:
                adapter_params.append(f"lora_a.{i}.bias")
            if lora_b.bias is not None:
                adapter_params.append(f"lora_b.{i}.bias")

        return adapter_params

    def forward(self, x: Tensor, activated: int = 0):
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, None)

        if self.disabled:
            return out, []

        bsz = x.shape[0] // len(self.rank)
        if bsz == 0:
            raise ValueError(f"Batch size per adapter is zero. x.shape[0]: {x.shape[0]}, len(self.rank): {len(self.rank)}")

        lora_outs = []

        # Iterate over each LoRA adapter
        for i in range(len(self.rank)):
            print(f"current adapter index is {i}")
            input_i = x[i * bsz : (i + 1) * bsz, ...]

            input_i = self.dropout(input_i)
            print(f"input_i is DTensor: {isinstance(input_i, DTensor)}, input_i shape is {input_i.shape}")

            lora_a_out_i = self.lora_a[i](input_i)
            print(f"LoRA A output shape: {lora_a_out_i.shape}, device: {lora_a_out_i.device}")

            lora_out_i = self.lora_b[i](lora_a_out_i)
            print(f"LoRA B output shape: {lora_out_i.shape}, device: {lora_out_i.device}")

            scaled_lora_out_i = (self.alpha[i] / self.rank[i]) * lora_out_i

            base_out_i = out[i * bsz : (i + 1) * bsz, ...]
            base_out_i = base_out_i.to(scaled_lora_out_i.device)

            lora_outs.append(base_out_i + scaled_lora_out_i)

        concatenated_lora = torch.cat(lora_outs, dim=0)
        print(f"lora outs shape is {concatenated_lora.shape}")
        return concatenated_lora


class LoRALinearColCol(nn.Module, AdapterModule):
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
        device_mesh: DeviceMesh,
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
        self.device_mesh = device_mesh
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
        # self.lora_b = nn.Linear(in_features=sum(self.rank), out_features=out_dim, bias=False)

        # Initialize ModuleLists for lora_a and lora_b
        self.lora_a = nn.ModuleList()
        self.lora_b = nn.ModuleList()
        for r in self.rank:
            # Standard initialization without DTensor.from_local
            local_lora_a = nn.Linear(self.in_dim, r, bias=False)
            self.lora_a.append(local_lora_a)
            if not local_lora_a.weight.is_leaf:
                print(f"lora_a weight is leaf: {local_lora_a.weight.is_leaf}")

            local_lora_b = nn.Linear(r, self.out_dim, bias=False)
            self.lora_b.append(local_lora_b)
            if not local_lora_b.weight.is_leaf:
                print(f"lora_b weight is leaf: {local_lora_b.weight.is_leaf}")

        self.world_size, self.device_rank = utils.get_world_size_and_rank()

        assert len(self.rank) % self.world_size == 0, "Must evenly divide num lora adapters and world size"
        # Create a rank-specific mask for the weight matrix
        self.start_row = sum(self.rank[:self.device_rank])
        self.end_row = sum(self.rank[:self.device_rank]) + self.rank[self.device_rank]


        self.merged = False
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_list_init_params(self.lora_a)
        _lora_b_list_init_params(self.lora_b)


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
        # adapter_params = ["lora_a.weight", "lora_b.weight"]

        adapter_params = []

        for i, (lora_a, lora_b) in enumerate(zip(self.lora_a, self.lora_b)):
            # Access the weights of each LoRA adapter in the ModuleList
            adapter_params.append(f"lora_a.{i}.weight")
            adapter_params.append(f"lora_b.{i}.weight")

            # If the LoRA layers have bias (if added in the future), include them as well
            if lora_a.bias is not None:
                adapter_params.append(f"lora_a.{i}.bias")
            if lora_b.bias is not None:
                adapter_params.append(f"lora_b.{i}.bias")

        return adapter_params

    def forward(self, x: Tensor, activated: int = 0):
        # Base model computation with column-partitioned base weight (W1 equivalent)
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, None)

        if self.disabled:
            return out, []

        bsz = x.shape[0] // len(self.rank)
        if bsz == 0:
            raise ValueError(f"Batch size per adapter is zero. x.shape[0]: {x.shape[0]}, len(self.rank): {len(self.rank)}")

        lora_outs = []

        # Process each LoRA adapter for W1 equivalent
        for i in range(len(self.rank)):
            input_i = x[i * bsz : (i + 1) * bsz, ...]

            # LoRA A1 (column-partitioned), apply dropout and projection
            input_i = self.dropout(input_i)
            # print(f"input shape is {input_i.shape}, lora_a[{i}] weight shape is {self.lora_a[i].weight.shape}")
            # print("get input")
            lora_a_out_i = self.lora_a[i](input_i)
            # print(f"finish lora a with shape {lora_a_out_i.shape}, type is {type(lora_a_out_i)}")


            # Directly gather the DTensor without `to_local()`
            lora_a_out_i_dtensor = lora_a_out_i.redistribute(
                device_mesh=self.device_mesh,
                placements=[Replicate()]  # Adjust this if your shard is on a different dimension
            )
            # print(f"finish lora a gather with shape {lora_a_out_i.shape}")

            # print(f"lora a out shape is {lora_a_out_i_dtensor.shape}")
            lora_b_out_i = self.lora_b[i](lora_a_out_i_dtensor)
            # print(f"finish lora b out shape is {lora_b_out_i.shape}")

            # Scale LoRA output
            scaled_lora_out_i = (self.alpha[i] / self.rank[i]) * lora_b_out_i

            # Combine with base model output
            # print(f"out shape is {out.shape}, type is {type(out)}")
            base_out_i = out[i * bsz : (i + 1) * bsz, ...]

            lora_outs.append(base_out_i + scaled_lora_out_i)

        concatenated_lora = torch.cat(lora_outs, dim=0)
        return concatenated_lora

        input_i = self.dropout(input_i)
        lora_a_out_i = self.lora_a[i](input_i)

    

def add_lora_sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, R]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    tmp_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_cutlass(v, x, wa_ptr, s, tmp)
    _kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp)


class LoRALinearRowCol(nn.Module, AdapterModule):
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
        device_mesh: DeviceMesh,
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
        self.device_mesh = device_mesh
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
        # self.lora_b = nn.Linear(in_features=sum(self.rank), out_features=out_dim, bias=False)

        # Initialize ModuleLists for lora_a and lora_b
        self.lora_a = nn.ModuleList()
        self.lora_b = nn.ModuleList()
        for r in self.rank:
            # Standard initialization without DTensor.from_local
            local_lora_a = nn.Linear(self.in_dim, r, bias=False)
            self.lora_a.append(local_lora_a)

            local_lora_b = nn.Linear(r, self.out_dim, bias=False)
            self.lora_b.append(local_lora_b)


        self.world_size, self.device_rank = utils.get_world_size_and_rank()

        assert len(self.rank) % self.world_size == 0, "Must evenly divide num lora adapters and world size"
        # Create a rank-specific mask for the weight matrix
        self.start_row = sum(self.rank[:self.device_rank])
        self.end_row = sum(self.rank[:self.device_rank]) + self.rank[self.device_rank]


        self.merged = False
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_list_init_params(self.lora_a)
        _lora_b_list_init_params(self.lora_b)


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
        # adapter_params = ["lora_a.weight", "lora_b.weight"]

        adapter_params = []

        for i, (lora_a, lora_b) in enumerate(zip(self.lora_a, self.lora_b)):
            # Access the weights of each LoRA adapter in the ModuleList
            adapter_params.append(f"lora_a.{i}.weight")
            adapter_params.append(f"lora_b.{i}.weight")

            # If the LoRA layers have bias (if added in the future), include them as well
            if lora_a.bias is not None:
                adapter_params.append(f"lora_a.{i}.bias")
            if lora_b.bias is not None:
                adapter_params.append(f"lora_b.{i}.bias")

        return adapter_params

    def forward(self, x: Tensor, activated: int = 0):
        # Base model computation with row-partitioned base weight (W2 equivalent)
        # print(f"input dimension is {x.shape}, type is {type(x)}, weight dimension is {self.weight.shape}, type is {type(self.weight)}")
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, None)

        if self.disabled:
            return out, []

        bsz = x.shape[0] // len(self.rank)
        if bsz == 0:
            raise ValueError(f"Batch size per adapter is zero. x.shape[0]: {x.shape[0]}, len(self.rank): {len(self.rank)}")

        lora_outs = []

        for i in range(len(self.rank)):
            input_i = x[i * bsz : (i + 1) * bsz, ...]

            # LoRA A2 (row-partitioned), apply dropout and projection
            input_i = self.dropout(input_i)
            # print(f"input dimension is {input_i.shape}, local dim is {input_i.to_local().shape}, placement is {input_i.placements}")
            lora_a_out_i = self.lora_a[i](input_i)
            # print(f"lora_a_out_i dimension is {lora_a_out_i.shape}, local dim is {lora_a_out_i.to_local().shape}, placement is {lora_a_out_i.placements}")
            # All-reduce after A2 to accumulate across ranks
            # dist.all_reduce(lora_a_out_i, op=dist.ReduceOp.SUM)
            lora_a_out_i_dtensor = lora_a_out_i.redistribute(
                device_mesh=self.device_mesh,
                placements=[Replicate()]  # Adjust this if your shard is on a different dimension
            )
            # print(f"lora_a_out_i_dtensor dimension is {lora_a_out_i_dtensor.shape}, local dim is {lora_a_out_i_dtensor.to_local().shape}, placement is {lora_a_out_i_dtensor.placements}")

            # LoRA B2 (column-partitioned) for output projection
            lora_b_out_i = self.lora_b[i](lora_a_out_i_dtensor)
            # print(f"lora_a_out_i_dtensor dimension is {lora_a_out_i_dtensor.shape}, local dim is {lora_a_out_i_dtensor.to_local().shape}, placement is {lora_a_out_i_dtensor.placements}")
            # Scale LoRA output
            scaled_lora_out_i = (self.alpha[i] / self.rank[i]) * lora_b_out_i

            # Combine with base model output
            base_out_i = out[i * bsz : (i + 1) * bsz, ...]
            # base_out_i = base_out_i.to(scaled_lora_out_i.device)
            lora_outs.append(base_out_i + scaled_lora_out_i)

        # Define a function to handle each loop iteration
        # def lora_iteration(i, x, bsz, self):
        #     input_i = x[i * bsz : (i + 1) * bsz, ...]
            
        #     # Apply dropout and LoRA A2 projection
        #     input_i = self.dropout(input_i)
        #     lora_a_out_i = self.lora_a[i](input_i)
            
        #     # Redistribute (all-reduce)
        #     lora_a_out_i_dtensor = lora_a_out_i.redistribute(
        #         device_mesh=self.device_mesh,
        #         placements=[Replicate()]
        #     )
            
        #     # LoRA B2 projection
        #     lora_b_out_i = self.lora_b[i](lora_a_out_i_dtensor)
            
        #     # Scale output
        #     scaled_lora_out_i = (self.alpha[i] / self.rank[i]) * lora_b_out_i
            
        #     # Combine with base model output
        #     base_out_i = self.out[i * bsz : (i + 1) * bsz, ...]
        #     base_out_i = base_out_i.to(scaled_lora_out_i.device)
            
        #     return base_out_i + scaled_lora_out_i

        # # Main loop using torch.jit.fork
        # lora_outs = []
        # futures = []

        # for i in range(len(self.rank)):
        #     future = torch.jit.fork(lora_iteration, i, x, bsz, self)
        #     futures.append(future)

        # # Wait for all tasks to complete and collect results
        # lora_outs = [torch.jit.wait(fut) for fut in futures]

        concatenated_lora = torch.cat(lora_outs, dim=0)
        return concatenated_lora