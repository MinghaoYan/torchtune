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

import cutlass

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

        # Prepare inputs for grouped GEMM
        plan_a = cutlass.op.GroupedGemm(element=x.dtype, layout=cutlass.LayoutType.RowMajor)
        plan_b = cutlass.op.GroupedGemm(element=x.dtype, layout=cutlass.LayoutType.RowMajor)

        As_a, Bs_a = [], []

        # Process inputs for grouped GEMM (LoRA A projection)
        for i in range(len(self.rank)):
            input_i = x[i * bsz : (i + 1) * bsz, ...]  # Split input for each adapter
            input_i = self.dropout(input_i)

            As_a.append(input_i)  # Input matrices for GEMM
            Bs_a.append(self.lora_a[i].weight.T)  # Transpose for GEMM

        As_a = [a.view(-1, a.size(-1)) for a in As_a]  # Flatten batch for GEMM

        As_a = [a.to_local().contiguous() for a in As_a]
        Bs_a = [b.to_local().contiguous() for b in Bs_a]

        # Run grouped GEMM for LoRA A
        Cs_a = [torch.zeros(a.size(0), w.size(1), device=a.device, dtype=a.dtype) for a, w in zip(As_a, Bs_a)]
        Ds_a = [torch.empty_like(c) for c in Cs_a]

        for i, (a, b, c, d) in enumerate(zip(As_a, Bs_a, Cs_a, Ds_a)):
            print(f"Col Lora A Adapter {i}: A shape: {a.shape}, B shape: {b.shape}, C shape: {c.shape}, D shape: {d.shape}")
            print(f"Col Lora A Adapter {i}: A dtype: {a.dtype}, B dtype: {b.dtype}, C dtype: {c.dtype}, D dtype: {d.dtype}")

        plan_a.run(As_a, Bs_a, Cs_a, Ds_a, print_module=False)

        # Convert Ds_a back to DTensor if needed for distribution
        # After plan_a.run and creating Ds_a_dtensor
        Ds_a_dtensor = [
            DTensor.from_local(d, device_mesh=self.device_mesh, placements=[Shard(1)])
            for d in Ds_a
        ]

        # Now perform all_gather by redistributing to Replicate placement.
        Ds_a_gathered = [
            d.redistribute(device_mesh=self.device_mesh, placements=[Replicate()])
            for d in Ds_a_dtensor
        ]

        # Ds_a_gathered now have the full LoRA rank dimension combined from all ranks.

        # LoRA B projection:
        # LoRA B weight: (d/N, r) or (r, d/N) depending on how you partitioned d. 
        # Typically LoRA B is (out_features, r), column partitioned along out_features => (d/N, r)
        # For GEMM: (B, r) * (r, d/N) = (B, d/N)

        As_b, Bs_b = [], []
        for i, lora_a_out_i_dtensor in enumerate(Ds_a_gathered):
            # Convert to local for GEMM
            lora_a_out_i_local = lora_a_out_i_dtensor.to_local().contiguous()  # (B*seq_len, r)

            # LoRA B weight is (d/N, r) or (r, d/N). Typically original is (d, r).
            # After column-part: (d/N, r).
            # For GEMM: we want (r, d/N) to get (B, d/N).
            lora_b_weight_local = self.lora_b[i].weight.to_local().contiguous()
            # If originally (d/N, r), transpose to (r, d/N)
            lora_b_weight_local = lora_b_weight_local.T

            As_b.append(lora_a_out_i_local)     # (M, K) = (B*seq_len, r)
            Bs_b.append(lora_b_weight_local)    # (K, N) = (r, d/N)


        # Run grouped GEMM for LoRA B
        # Cs_b = [torch.zeros(a.size(0), w.size(1), device=a.device, dtype=a.dtype) for a, w in zip(As_b, Bs_b)]
        Cs_b = [
            base_out_i.view(-1, Bs_b[i].size(1))
            for i, base_out_i in enumerate(out[i * bsz : (i + 1) * bsz, ...].to_local() for i in range(len(self.rank)))
        ]

        Ds_b = [torch.empty_like(c) for c in Cs_b]

        for i, (a, b, c, d) in enumerate(zip(As_b, Bs_b, Cs_b, Ds_b)):
            print(f"Col Lora B Adapter {i}: A shape: {a.shape}, B shape: {b.shape}, C shape: {c.shape}, D shape: {d.shape}")
            print(f"Col Lora B Adapter {i}: A dtype: {a.dtype}, B dtype: {b.dtype}, C dtype: {c.dtype}, D dtype: {d.dtype}")

        plan_b.run(As_b, Bs_b, Cs_b, Ds_b, print_module=False)

                # Convert Ds_a back to DTensor if needed for distribution
        # After plan_a.run and creating Ds_a_dtensor
        Ds_b_dtensor = [
            DTensor.from_local(d, device_mesh=self.device_mesh, placements=[Shard(1)])
            for d in Ds_b
        ]

        # Now perform all_gather by redistributing to Replicate placement.
        Ds_b_gathered = [
            d.redistribute(device_mesh=self.device_mesh, placements=[Replicate()])
            for d in Ds_b_dtensor
        ]

        # After plan_b.run
        seq_len = x.size(1)  # Sequence length (from input x)
        total_dim = out.size(-1)  # Full output dimension (d = 4096)

        # Reshape Ds_b into (bsz, seq_len, d/N), then gather to match full output dimension
        Ds_b_reshaped = [
            DTensor.from_local(
                d.view(bsz, seq_len, -1),  # Reshape to (bsz, seq_len, d/N)
                device_mesh=self.device_mesh,
                placements=[Shard(-1)]
            ) for d in Ds_b_gathered
            # .redistribute(placements=[Replicate()]) 
        ]
        # Combine results and finalize outputs
        # for i, lora_b_out_i in enumerate(Ds_b_reshaped):
        #     # Scale LoRA output
        #     scaled_lora_out_i = (self.alpha[i] / self.rank[i]) * lora_b_out_i

        #     # Combine with base model output
        #     base_out_i = out[i * bsz : (i + 1) * bsz, ...]
        #     print(f"Adapter {i}: Base shape: {base_out_i.shape}, type is {type(base_out_i)} scaled shape: {scaled_lora_out_i.shape}, type is {type(scaled_lora_out_i)}")
        #     # if base_out_i.placements != scaled_lora_out_i.placements:
        #     print(f"Base placement is {base_out_i.placements}, scaled placement is {scaled_lora_out_i.placements}")
        #     print(f"Adapter {i}: Base local shape: {base_out_i.to_local().shape}, Scaled local shape: {scaled_lora_out_i.to_local().shape}")
        #     lora_outs.append(base_out_i + scaled_lora_out_i)

        # Final scaled outputs (scale the result after GEMM)
        lora_outs = []
        for i, lora_b_out_i in enumerate(Ds_b_reshaped):
            # Scale the entire GEMM result
            scaled_lora_out_i = (self.alpha[i] / self.rank[i]) * lora_b_out_i
            print(f"Adapter {i}: Scaled shape: {scaled_lora_out_i.shape}, type is {type(scaled_lora_out_i)}")
            lora_outs.append(scaled_lora_out_i)


        # Concatenate outputs along the batch dimension
        concatenated_lora = torch.cat(lora_outs, dim=0)
        return concatenated_lora


    

# def add_lora_sgmv_cutlass(
#     y: torch.Tensor,
#     x: torch.Tensor,
#     wa_ptr: torch.Tensor,
#     wb_ptr: torch.Tensor,
#     s: torch.Tensor,
#     lora_rank: int,
# ):
#     """
#   Semantics:
#     y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

#   Args:
#     y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
#     x: Shape: `[B, H1]`. Input vectors.
#     wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
#       Weight matrix shape: `[num_layers, H1, R]`.
#     wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
#       Weight matrix shape: `[num_layers, R, H2]`.
#     s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
#       `s[0] == 0`, `s[-1] == B`.
#     layer_idx: Layer index of the weight matrices.
#   """
#     tmp_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
#     tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
#     v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
#     _kernels.sgmv_cutlass(v, x, wa_ptr, s, tmp)
#     _kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp)


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
        # Base model computation with row-partitioned weights (W1 equivalent, row partitioned)
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, None)

        if self.disabled:
            return out, []

        bsz = x.shape[0] // len(self.rank)
        if bsz == 0:
            raise ValueError(f"Batch size per adapter is zero. x.shape[0]: {x.shape[0]}, len(self.rank): {len(self.rank)}")

        plan_a = cutlass.op.GroupedGemm(element=x.dtype, layout=cutlass.LayoutType.RowMajor)
        plan_b = cutlass.op.GroupedGemm(element=x.dtype, layout=cutlass.LayoutType.RowMajor)

        # LoRA A projection (row-partitioned)
        As_a, Bs_a = [], []
        for i in range(len(self.rank)):
            input_i = x[i * bsz : (i + 1) * bsz, ...]
            input_i = self.dropout(input_i)
            # Flatten (B, seq_len, h/N) -> (B*seq_len, h/N)
            a_local = input_i.to_local().contiguous().view(-1, input_i.size(-1))

            # LoRA A weight is row-partitioned (e.g., (r, h/N)), transpose if needed
            lora_a_w_local = self.lora_a[i].weight.to_local().contiguous().T
            As_a.append(a_local)   # (M, K) = (B*seq_len, h/N)
            Bs_a.append(lora_a_w_local)  # (K, r) = (h/N, r)

        Cs_a = [torch.zeros(a.size(0), b.size(1), device=a.device, dtype=a.dtype) for a, b in zip(As_a, Bs_a)]
        Ds_a = [torch.empty_like(c) for c in Cs_a]

        for i, (a, b, c, d) in enumerate(zip(As_a, Bs_a, Cs_a, Ds_a)):
            print(f"Row Lora A Adapter {i}: A shape: {a.shape}, B shape: {b.shape}, C shape: {c.shape}, D shape: {d.shape}")
            print(f"Row Lora A Adapter {i}: A dtype: {a.dtype}, B dtype: {b.dtype}, C dtype: {c.dtype}, D dtype: {d.dtype}")

        plan_a.run(As_a, Bs_a, Cs_a, Ds_a, print_module=False)

        # Convert Ds_a into a DTensor with Partial placement to represent partial sums from row partitioning
        # Assume partial sums along the "r" dimension (which is now Ds_a's second dimension)
        Ds_a_dtensor = [
            DTensor.from_local(d, device_mesh=self.device_mesh, placements=[Partial()]) 
            for d in Ds_a
        ]

        # All-reduce to combine partial results along the r dimension
        Ds_a_reduced = [
            d.redistribute(device_mesh=self.device_mesh, placements=[Replicate()])
            for d in Ds_a_dtensor
        ]

        # LoRA B projection (column-partitioned)
        # After LoRA A, we have a fully reduced (summed) output (B*seq_len, r)
        # LoRA B weight is column-partitioned (d/N, r), transpose to get (r, d/N)
        As_b, Bs_b = [], []
        for i, lora_a_out_i_dtensor in enumerate(Ds_a_reduced):
            lora_a_out_i_local = lora_a_out_i_dtensor.to_local().contiguous() # (B*seq_len, r)
            lora_b_w_local = self.lora_b[i].weight.to_local().contiguous().T   # (r, d/N)
            As_b.append(lora_a_out_i_local)
            Bs_b.append(lora_b_w_local)

        # C for LoRA B is initialized with base_out_i (already row-partitioned)
        # Convert base_out_i to local, reshape to (M, d/N)
        Cs_b = [
            out[i * bsz : (i + 1) * bsz, ...].to_local().view(-1, Bs_b[i].size(1))
            for i in range(len(self.rank))
        ]
        Ds_b = [torch.empty_like(c) for c in Cs_b]

        for i, (a, b, c, d) in enumerate(zip(As_a, Bs_a, Cs_a, Ds_a)):
            print(f"Row Lora B Adapter {i}: A shape: {a.shape}, B shape: {b.shape}, C shape: {c.shape}, D shape: {d.shape}")
            print(f"Row Lora B Adapter {i}: A dtype: {a.dtype}, B dtype: {b.dtype}, C dtype: {c.dtype}, D dtype: {d.dtype}")

        plan_b.run(As_b, Bs_b, Cs_b, Ds_b, print_module=False)

        # After LoRA B, we have partial sums along the "d/N" dimension (since column-partitioned)
        Ds_b_dtensor = [
            DTensor.from_local(d, device_mesh=self.device_mesh, placements=[Partial()]) 
            for d in Ds_b
        ]

        # All-reduce again to get a fully combined result across the column partition
        Ds_b_reduced = [
            d.redistribute(device_mesh=self.device_mesh, placements=[Replicate()])
            for d in Ds_b_dtensor
        ]

        seq_len = x.size(1)
        # Reshape to (bsz, seq_len, d) after all-reduce since now each rank has full dimension
        Ds_b_reshaped = [
            DTensor.from_local(
                d.to_local().view(bsz, seq_len, -1),
                device_mesh=self.device_mesh,
                placements=[Replicate()]  # Fully replicated after reduce
            ) for d in Ds_b_reduced
        ]

        # Scale the final output after both LoRA projections and reductions
        lora_outs = []
        for i, lora_b_out_i in enumerate(Ds_b_reshaped):
            scaled_lora_out_i = (self.alpha[i] / self.rank[i]) * lora_b_out_i
            lora_outs.append(scaled_lora_out_i)

        # Concatenate along the batch dimension
        concatenated_lora = torch.cat(lora_outs, dim=0)
        return concatenated_lora

