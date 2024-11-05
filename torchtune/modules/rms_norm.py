# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn, Tensor
from torch.distributed.tensor import DTensor, Shard
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate

# class RMSNorm(nn.Module):
#     """
#     Implements Root Mean Square Normalization introduced in
#     https://arxiv.org/pdf/1910.07467.pdf.

#     Reference implementation (used for correctness verfication)
#     can be found here:
#     https://github.com/facebookresearch/llama/blob/main/llama/model.py

#     Args:
#         dim (int): embedding size
#         eps (float): small value to avoid division by zero. Default: 1e-6
#     """

#     def __init__(self, dim: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.eps = eps
#         self.scale = nn.Parameter(torch.ones(dim))

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x (Tensor): input tensor to normalize

#         Returns:
#             Tensor: The output tensor after applying RMSNorm.
#         """
#         # computation is in fp32
#         x_fp32 = x.float()
#         x_normed = (
#             x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
#         ).type_as(x)
#         print(f"x_norm local shape is {x_normed.to_local().shape}, placement is {x_normed.placements}, type is {type(x_normed)}")
#         print(f"scale locals shape is {self.scale.to_local().shape}, placement is {self.scale.placements}, type is {type(self.scale)}")
#         return x_normed * self.scale


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        output = DTensor.redistribute(output.device_mesh, placements=[Shard(1)])
        print(f"x_norm local shape is {output.to_local().shape}, placement is {output.placements}, type is {type(output)}")
        print(f"scale locals shape is {self.scale.to_local().shape}, placement is {self.scale.placements}, type is {type(self.scale)}, scale dimension is {self.scale.numel()}")
        return output * self.scale