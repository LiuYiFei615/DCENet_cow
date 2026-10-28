# -*- coding: utf-8 -*-
"""Standalone STDFM reference implementation from the DCENet research files.
This is a component release, not a claim of matching final full-model checkpoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class STDFM(nn.Module):
    """Standalone topology-inspired deformable fusion module."""

    def __init__(self, input_channels: int):
        super().__init__()

        if input_channels < 2:
            raise ValueError("input_channels 必须大于或等于 2")

        c = input_channels
        hidden = c // 2
        if (8 * c) % 4 or (4 * c) % 4 or hidden % 4:
            raise ValueError("采用 groups=4 时，input_channels /

        # V = φ_v([X_c, X_n])，输入为 8C 通道
        self.phi_v = nn.Sequential(
            nn.Conv2d(8 * c, hidden, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # E = φ_e(X_n - X_c)，输入为 4C 通道
        self.phi_e = nn.Sequential(
            nn.Conv2d(4 * c, hidden, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # T = φ_t(V ⊙ E)
        self.phi_t = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

               self.offset_conv = nn.Conv2d(
            c, 18, kernel_size=3, padding=1
        )
        self.deform_conv = DeformConv2d(
            c, c, kernel_size=3, padding=1
        )

        # I_STDFM = ψ([X_def, T])
        self.psi = nn.Sequential(
            nn.Conv2d(c + hidden, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("输入形状应为 [B, C, H, W]")
        if x.shape[1] != self.offset_conv.in_channels:
            raise ValueError("输入通道数与 input_channels 不一致")

        # Topology-inspired branch
        xu, xd, xl, xr = self.four_neighbors(x)
        xn = torch.cat((xu, xd, xl, xr), dim=1)
        xc = torch.cat((x, x, x, x), dim=1)

        v = self.phi_v(torch.cat((xc, xn), dim=1))
        e = self.phi_e(xn - xc) 
        t = self.phi_t(v * e)

        # Deformation branch
        offsets = self.offset_conv(x)
        x_def = self.deform_conv(x, offsets)

        # Branch fusion
        return self.psi(torch.cat((x_def, t), dim=1))



if __name__ == "__main__":
    module = STDFM(input_channels=32)
    feature = torch.randn(2, 32, 128, 128)
    output = module(feature)
    print(output.shape)  # torch.Size([2, 32, 128, 128])