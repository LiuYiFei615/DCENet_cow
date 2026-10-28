# -*- coding: utf-8 -*-
"""Standalone FPMM reference implementation from the DCENet research files.
This is a component release, not a claim of matching final full-model checkpoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPMM(nn.Module):
    """
    

    Independent FPMM module implementation.
    External fusion with a fixed coefficient (0.2 in the study) is applied
    by the calling model, not by this module's forward() method.
    """

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        if dim % 4 != 0:
            raise ValueError("FPMM 采用 groups=4 时，dim 必须能被 4 整除")

        #
        self.layer_norm = nn.LayerNorm(dim)

  
        self.filter_real = nn.Conv2d(dim, dim, kernel_size=1, groups=4)
        self.filter_imag = nn.Conv2d(dim, dim, kernel_size=1, groups=4)

        nn.init.xavier_uniform_(self.filter_real.weight)
        nn.init.xavier_uniform_(self.filter_imag.weight)
        nn.init.zeros_(self.filter_real.bias)
        nn.init.zeros_(self.filter_imag.bias)

        self.modpf = MPF(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x

        # [B, C, H, W] → [B, H, W, C]
        z = self.layer_norm(x.permute(0, 2, 3, 1))


        spectrum = torch.fft.fft2(
            z, dim=(1, 2), norm="backward"
        )

     
        refined = torch.complex(
            self.filter_real(spectrum.real.permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
            self.filter_imag(spectrum.imag.permute(0, 3, 1, 2)).permute(0, 2, 3, 1),
        )


        modulated = self.modpf(refined)

     
        spatial = torch.fft.ifft2(
            modulated, dim=(1, 2), norm="backward"
        ).real

        spatial = self.dropout(spatial)

        # [B, H, W, C] → [B, C, H, W]
        spatial = spatial.permute(0, 3, 1, 2).contiguous()

        # Internal residual fusion
        i_fpmm = residual + spatial
        return i_fpmm
# 主网络 __init__ 中
# self.fpmm = FPMM(dim=channels)

# Example: external fusion in the calling network
# x = x + 0.2 * self.fpmm(x)


if __name__ == "__main__":
    module = FPMM(dim=64)
    feature = torch.randn(2, 64, 16, 16)
    i_fpmm = module(feature)
    fused_feature = feature + 0.2 * i_fpmm  # 仅演示原公式（19）调用方式
    print("FPMM output:", tuple(i_fpmm.shape))
    print("Fused output:", tuple(fused_feature.shape))