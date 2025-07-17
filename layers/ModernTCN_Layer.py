import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, groups=groups
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, ffn_ratio=4, dropout=0.1):
        super().__init__()
        # Depthwise convolution
        self.dw_conv = ReparamLargeKernelConv(
            in_channels, in_channels, kernel_size, groups=groups
        )

        # Pointwise convolutions for FFN
        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * ffn_ratio, 1, groups=groups),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels * ffn_ratio, out_channels, 1, groups=groups),
            nn.Dropout(dropout)
        )

        self.shortcut = nn.Identity() if in_channels == out_channels else \
            nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.dw_conv(x)
        x = self.ffn(x)
        return identity + x
