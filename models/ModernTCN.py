import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ModernTCN_Layer import RevIN, Block


class ModernTCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 保存配置
        self.configs = configs
        self.seq_len = configs.seq_len
        self.nvars = configs.enc_in
        self.revin = getattr(configs, 'revin', True)

        # 模型参数（带默认值）
        self.patch_size = getattr(configs, 'patch_size', 32)
        self.patch_stride = getattr(configs, 'patch_stride', 16)
        self.embed_dim = getattr(configs, 'embed_dim', 64)
        self.num_blocks = getattr(configs, 'num_blocks', 3)
        self.kernel_size = getattr(configs, 'kernel_size', 65)
        self.ffn_ratio = getattr(configs, 'ffn_ratio', 4)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 确保embed_dim是nvars的倍数（修复分组卷积问题）
        if self.embed_dim % self.nvars != 0:
            self.embed_dim = ((self.embed_dim // self.nvars) + 1) * self.nvars
            print(f"Adjust embed_dim to {self.embed_dim} to be divisible by nvars {self.nvars}")

        # 计算patch数量
        self.num_patches = (self.seq_len - self.patch_size) // self.patch_stride + 1

        if self.revin:
            self.revin_layer = RevIN(self.nvars, affine=True)

        # Patch embedding
        self.patch_embed = nn.Conv1d(
            self.nvars, self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride
        )

        # 计算卷积后序列长度
        self.conv_output_length = self._conv_output_length(self.seq_len)

        # TCN主干网络
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(Block(
                self.embed_dim, self.embed_dim, self.kernel_size,
                groups=self.nvars, ffn_ratio=self.ffn_ratio,
                dropout=self.dropout
            ))

        # 重建层 - 使用转置卷积确保输出长度匹配输入
        self.reconstruction = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim * self.ffn_ratio, 1),
            nn.GELU(),
            nn.ConvTranspose1d(
                self.embed_dim * self.ffn_ratio, self.nvars,
                kernel_size=self.patch_size,
                stride=self.patch_stride,
                output_padding=(self.seq_len - self._conv_transpose_output_length(self.conv_output_length))
            ))

    def _conv_output_length(self, input_length):
        """计算卷积后的序列长度"""
        return (input_length - self.patch_size) // self.patch_stride + 1

    def _conv_transpose_output_length(self, input_length):
        """计算转置卷积后的序列长度"""
        return (input_length - 1) * self.patch_stride + self.patch_size

    def forward(self, x):
        """
        输入形状: (batch_size, seq_len, nvars)
        输出形状: (batch_size, seq_len, nvars)
        """
        # 保存原始输入
        B, L, C = x.shape

        # 应用RevIN归一化
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # 调整维度: (B, C, L)
        x = x.permute(0, 2, 1)

        # 分块嵌入
        x = self.patch_embed(x)

        # 通过TCN块
        for block in self.blocks:
            x = block(x)

        # 重建序列
        reconstructions = self.reconstruction(x)

        # 确保输出长度匹配
        if reconstructions.size(2) != self.seq_len:
            reconstructions = F.interpolate(
                reconstructions,
                size=self.seq_len,
                mode='linear',
                align_corners=True
            )

        # 调整维度: (B, L, C)
        reconstructions = reconstructions.permute(0, 2, 1)

        # 反归一化
        if self.revin:
            reconstructions = self.revin_layer(reconstructions, 'denorm')

        return reconstructions


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = ModernTCN(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 只使用x_enc作为输入
        return self.model(x_enc)