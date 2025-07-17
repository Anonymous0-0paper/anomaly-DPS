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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.ModernTCN_Layer import RevIN, Block
#
# class ModernTCN_AnomalyDetection(nn.Module):
#     def __init__(self,
#                  input_dim: int,
#                  seq_len: int = 512,
#                  patch_size: int = 32,
#                  patch_stride: int = 16,
#                  embed_dim: int = 64,
#                  num_blocks: int = 3,
#                  kernel_size: int = 65,
#                  ffn_ratio: int = 4,
#                  revin: bool = True,
#                  dropout: float = 0.1):
#         """
#         精简的ModernTCN异常检测模型
#
#         参数:
#         input_dim: 输入特征的维度 (通道数)
#         seq_len: 输入序列长度
#         patch_size: 分块大小
#         patch_stride: 分块步长
#         embed_dim: 嵌入维度
#         num_blocks: TCN块的数量
#         kernel_size: 卷积核大小
#         ffn_ratio: FFN扩展比例
#         revin: 是否使用RevIN归一化
#         dropout: Dropout概率
#         """
#         super().__init__()
#         self.seq_len = seq_len
#         self.patch_size = patch_size
#         self.patch_stride = patch_stride
#         self.revin = revin
#         self.num_patches = (seq_len - patch_size) // patch_stride + 1
#
#         if self.revin:
#             self.revin_layer = RevIN(input_dim)
#
#         # Patch embedding
#         self.patch_embed = nn.Conv1d(
#             input_dim, embed_dim,
#             kernel_size=patch_size,
#             stride=patch_stride
#         )
#
#         # TCN主干网络
#         self.blocks = nn.ModuleList()
#         for _ in range(num_blocks):
#             self.blocks.append(Block(
#                 embed_dim, embed_dim, kernel_size,
#                 groups=input_dim, ffn_ratio=ffn_ratio,
#                 dropout=dropout
#             ))
#
#         # 重建层
#         self.reconstruction = nn.Sequential(
#             nn.Conv1d(embed_dim, embed_dim * ffn_ratio, 1),
#             nn.GELU(),
#             nn.ConvTranspose1d(
#                 embed_dim * ffn_ratio, input_dim,
#                 kernel_size=patch_size,
#                 stride=patch_stride
#             )
#         )
#
#     def forward(self, x):
#         """
#         输入形状: (batch_size, seq_len, input_dim)
#         输出形状: (batch_size, seq_len, input_dim)
#         """
#         # 保存原始输入用于重建损失计算
#         original = x
#
#         # 应用RevIN归一化
#         if self.revin:
#             x = self.revin_layer(x, 'norm')
#
#         # 调整维度: (B, C, L)
#         x = x.permute(0, 2, 1)
#
#         # 分块嵌入
#         x = self.patch_embed(x)
#
#         # 通过TCN块
#         for block in self.blocks:
#             x = block(x)
#
#         # 重建序列
#         reconstructions = self.reconstruction(x)
#
#         # 裁剪到原始长度
#         reconstructions = reconstructions[:, :, :self.seq_len]
#
#         # 调整维度: (B, L, C)
#         reconstructions = reconstructions.permute(0, 2, 1)
#
#         # 反归一化
#         if self.revin:
#             reconstructions = self.revin_layer(reconstructions, 'denorm')
#
#         return reconstructions
#
#     def detect_anomalies(self, x, threshold=0.05):
#         """
#         异常检测接口
#
#         参数:
#         x: 输入序列 (batch_size, seq_len, input_dim)
#         threshold: 异常阈值 (0-1之间)
#
#         返回:
#         anomalies: 异常标志 (batch_size, seq_len)
#         scores: 异常分数 (batch_size, seq_len)
#         """
#         with torch.no_grad():
#             # 获取重建序列
#             reconstructions = self.forward(x)
#
#             # 计算重建误差
#             errors = torch.abs(x - reconstructions)
#
#             # 计算每个时间点的平均误差
#             scores = errors.mean(dim=-1)
#
#             # 基于阈值检测异常
#             max_errors = scores.max(dim=1, keepdim=True).values
#             anomalies = scores > (threshold * max_errors)
#
#             return anomalies, scores
#
# # import torch
# # from torch import nn
# # from layers.ModernTCN_RevIN import RevIN
# #
# #
# # class LayerNorm(nn.Module):
# #     def __init__(self, channels, eps=1e-6, data_format="channels_last"):
# #         super().__init__()
# #         self.norm = nn.LayerNorm(channels)
# #
# #     def forward(self, x):
# #         B, M, D, N = x.shape
# #         x = x.permute(0, 1, 3, 2).reshape(B * M, N, D)
# #         x = self.norm(x)
# #         return x.reshape(B, M, N, D).permute(0, 1, 3, 2)
# #
# #
# # class ModernTCN(nn.Module):
# #     def __init__(self, configs):
# #         super().__init__()
# #         # 固定参数配置
# #         self.configs = configs
# #         self.seq_len = configs.seq_len
# #         self.nvars = configs.enc_in
# #         self.revin = configs.revin if hasattr(configs, 'revin') else True
# #
# #         # 模型维度参数 - 根据数据集动态调整
# #         self.patch_size = 16
# #         self.patch_stride = 8
# #         self.downsample_ratio = 2
# #
# #         # 初始维度 - 后续会根据实际输入调整
# #         self.dims = [256]
# #         self.large_size = [31]
# #         self.small_size = [5]
# #         self.num_blocks = [1]
# #
# #         if self.revin:
# #             self.revin_layer = RevIN(self.nvars, affine=True)
# #
# #         # 下采样层 - 初始为空，将在首次forward时创建
# #         self.downsample_layers = nn.ModuleList()
# #         self.stages = nn.ModuleList()
# #         self.head = None
# #         self.initialized = False
# #
# #     def initialize_model(self, x):
# #         """根据输入数据的维度动态初始化模型参数"""
# #         B, L, M = x.shape
# #         self.nvars = M
# #
# #         # 计算patch数量
# #         patch_num = (L - self.patch_size) // self.patch_stride + 1
# #
# #         # 设置模型维度
# #         self.dims = [min(256, patch_num)]  # 确保维度不超过序列长度
# #         self.large_size = [min(31, patch_num)]
# #         self.small_size = [5]
# #         self.num_blocks = [1]
# #
# #         # 创建下采样层
# #         self.downsample_layers = nn.ModuleList()
# #         self.downsample_layers.append(nn.Linear(self.patch_size, self.dims[0]))
# #
# #         # 创建主干网络
# #         self.stages = nn.ModuleList()
# #         for i in range(len(self.dims)):
# #             self.stages.append( Stage (
# #                 self.num_blocks[i],
# #                 self.large_size[i],
# #                 self.small_size[i],
# #                 self.dims[i],
# #                 self.nvars,
# #                 self.configs.dropout)
# #             )
# #
# #         # 创建输出层
# #         self.head = nn.Sequential(
# #             nn.LayerNorm(self.dims[-1]),
# #             nn.Linear(self.dims[-1], self.patch_size)
# #         )
# #
# #         self.initialized = True
# #
# #     def forward_feature(self, x):
# #         # 输入x: [B, M, L]
# #         B, M, L = x.shape
# #
# #         # 初始patch嵌入
# #         x = x.unfold(-1, self.patch_size, self.patch_stride)  # [B, M, P, S]
# #         x = self.downsample_layers[0](x)  # [B, M, P, D]
# #         x = x.permute(0, 1, 3, 2)  # [B, M, D, P]
# #
# #         # 通过各阶段
# #         for i in range(len(self.stages)):
# #             # 应用当前stage
# #             x = self.stages[i](x)
# #
# #         return x
# #
# #     def forward(self, x):
# #         # 输入x: [B, L, M]
# #         # 首次运行时初始化模型
# #         if not self.initialized:
# #             self.initialize_model(x)
# #
# #         # 保存原始输入用于后续反归一化
# #         x_original = x
# #
# #         # RevIN归一化
# #         if self.revin:
# #             # 转置为 [B, M, L] 进行归一化
# #             x_norm = x.permute(0, 2, 1)
# #             x_norm = self.revin_layer(x_norm, 'norm')
# #         else:
# #             x_norm = x.permute(0, 2, 1)
# #
# #         # 特征提取
# #         features = self.forward_feature(x_norm)  # [B, M, D, P]
# #
# #         # 输出重构
# #         B, M, D, P = features.shape
# #         output = self.head(features.permute(0, 1, 3, 2))  # [B, M, P, S]
# #         output = output.reshape(B, M, -1)  # [B, M, L]
# #         output = output[:, :, :self.seq_len]  # 确保长度一致
# #
# #         # 反归一化
# #         if self.revin:
# #             output = self.revin_layer(output, 'denorm')
# #
# #         # 转置回原始维度 [B, L, M]
# #         return output.permute(0, 2, 1)
# #
# #
# # class Model(nn.Module):
# #     def __init__(self, configs):
# #         super().__init__()
# #         self.model = ModernTCN(configs)
# #
# #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
# #         return self.model(x_enc)