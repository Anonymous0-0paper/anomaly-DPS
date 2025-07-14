import sys

sys.path.append("../..")
from layers.ConvNet import TemporalConvNet
from layers.Embed import DataEmbedding  # 新增时间嵌入层
import math
import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        # 使用TimesNet风格的参数命名
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.c_out = configs.c_out

        # 时间嵌入层（类似TimesNet）
        self.enc_embedding = DataEmbedding(
            self.enc_in, self.d_model, configs.embed, configs.freq, configs.dropout
        )

        # 时序卷积网络
        self.encoder = TemporalConvNet(
            num_inputs=self.d_model,
            num_channels=[self.d_model, self.d_model // 2],
            kernel_size=3,
            dilation_factor=2,
            dropout=configs.dropout
        )

        # 重构层（1x1卷积）
        self.decoder = nn.Conv1d(
            self.d_model // 2,
            self.enc_in,
            kernel_size=1
        )

        # 异常评分层
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()
        )

        # 输出投影层（类似TimesNet）
        self.projection = nn.Linear(self.enc_in + 1, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # TimesNet风格归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # 时间嵌入 [B, seq_len, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 调整维度 [B, d_model, seq_len]
        enc_out = enc_out.permute(0, 2, 1)

        # 时序卷积编码
        conv_out = self.encoder(enc_out)  # [B, d_model//2, seq_len]

        # 序列重构 [B, enc_in, seq_len]
        dec_out = self.decoder(conv_out).permute(0, 2, 1)  # [B, seq_len, enc_in]

        # 异常评分 [B, seq_len, 1]
        anomaly_scores = self.anomaly_scorer(conv_out.permute(0, 2, 1))

        # 只对重构部分进行反归一化
        dec_out_denorm = dec_out * stdev + means

        # 拼接重构序列和异常评分 [B, seq_len, enc_in+1]
        combined = torch.cat([dec_out_denorm, anomaly_scores], dim=-1)

        # 投影到输出维度 [B, seq_len, c_out]
        output = self.projection(combined)

        # 扩展到预测长度 [B, seq_len+pred_len, c_out]
        output = F.pad(output, (0, 0, 0, self.pred_len))

        return output

# import sys
#
# sys.path.append("../..")
# from layers.ConvNet import TemporalConvNet
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DeepSVDD(nn.Module):
#     def __init__(self, input_dim):
#         super(DeepSVDD, self).__init__()
#         self.center = nn.Parameter(torch.Tensor(input_dim))
#         self.radius = nn.Parameter(torch.Tensor(1))
#         nn.init.constant_(self.center, 0.0)
#         nn.init.constant_(self.radius, 0.0)
#
#     def forward(self, x):
#         # 计算每个时间步到中心的距离 [B, L]
#         dist = torch.sum((x - self.center) ** 2, dim=-1)
#         return dist.unsqueeze(-1), self.center, self.radius ** 2
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.configs = configs
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#
#         # 动态参数设置
#         num_inputs = configs.enc_in
#         num_channels = [configs.d_model] * 2  # 使用TimesNet的d_model配置
#
#         # 序列编码器 - 确保保持序列长度
#         self.encoder = TemporalConvNet(
#             num_inputs=num_inputs,
#             num_channels=num_channels,
#             kernel_size=configs.kernel_size if hasattr(configs, 'kernel_size') else 3,
#             dilation_factor=configs.dilation_factor if hasattr(configs, 'dilation_factor') else 2,
#             dropout=configs.dropout
#         )
#
#         # 添加1x1卷积层确保输出通道数与输入相同
#         self.channel_adjust = nn.Conv1d(
#             num_channels[-1],
#             num_inputs,
#             kernel_size=1
#         )
#
#         # 时间步异常检测
#         self.anomaly_scorer = DeepSVDD(
#             input_dim=num_channels[-1]
#         )
#
#     def anomaly_detection(self, x_enc):
#         B, L, D = x_enc.shape  # 获取输入序列长度
#
#         # TimesNet标准归一化流程
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc.sub(means)
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc = x_enc.div(stdev)
#
#         # 编码序列 [B, D, L]
#         x_transposed = x_enc.transpose(1, 2)
#         enc_out = self.encoder(x_transposed)
#
#         # 确保编码器输出长度与输入相同
#         if enc_out.size(2) != L:
#             # 使用插值调整序列长度
#             enc_out = F.interpolate(enc_out, size=L, mode='linear', align_corners=True)
#
#         # 调整通道数 [B, D, L]
#         dec_out = self.channel_adjust(enc_out).transpose(1, 2)
#
#         # 时间步异常评分 [B, L, 1]
#         time_step_features = enc_out.transpose(1, 2)  # [B, L, C]
#         anomaly_scores, _, _ = self.anomaly_scorer(time_step_features)
#
#         # TimesNet标准反归一化
#         dec_out = dec_out * stdev + means
#
#         # 融合异常评分到输出通道
#         return torch.cat([dec_out, anomaly_scores], dim=-1)
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'anomaly_detection':
#             return self.anomaly_detection(x_enc)
#         else:
#             # 其他任务兼容性
#             return self.anomaly_detection(x_enc)[:, :self.pred_len, :]
#
#
# # import sys
# #
# # sys.path.append("../..")
# # from layers.ConvNet import TemporalConvNet
# # from layers.MLP import MLP
# # import math
# #
# # import torch
# # import torch.nn.functional as F
# # from torch import nn
# # from torch.autograd import Function
# # from torch.nn.utils import weight_norm
# #
# # # Helper function for reversing the discriminator backprop
# # from torch.autograd import Function
# #
# #
# # class ReverseLayerF(Function):
# #     @staticmethod
# #     def forward(ctx, x, alpha):
# #         ctx.alpha = alpha
# #         return x.view_as(x)
# #
# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         output = grad_output.neg() * ctx.alpha
# #         return output, None
# #
# #
# # class Discriminator(nn.Module):
# #     def __init__(self, input_dim, hidden_dim=256, output_dim=1):
# #         super(Discriminator, self).__init__()
# #
# #         self.model = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.LeakyReLU(0.2),
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.LeakyReLU(0.2),
# #             nn.Linear(hidden_dim, output_dim),
# #             nn.Sigmoid()
# #         )
# #
# #     def forward(self, x):
# #         return self.model(x)
# #
# # class Model(nn.Module):
# #     def __init__(self, configs):
# #         super(Model, self).__init__()
# #         self.configs = configs
# #         self.task_name = configs.task_name
# #         self.seq_len = configs.seq_len
# #         self.pred_len = configs.pred_len
# #
# #         # 动态参数设置
# #         num_inputs = configs.enc_in
# #         num_channels = [64, 128]  # 示例通道设置
# #         mlp_hidden_dim = 256
# #         use_batch_norm = True
# #         kernel_size = 3
# #         stride = 1
# #         dilation_factor = 2
# #         dropout = 0.2
# #
# #         # 序列编码器
# #         self.encoder = TemporalConvNet(
# #             num_inputs=num_inputs,
# #             num_channels=num_channels,
# #             kernel_size=kernel_size,
# #             stride=stride,
# #             dilation_factor=dilation_factor,
# #             dropout=dropout
# #         )
# #
# #         # 序列解码器 (1x1卷积重构)
# #         self.decoder = nn.Conv1d(
# #             num_channels[-1],
# #             num_inputs,
# #             kernel_size=1
# #         )
# #
# #         # 时间步异常检测
# #         self.anomaly_scorer = DeepSVDD(
# #             input_dim=num_channels[-1],
# #             hidden_dim=mlp_hidden_dim,
# #             output_dim=1,
# #             use_batch_norm=use_batch_norm
# #         )
# #
# #     def anomaly_detection(self, x_enc):
# #         # 归一化 (TimesNet标准流程)
# #         means = x_enc.mean(1, keepdim=True).detach()
# #         x_enc = x_enc.sub(means)
# #         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
# #         x_enc = x_enc.div(stdev)
# #
# #         # 编码序列 [B, D, L]
# #         enc_out = self.encoder(x_enc.transpose(1, 2))
# #
# #         # 重构序列 [B, L, D]
# #         dec_out = self.decoder(enc_out).transpose(1, 2)
# #
# #         # 时间步异常评分 [B, L, 1]
# #         time_step_features = enc_out.transpose(1, 2)  # [B, L, C]
# #         anomaly_scores, _, _ = self.anomaly_scorer(time_step_features)
# #
# #         # 反归一化重构序列
# #         dec_out = dec_out * stdev + means
# #
# #         # 融合异常评分到输出通道
# #         return torch.cat([dec_out, anomaly_scores.unsqueeze(-1)], dim=-1)
# #
# #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
# #         return self.anomaly_detection(x_enc)
# #
# #
# # class DeepSVDD(nn.Module):
# #     def __init__(self, input_dim, hidden_dim, output_dim, use_batch_norm):
# #         super(DeepSVDD, self).__init__()
# #
# #         # Encoder layers
# #         # self.encoder = nn.Sequential(
# #         #     nn.Linear(input_dim, hidden_dim),
# #         #     nn.ReLU(inplace=True),
# #         #     nn.Linear(hidden_dim, hidden_dim),
# #         #     nn.ReLU(inplace=True),
# #         # )
# #
# #         # Center and radius of the hypersphere
# #         self.center = nn.Parameter(torch.Tensor(input_dim))
# #         self.radius = nn.Parameter(torch.Tensor(1))
# #
# #         # # Decoder layers (optional)
# #         # self.decoder = nn.Sequential(
# #         #     nn.Linear(hidden_dim, hidden_dim),
# #         #     nn.ReLU(inplace=True),
# #         #     nn.Linear(hidden_dim, output_dim),
# #         # )
# #         #
# #         # # Batch normalization
# #         # self.use_batch_norm = use_batch_norm
# #         # if use_batch_norm:
# #         #     self.batch_norm = nn.BatchNorm1d(hidden_dim)
# #
# #         # Initialize parameters
# #         nn.init.constant_(self.center, 0.0)
# #         nn.init.constant_(self.radius, 0.0)
# #
# #     def _init_weights(self):
# #         # nn.init.xavier_uniform_(self.encoder[0].weight)
# #         # nn.init.constant_(self.encoder[0].bias, 0.0)
# #         # nn.init.xavier_uniform_(self.encoder[2].weight)
# #         # nn.init.constant_(self.encoder[2].bias, 0.0)
# #         #
# #         # if self.use_batch_norm:
# #         #     nn.init.constant_(self.batch_norm.weight, 1)
# #         #     nn.init.constant_(self.batch_norm.bias, 0)
# #
# #         nn.init.constant_(self.center, 0.0)
# #         nn.init.constant_(self.radius, 0.0)
# #
# #     def forward(self, x, statics):
# #         # # Encode the input
# #         # encoded = self.encoder(x)
# #         #
# #         # # Apply batch normalization if enabled
# #         # if self.use_batch_norm:
# #         #     encoded = self.batch_norm(encoded)
# #         #
# #         # decoded = self.decoder(encoded)
# #         # encoded = x.clone()
# #         # decoded = x.clone()
# #         tmp_x = x.clone()
# #
# #         # Calculate the distance to the center
# #         dist = torch.sum((tmp_x - self.center) ** 2, dim=1)
# #
# #         # Calculate the squared radius
# #         squared_radius = self.radius ** 2
# #
# #         # Return the encoded representation, distance, and squared radius
# #         return dist.unsqueeze(-1), self.center, squared_radius
