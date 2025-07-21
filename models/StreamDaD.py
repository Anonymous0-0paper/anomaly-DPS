import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class TADBlock(nn.Module):
    """StreamTAD的核心模块 - 异常检测版本"""

    def __init__(self, configs):
        super(TADBlock, self).__init__()
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.kernel_size = configs.kernel_size if hasattr(configs, 'kernel_size') else 3

        # 时间注意力机制 (带因果掩码)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=configs.n_heads if hasattr(configs, 'n_heads') else 4,
            dropout=configs.dropout,
            batch_first=True
        )

        # 因果卷积层
        self.causal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=configs.d_model,
                out_channels=configs.d_ff,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1)  # 这里修复了括号
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=configs.d_ff,
                out_channels=configs.d_model,
                kernel_size=1
            )
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.Sigmoid()
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        B, T, C = x.size()

        # 创建因果注意力掩码
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # 残差连接
        residual = x

        # 时间注意力
        attn_out, _ = self.temporal_attn(x, x, x, attn_mask=mask)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)

        # 因果卷积
        conv_out = self.causal_conv(x.permute(0, 2, 1))
        conv_out = conv_out[:, :, :-(self.kernel_size - 1)].permute(0, 2, 1)

        # 门控融合
        gate_value = self.gate(torch.cat([x, conv_out], dim=-1))
        fused_out = gate_value * x + (1 - gate_value) * conv_out

        return self.norm2(residual + fused_out)


class Model(nn.Module):
    """与TimesNet输出格式兼容的StreamTAD模型"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.total_len = self.seq_len + self.pred_len

        # 数据嵌入层
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # TAD块堆叠
        self.tad_blocks = nn.ModuleList([
            TADBlock(configs) for _ in range(configs.e_layers)
        ])

        # 归一化
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 输出投影
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 自适应输出层
        if self.seq_len != self.total_len:
            self.adapt_conv = nn.Conv1d(
                in_channels=self.seq_len,
                out_channels=self.total_len,
                kernel_size=1
            )
        else:
            self.adapt_conv = None

    def anomaly_detection(self, x_enc):
        # 归一化处理
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 创建全序列张量（包含预测部分）
        input_seq = torch.zeros(x_enc.size(0), self.total_len, x_enc.size(2)).to(x_enc.device)
        input_seq[:, :self.seq_len, :] = x_enc

        # 嵌入层
        enc_out = self.enc_embedding(input_seq, None)  # [B, T_total, C]

        # 通过TAD块
        for block in self.tad_blocks:
            enc_out = block(enc_out)
            enc_out = self.layer_norm(enc_out)

        # 输出投影
        dec_out = self.projection(enc_out)  # [B, T_total, D_out]

        # 处理长度适配 - 如果需要调整时间维度
        if self.adapt_conv is not None:
            # [B, T_total, D_out] -> [B, D_out, T_total]
            dec_out = dec_out.permute(0, 2, 1)
            dec_out = self.adapt_conv(dec_out)
            # [B, D_out, T_total] -> [B, T_total, D_out]
            dec_out = dec_out.permute(0, 2, 1)

        # 反归一化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

        # 返回与TimesNet相同的输出格式
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """统一的前向传播接口 - 返回张量而非字典"""
        # 在异常检测任务中使用相同的处理流程
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)

        # 其他任务的处理逻辑可以在此扩展
        raise NotImplementedError(f"Task {self.task_name} not implemented for StreamTAD")


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from layers.Embed import DataEmbedding
# #
# #
# # class TADBlock(nn.Module):
# #     """StreamTAD的核心模块，替换TimesNet中的TimesBlock"""
# #
# #     def __init__(self, configs):
# #         super(TADBlock, self).__init__()
# #         self.d_model = configs.d_model
# #         self.d_ff = configs.d_ff
# #         self.kernel_size = configs.kernel_size if hasattr(configs, 'kernel_size') else 3
# #
# #         # 时间注意力机制
# #         self.temporal_attn = nn.MultiheadAttention(
# #             embed_dim=configs.d_model,
# #             num_heads=configs.n_heads if hasattr(configs, 'n_heads') else 4,
# #             dropout=configs.dropout,
# #             batch_first=True
# #         )
# #
# #         # 因果卷积层 (确保不会使用未来信息)
# #         self.causal_conv = nn.Sequential(
# #             nn.Conv1d(
# #                 in_channels=configs.d_model,
# #                 out_channels=configs.d_ff,
# #                 kernel_size=self.kernel_size,
# #                 padding=(self.kernel_size - 1),  # 保持长度不变
# #                 padding_mode='replicate'
# #             ),
# #             nn.GELU(),
# #             nn.Conv1d(
# #                 in_channels=configs.d_ff,
# #                 out_channels=configs.d_model,
# #                 kernel_size=1
# #             )
# #         )
# #
# #         # 自适应门控机制
# #         self.gate = nn.Sequential(
# #             nn.Linear(configs.d_model * 2, configs.d_model),
# #             nn.Sigmoid()
# #         )
# #
# #         # 归一化层
# #         self.norm1 = nn.LayerNorm(configs.d_model)
# #         self.norm2 = nn.LayerNorm(configs.d_model)
# #         self.dropout = nn.Dropout(configs.dropout)
# #
# #     def forward(self, x):
# #         """
# #         输入: [B, T, C]
# #         输出: [B, T, C]
# #         """
# #         # 残差连接
# #         residual = x
# #
# #         # 时间注意力（添加因果掩码）
# #         device = x.device
# #         batch_size, seq_len, _ = x.size()
# #         mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
# #         attn_out, _ = self.temporal_attn(x, x, x, attn_mask=mask)
# #         attn_out = self.dropout(attn_out)
# #         x = self.norm1(x + attn_out)
# #
# #         # 因果卷积处理 (需要转换为 [B, C, T] 格式)
# #         conv_out = self.causal_conv(x.permute(0, 2, 1))
# #
# #         # 裁剪右侧多余的填充 (保持因果性)
# #         conv_out = conv_out[:, :, :-(self.kernel_size - 1)].permute(0, 2, 1)
# #
# #         # 门控融合
# #         gate_input = torch.cat([x, conv_out], dim=-1)
# #         gate_value = self.gate(gate_input)
# #         fused_out = gate_value * x + (1 - gate_value) * conv_out
# #
# #         # 最终输出
# #         return self.norm2(residual + fused_out)
# #
# #
# # class Model(nn.Module):
# #     """完整的StreamTAD模型，兼容TimesNet的配置接口"""
# #
# #     def __init__(self, configs):
# #         super(Model, self).__init__()
# #         self.configs = configs
# #         self.task_name = configs.task_name
# #         self.seq_len = configs.seq_len
# #         self.label_len = configs.label_len
# #         self.pred_len = configs.pred_len
# #         self.total_len = self.seq_len + self.pred_len
# #
# #         # 数据嵌入层 (复用TimesNet的Embedding)
# #         self.enc_embedding = DataEmbedding(
# #             configs.enc_in,
# #             configs.d_model,
# #             configs.embed,
# #             configs.freq,
# #             configs.dropout
# #         )
# #
# #         # StreamTAD块堆叠
# #         self.tad_blocks = nn.ModuleList([
# #             TADBlock(configs) for _ in range(configs.e_layers)
# #         ])
# #
# #         # 归一化和输出投影
# #         self.layer_norm = nn.LayerNorm(configs.d_model)
# #         self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
# #
# #         # 修复自适应输出层问题
# #         # 使用插值替代卷积进行长度调整
# #         self.adapt_output = nn.Sequential(
# #             nn.Linear(self.seq_len, self.total_len),
# #             nn.ReLU()
# #         ) if self.seq_len != self.total_len else nn.Identity()
# #
# #     def anomaly_detection(self, x_enc):
# #         """异常检测模式，保持与TimesNet相同的归一化流程"""
# #         # 归一化处理
# #         means = x_enc.mean(1, keepdim=True).detach()
# #         x_enc = x_enc.sub(means)
# #         stdev = torch.sqrt(
# #             torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
# #         )
# #         x_enc = x_enc.div(stdev)
# #
# #         # 创建全序列张量（包含预测部分）
# #         input_seq = torch.zeros(x_enc.size(0), self.total_len, x_enc.size(2)).to(x_enc.device)
# #         input_seq[:, :self.seq_len, :] = x_enc
# #
# #         # 嵌入层
# #         enc_out = self.enc_embedding(input_seq, None)  # [B, T_total, C]
# #
# #         # 通过StreamTAD块
# #         for block in self.tad_blocks:
# #             enc_out = block(enc_out)
# #             enc_out = self.layer_norm(enc_out)
# #
# #         # 输出投影
# #         dec_out = self.projection(enc_out)  # [B, T_total, D_out]
# #
# #         # 处理长度适配 - 使用线性层替代卷积
# #         if self.seq_len != self.total_len:
# #             # [B, T_total, D_out] -> [B, D_out, T_total]
# #             dec_out = dec_out.permute(0, 2, 1)
# #             dec_out = self.adapt_output(dec_out)  # 线性层调整
# #             dec_out = dec_out.permute(0, 2, 1)
# #
# #         # 反归一化
# #         dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
# #         return dec_out
# #
# #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
# #         """统一的前向传播接口"""
# #         # 在异常检测任务中使用相同的处理流程
# #         if self.task_name == 'anomaly_detection':
# #             return self.anomaly_detection(x_enc)
# #
# #         # 其他任务的处理逻辑可以在此扩展
# #         raise NotImplementedError(f"Task {self.task_name} not implemented for StreamTAD")