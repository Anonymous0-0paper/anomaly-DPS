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

