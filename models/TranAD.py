import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

from layers.Transformer_EncDec import PositionalEncoding
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = 'TranAD'
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # 使用TimesNet的嵌入层
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # TranAD模型参数
        self.n_window = configs.seq_len
        self.n = self.d_model * self.n_window

        # 位置编码
        self.pos_encoder = PositionalEncoding(2 * self.d_model, 0.1, self.n_window)

        # 使用PyTorch内置Transformer层
        encoder_layer = TransformerEncoderLayer(
            d_model=2 * self.d_model,
            nhead=self.d_model,
            dim_feedforward=16,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, 1)

        decoder_layer1 = TransformerDecoderLayer(
            d_model=2 * self.d_model,
            nhead=self.d_model,
            dim_feedforward=16,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layer1, 1)

        decoder_layer2 = TransformerDecoderLayer(
            d_model=2 * self.d_model,
            nhead=self.d_model,
            dim_feedforward=16,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layer2, 1)

        self.fcn = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid()
        )

        # 输出投影层
        self.projection = nn.Linear(self.d_model, configs.c_out)

    def encode(self, src, c):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        return memory

    def core_forward(self, x_enc, x_mark_enc):
        """核心前向传播逻辑"""
        # 嵌入层处理
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 转换为[T, B, d_model]格式
        src = enc_out.permute(1, 0, 2)

        # 第一阶段：无异常分数
        c0 = torch.zeros_like(src)
        memory = self.encode(src, c0)
        tgt = src.repeat(1, 1, 2)
        x1 = self.fcn(self.transformer_decoder1(tgt, memory))

        # 第二阶段：带异常分数
        c1 = (x1 - src) ** 2
        memory = self.encode(src, c1)
        x2 = self.fcn(self.transformer_decoder2(tgt, memory))

        # 转换回[B, T, d_model]格式
        x2 = x2.permute(1, 0, 2)
        return self.projection(x2)

    def anomaly_detection(self, x_enc):
        """异常检测专用方法"""
        # TimesNet风格归一化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 通过模型核心逻辑
        dec_out = self.core_forward(x_enc, None)

        # 反归一化
        dec_out = dec_out * stdev
        dec_out = dec_out.add(means)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """兼容TimesNet的5参数接口"""
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")