import torch
import torch.nn as nn
import math


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = 'OmniAnomaly'
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # OmniAnomaly 参数设定
        self.n_feats = configs.enc_in
        self.n_hidden = 32  # GRU隐藏层大小
        self.n_latent = 8  # 潜在空间维度
        self.beta = 0.01  # KL散度权重

        # GRU层 - 处理时序依赖
        self.gru = nn.GRU(
            input_size=self.n_feats,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=False  # 使用(seq_len, batch, features)格式
        )

        # VAE编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)  # 输出mu和logvar
        )

        # VAE解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats)  # 输出重构特征
        )

        # 输出投影层 (保持与模板兼容)
        self.projection = nn.Linear(self.n_feats, configs.c_out)

    def core_forward(self, x_enc, hidden=None):
        """OmniAnomaly核心前向传播"""
        # 输入形状转换: [batch, seq_len, feats] -> [seq_len, batch, feats]
        x = x_enc.permute(1, 0, 2)
        batch_size = x.size(1)

        # 初始化隐藏状态
        if hidden is None:
            hidden = torch.zeros(2, batch_size, self.n_hidden, device=x.device)

        # GRU处理时序
        out, hidden = self.gru(x, hidden)

        # 编码器获取分布参数
        encoded = self.encoder(out.contiguous().view(-1, self.n_hidden))
        mu, logvar = torch.split(encoded, [self.n_latent, self.n_latent], dim=-1)

        # 重参数化采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 解码器重构
        recon = self.decoder(z)

        # 形状转换: [seq_len * batch, feats] -> [batch, seq_len, feats]
        recon = recon.view(self.seq_len, batch_size, -1).permute(1, 0, 2)
        mu = mu.view(self.seq_len, batch_size, -1).permute(1, 0, 2)
        logvar = logvar.view(self.seq_len, batch_size, -1).permute(1, 0, 2)

        return recon, mu, logvar, hidden

    def anomaly_detection(self, x_enc):
        """异常检测专用方法"""
        # Z-score归一化 (保持与模板兼容)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 通过核心模型
        recon, mu, logvar, _ = self.core_forward(x_enc)

        # 反归一化
        recon = recon * stdev + means

        # 返回重构结果 (符合模板接口)
        return self.projection(recon)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """兼容TimesNet的5参数接口"""
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

    def loss_function(self, recon_x, x, mu, logvar):
        """OmniAnomaly损失函数 (可在训练循环中使用)"""
        # 重构损失 (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.beta * kl_loss