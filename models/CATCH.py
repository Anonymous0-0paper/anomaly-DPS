'''
* @author: EmpyreanMoon
*
* @create: 2024-08-26 10:28
*
* @description: The structure of CATCH (modified for TimesNet compatibility)
'''

from layers.RevIN import RevIN
from layers.cross_channel_Transformer import Trans_C
# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.channel_mask import channel_mask_generator


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        configs.patch_size = 16
        configs.patch_stride = 8
        configs.cf_dim = 4
        configs.head_dim = 16
        configs.regular_lambda = 0.3
        configs.temperature = 0.07
        configs.individual = 0
        configs.head_dropout = 0.1

        # TimesNet compatibility parameters
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out if hasattr(configs, 'c_out') else configs.enc_in
        self.d_model = configs.d_model if hasattr(configs, 'd_model') else 512

        # Original CATCH parameters
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=0)
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)

        # Backbone
        self.re_attn = True
        self.mask_generator = channel_mask_generator(input_size=configs.patch_size, n_vars=self.enc_in)
        self.frequency_transformer = Trans_C(
            dim=configs.cf_dim,
            depth=configs.e_layers,
            heads=configs.n_heads,
            mlp_dim=configs.d_ff,
            dim_head=configs.head_dim,
            dropout=configs.dropout,
            patch_dim=configs.patch_size * 2,
            horizon=(self.seq_len + self.pred_len) * 2,  # Include pred_len
            d_model=configs.d_model * 2,
            regular_lambda=configs.regular_lambda,
            temperature=configs.temperature
        )

        # Head
        self.head_nf_f = configs.d_model * 2 * patch_num
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(
            self.individual,
            self.enc_in,
            self.head_nf_f,
            self.seq_len + self.pred_len,  # Support pred_len
            head_dropout=configs.head_dropout
        )
        self.head_f2 = Flatten_Head(
            self.individual,
            self.enc_in,
            self.head_nf_f,
            self.seq_len + self.pred_len,  # Support pred_len
            head_dropout=configs.head_dropout
        )

        self.ircom = nn.Linear((self.seq_len + self.pred_len) * 2, self.seq_len + self.pred_len)
        self.rfftlayer = nn.Linear((self.seq_len + self.pred_len) * 2 - 2, self.seq_len + self.pred_len)
        self.final = nn.Linear(self.seq_len + self.pred_len, self.pred_len if self.pred_len > 0 else self.seq_len)
        self.projection = nn.Linear(self.enc_in, self.c_out)

        # break up R&I:
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # TimesNet compatible interface
        z = x_enc
        z = self.revin_layer(z, 'norm')

        z = z.permute(0, 2, 1)
        z_fft = torch.fft.fft(z)
        z1 = z_fft.real
        z2 = z_fft.imag

        # Patching
        z1 = z1.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

        # Channel-wise processing
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        # Model shape
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_size = z1.shape[3]

        # Reshape and concatenate
        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, patch_size))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, patch_size))
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)
        z, dcloss = self.frequency_transformer(z_cat, channel_mask)

        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)

        complex_z = torch.complex(z1, z2)
        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # Project to output dimension
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')
        z = self.projection(z)

        # Select prediction part if needed
        if self.pred_len > 0:
            z = z[:, -self.pred_len:, :]

        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, output_len, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.output_len = output_len

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, output_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Sequential(
                nn.Linear(nf, nf),
                nn.ReLU(),
                nn.Linear(nf, nf),
                nn.ReLU(),
                nn.Linear(nf, output_len),
                nn.Dropout(head_dropout)
            )

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
        return x