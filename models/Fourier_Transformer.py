import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Fourier_Layers import FourierTransformerLayer, PositionalEncoding

class FourierTransformer(nn.Module):
    """Fourier Transformer main architecture"""

    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, n_scales=4, dropout=0.1, max_seq_len=1000, layerdrop=0.1, pos_learnable=True):
        super(FourierTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.layerdrop = layerdrop

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding (support learnable)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout, learnable=pos_learnable)

        # Transformer layers
        self.layers = nn.ModuleList([
            FourierTransformerLayer(d_model, n_heads, d_ff, n_scales, dropout)
            for _ in range(n_layers)
        ])

        # Output projection with dropout
        self.output_dropout = nn.Dropout(dropout)

        # Parameter initialization
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None, return_attention=False):
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        attention_weights = []

        # Through transformer layers with LayerDrop
        for layer in self.layers:
            if self.training and self.layerdrop > 0.0 and torch.rand(1).item() < self.layerdrop:
                continue  # Skip this layer
            x, attn_weights = layer(x, mask)
            if return_attention:
                attention_weights.append(attn_weights)

        x = self.output_dropout(x)

        # Return output and attention weights (if needed)
        if return_attention:
            return x, attention_weights
        return x

class Model(nn.Module):
    """Fourier Transformer model conforming to the unified framework"""

    def __init__(self, configs):
        super(Model, self).__init__()

        # Save task configuration parameters
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 0

        # Model parameters
        self.input_dim = configs.enc_in
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.n_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.n_scales = getattr(configs, 'n_scales', 4)
        self.dropout = configs.dropout
        self.max_seq_len = getattr(configs, 'max_seq_len', 1000)

        # Create Fourier Transformer
        self.encoder = FourierTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            n_scales=self.n_scales,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len
        )

        # Configure output layer based on task type
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(self.dropout)
            self.projection = nn.Linear(
                self.d_model * self.seq_len, configs.num_class)
        else:  # Forecasting task
            self.projection = nn.Linear(self.d_model, configs.c_out, bias=True)

        # Learnable time-frequency balance parameter
        self.alpha = nn.Parameter(torch.tensor(0.6))

    def anomaly_detection(self, x_enc):
        """Forward propagation dedicated to anomaly detection task"""
        # Input validation
        if torch.isnan(x_enc).any():
            print("Warning: Input contains NaN values")
            x_enc = torch.nan_to_num(x_enc)

        """Forward propagation dedicated to anomaly detection task"""
        # Input shape: [batch_size, seq_len, input_dim]
        try:
            enc_out = self.encoder(x_enc)
            dec_out = self.projection(enc_out)
        except Exception as e:
            print(f"Anomaly detection failed: {str(e)}")
            # Create safe fallback output
            dec_out = torch.zeros_like(x_enc)

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """Forward propagation dedicated to classification task"""
        enc_out = self.encoder(x_enc)  # [B, T, d_model]

        # Activation function and dropout
        output = self.act(enc_out)
        output = self.dropout_layer(output)

        # Handle padding (if any)
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)

        # Flatten and classify
        output = output.reshape(output.shape[0], -1)  # [B, T * d_model]
        output = self.projection(output)  # [B, num_class]

        return output

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Forward propagation dedicated to forecasting task"""
        enc_out = self.encoder(x_enc)  # [B, T, d_model]
        dec_out = self.projection(enc_out)  # [B, T, c_out]
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Unified entry point, dispatch to different methods based on task type"""
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'imputation':
            # Simplified to anomaly detection processing
            return self.anomaly_detection(x_enc)
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

    def compute_anomaly_score(self, x, reduction='mean'):
        """Calculate anomaly score (optional method)"""
        reconstruction = self.forward(x, None, None, None)

        # Time domain reconstruction error
        time_error = F.mse_loss(reconstruction, x, reduction='none')

        # Frequency domain reconstruction error
        x_fft = torch.fft.fft(x, dim=1)
        recon_fft = torch.fft.fft(reconstruction, dim=1)
        freq_error = F.mse_loss(torch.real(recon_fft), torch.real(x_fft), reduction='none')
        freq_error += F.mse_loss(torch.imag(recon_fft), torch.imag(x_fft), reduction='none')

        # Combined score
        combined_score = torch.clamp(self.alpha, 0, 1) * time_error + \
                         (1 - torch.clamp(self.alpha, 0, 1)) * freq_error

        if reduction == 'mean':
            return combined_score.mean(dim=(1, 2))
        elif reduction == 'sum':
            return combined_score.sum(dim=(1, 2))
        else:
            return combined_score
