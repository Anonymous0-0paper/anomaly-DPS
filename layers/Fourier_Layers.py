import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralFeedForward(nn.Module):
    """Spectral Feed Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(SpectralFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.freq_transform = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Standard feed forward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))

        # Frequency domain processing
        x_fft = torch.fft.fft(x, dim=1)
        x_freq_real = torch.real(x_fft)
        freq_output = self.freq_transform(x_freq_real)
        freq_output = torch.real(torch.fft.ifft(
            torch.complex(freq_output, torch.zeros_like(freq_output)), dim=1
        ))

        # Combine outputs
        return ff_output + freq_output

class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FourierTransformerLayer(nn.Module):
    """Fourier Transformer Layer"""
    def __init__(self, d_model, n_heads, d_ff, n_scales=4, dropout=0.1):
        super(FourierTransformerLayer, self).__init__()
        self.d_model = d_model

        # Core components
        self.multi_scale_attention = MultiScaleFourierAttention(
            d_model, n_heads, n_scales, dropout
        )
        self.adaptive_gating = AdaptiveFrequencyGating(d_model)
        self.spectral_ffn = SpectralFeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-scale Fourier attention
        attn_output, attention_weights = self.multi_scale_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Adaptive frequency gating
        gated_output = self.adaptive_gating(x)
        x = self.norm2(x + self.dropout(gated_output))

        # Spectral feed forward network
        ffn_output = self.spectral_ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x, attention_weights

class FourierAttention(nn.Module):
    """
    Fourier Attention mechanism that operates on both time and frequency domains.
    This attention mechanism computes similarities in both temporal and spectral spaces,
    allowing the model to capture anomalies that manifest in frequency characteristics.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(FourierAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for queries, keys, values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Frequency weighting parameters
        self.freq_weight = nn.Parameter(torch.ones(1))
        self.time_weight = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        Forward pass of Fourier Attention.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)
        V = self.w_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute time domain attention scores
        time_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, n_heads, seq_len, seq_len)

        # Compute frequency domain representations
        Q_fft = torch.fft.fft(Q, dim=-2)  # FFT along sequence dimension
        K_fft = torch.fft.fft(K, dim=-2)

        # Compute frequency domain attention scores
        # Use real part of complex multiplication for stability
        freq_scores = torch.real(torch.matmul(Q_fft, torch.conj(K_fft).transpose(-2, -1)))

        # Combine time and frequency scores
        combined_scores = (self.time_weight * time_scores +
                           self.freq_weight * freq_scores) / self.scale

        # Apply mask if provided
        if mask is not None:
            combined_scores.masked_fill_(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(combined_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(output)

        return output, attention_weights

class MultiScaleFourierAttention(nn.Module):
    """
    Multi-scale Fourier Attention that processes different frequency bands separately.
    """
    def __init__(self, d_model, n_heads, n_scales=4, dropout=0.1):
        super(MultiScaleFourierAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_scales = n_scales

        # Create attention modules for each scale
        self.scale_attentions = nn.ModuleList([
            FourierAttention(d_model, n_heads, dropout)
            for _ in range(n_scales)
        ])

        # Scale combination weights
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        self.combination_layer = nn.Linear(d_model * n_scales, d_model)

        # Chunk processing parameters
        self.chunk_size = 2  # Number of scales to process at a time, adjustable

    def create_frequency_bands(self, x, scale_idx):
        """
        Create frequency band filters for multi-scale analysis.
        Uses real FFT for memory optimization (50% less memory usage)
        """
        batch_size, seq_len, d_model = x.shape

        # Ensure input tensor is contiguous
        x = x.contiguous()

        # Use real FFT for optimization (50% less memory)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        # Create frequency band mask
        freq_bins = x_fft.shape[1]  # Number of bins for real FFT
        band_size = max(1, freq_bins // self.n_scales)  # Ensure at least 1 bin
        start_bin = scale_idx * band_size
        end_bin = min((scale_idx + 1) * band_size, freq_bins)

        # Create mask on the correct device
        mask = torch.zeros(freq_bins, device=x.device)
        mask[start_bin:end_bin] = 1.0

        # Apply mask
        x_fft_filtered = x_fft * mask.unsqueeze(0).unsqueeze(-1)

        # Inverse transform back to time domain
        x_filtered = torch.fft.irfft(x_fft_filtered, n=seq_len, dim=1, norm='ortho')

        return x_filtered

    def forward(self, x, mask=None):
        """
        Forward pass of Multi-scale Fourier Attention with chunked processing.
        """
        # Chunk processing - reduce the number of scales processed simultaneously
        active_scales = min(self.n_scales, self.chunk_size)  # Process at most chunk_size scales at once
        scale_outputs = []
        all_attention_weights = []

        # Add input validation
        if torch.isnan(x).any():
            print("Warning: Input contains NaN values")
            x = torch.nan_to_num(x)

        # Process scales in chunks
        for scale_group in range(0, self.n_scales, active_scales):
            group_outputs = []
            group_attentions = []

            for scale_idx in range(scale_group, min(scale_group + active_scales, self.n_scales)):
                try:
                    # Create frequency band
                    x_scale = self.create_frequency_bands(x, scale_idx)

                    # Ensure frequency band data is valid
                    if torch.isnan(x_scale).any():
                        print(f"Frequency band {scale_idx} contains NaN values")
                        x_scale = torch.nan_to_num(x_scale)

                    # Apply attention
                    output, attn_weights = self.scale_attentions[scale_idx](x_scale, mask)
                    group_outputs.append(output)
                    group_attentions.append(attn_weights)

                    # Immediately release intermediate variables
                    del x_scale, output, attn_weights

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Fallback to original input when out of memory
                        print(f"Warning: Scale {scale_idx} out of memory, using original input")
                        output, attn_weights = self.scale_attentions[scale_idx](x, mask)
                        group_outputs.append(output)
                        group_attentions.append(attn_weights)
                    else:
                        raise
                except Exception as e:
                    print(f"Error processing scale {scale_idx}: {str(e)}")
                    # Fallback to original input on failure
                    output, attn_weights = self.scale_attentions[scale_idx](x, mask)
                    group_outputs.append(output)
                    group_attentions.append(attn_weights)

            # Save this group's results
            scale_outputs.extend(group_outputs)
            all_attention_weights.extend(group_attentions)

            # Manually empty CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Combine outputs: concatenate outputs from all scales
        combined_output = torch.cat(scale_outputs, dim=-1)
        final_output = self.combination_layer(combined_output)

        # Combine attention weights: weighted average
        if all_attention_weights:
            combined_attention = torch.stack(all_attention_weights, dim=0)
            weighted_attention = torch.sum(
                combined_attention * self.scale_weights.view(-1, 1, 1, 1, 1),
                dim=0
            )
        else:
            # Fallback: create empty attention weights
            weighted_attention = torch.zeros(
                x.size(0), self.n_heads, x.size(1), x.size(1),
                device=x.device
            )

        # Release variables no longer needed
        del combined_output, scale_outputs, all_attention_weights

        return final_output, weighted_attention

class AdaptiveFrequencyGating(nn.Module):
    """
    Adaptive gating mechanism that dynamically weights time vs frequency contributions.
    """
    def __init__(self, d_model):
        super(AdaptiveFrequencyGating, self).__init__()
        self.d_model = d_model

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply adaptive frequency gating.
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Gated output combining time and frequency representations
        """
        # Compute frequency domain representation
        x_fft = torch.fft.fft(x, dim=1)
        x_freq = torch.real(torch.fft.ifft(x_fft, dim=1))

        # Concatenate time and frequency features
        combined = torch.cat([x, x_freq], dim=-1)

        # Compute gating weights
        gate = self.gate_net(combined)

        # Apply gating: weighted combination of time and frequency
        output = gate * x + (1 - gate) * x_freq

        return output
