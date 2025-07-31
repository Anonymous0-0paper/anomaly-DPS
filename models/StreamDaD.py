import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import logging

from layers.gating import DynamicGatingNetwork
from layers.continual import ContinualLearner
from layers.drift_detection import DriftDetector

logger = logging.getLogger(__name__)


class GRUEncoder(nn.Module):
    """GRU-based encoder for processing gated input windows."""

    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.enc_in
        self.hidden_dim = configs.d_model
        self.num_layers = configs.e_layers

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=configs.dropout if self.num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.gru(x, hidden)
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output, hidden


class GRUDecoder(nn.Module):
    """GRU-based decoder for reconstructing input from encoded representations."""

    def __init__(self, configs):
        super().__init__()
        self.hidden_dim = configs.d_model
        self.output_dim = configs.enc_in
        self.num_layers = configs.e_layers

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=configs.dropout if self.num_layers > 1 else 0
        )

        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, encoded: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        output, _ = self.gru(encoded, hidden)
        output = self.layer_norm(output)
        output = self.dropout(output)
        reconstructed = self.output_projection(output)
        return reconstructed


class IdentityGating(nn.Module):
    """Identity gating layer for ablation studies."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, prev_hidden, drift_signal, correlations):
        batch_size = drift_signal.shape[0]
        return torch.ones(batch_size, self.input_dim, device=drift_signal.device)


class Model(nn.Module):
    """Main Stream-DAD model with TimesNet-compatible interface."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        # Core encoder-decoder architecture
        self.encoder = GRUEncoder(configs)
        self.decoder = GRUDecoder(configs)

        # Dynamic gating mechanism
        if getattr(configs, 'disable_gating', False):
            self.gating_network = IdentityGating(self.enc_in)
        else:
            self.gating_network = DynamicGatingNetwork(
                input_dim=self.enc_in,
                hidden_dim=self.d_model,
                configs=configs
            )

        # Drift detection
        self.drift_detector = DriftDetector(
            input_dim=self.enc_in,
            window_size=getattr(configs, 'drift_window_size', 50),
            configs=configs
        )

        # Continual learning components
        self.continual_learner = ContinualLearner(
            model=self,
            configs=configs
        )

        # Normalization statistics
        self.register_buffer('running_mean', torch.zeros(self.enc_in))
        self.register_buffer('running_var', torch.ones(self.enc_in))
        self.register_buffer('num_samples', torch.tensor(0))

        # Previous states
        self.prev_hidden = None
        self.prev_gates = None
        self.adaptation_mode = False

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=(0, 1))
            batch_var = x.var(dim=(0, 1), unbiased=False)
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            self.num_samples += x.shape[0] * x.shape[1]
        return (x - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # TimesNet-compatible forward pass
        result = self.anomaly_detection(x_enc)

        # 修复：返回重构结果而不是字典
        return result['reconstructed']

    def anomaly_detection(self, x_enc):
        # TimesNet-style normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # Main processing
        x = x_enc
        batch_size = x.shape[0]

        # Dynamic gating
        drift_signal = self.drift_detector(x)
        correlations = self.compute_spatial_correlations(x)
        gates = self.gating_network(
            prev_hidden=self.prev_hidden,
            drift_signal=drift_signal,
            correlations=correlations
        )
        x_gated = x * gates.unsqueeze(1)

        # Encode and decode
        encoded, hidden = self.encoder(x_gated, self.prev_hidden)
        reconstructed = self.decoder(encoded)

        # Anomaly scores
        anomaly_scores = torch.norm(x - reconstructed, dim=-1, p=2)

        # Update states
        if batch_size == 1:
            self.prev_hidden = hidden.detach()
            self.prev_gates = gates.detach()

        # TimesNet-style denormalization
        reconstructed = reconstructed.mul(stdev) + means

        # 返回字典用于其他方法
        self.last_output = {
            'reconstructed': reconstructed,
            'anomaly_scores': anomaly_scores,
            'gates': gates
        }

        return self.last_output

    def compute_spatial_correlations(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, input_dim = x.shape
        correlations = torch.zeros(batch_size, input_dim, device=x.device)
        for b in range(batch_size):
            sample = x[b]
            sample_centered = sample - sample.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(sample_centered.T, sample_centered) / (window_size - 1)
            std_devs = torch.sqrt(torch.diag(cov_matrix))
            corr_matrix = cov_matrix / (std_devs.unsqueeze(0) * std_devs.unsqueeze(1) + 1e-8)
            threshold = getattr(self.configs, 'correlation_threshold', 0.3)
            significant_corrs = (torch.abs(corr_matrix) > threshold).float()
            correlations[b] = significant_corrs.sum(dim=1) - 1
        return correlations

    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 使用最后的前向传播输出
        if not hasattr(self, 'last_output'):
            self.anomaly_detection(x)

        reconstructed = self.last_output['reconstructed']
        gates = self.last_output.get('gates', None)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)

        # Regularization losses
        ewc_loss = self.continual_learner.compute_ewc_loss()

        consistency_loss = torch.tensor(0.0, device=x.device)
        if gates is not None and self.prev_gates is not None:
            consistency_loss = F.mse_loss(gates, self.prev_gates)

        sparsity_loss = torch.tensor(0.0, device=x.device)
        if gates is not None:
            l1_loss = torch.norm(gates, p=1, dim=-1).mean()
            entropy_loss = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean()
            sparsity_loss = l1_loss + getattr(self.configs, 'lambda_entropy', 0.001) * entropy_loss

        # Total loss with configurable weights
        total_loss = (recon_loss +
                      getattr(self.configs, 'lambda_ewc', 0.01) * ewc_loss +
                      getattr(self.configs, 'lambda_cons', 0.001) * consistency_loss +
                      getattr(self.configs, 'lambda_sparsity', 0.0001) * sparsity_loss)

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'ewc_loss': ewc_loss,
            'consistency_loss': consistency_loss,
            'sparsity_loss': sparsity_loss
        }

    def adapt_to_drift(self, x: torch.Tensor) -> None:
        self.adaptation_mode = True
        self.continual_learner.update_fisher_information(x)
        drift_magnitude = self.drift_detector.get_current_drift_magnitude()
        self.adapt_hyperparameters(drift_magnitude)
        self.adaptation_mode = False

    def adapt_hyperparameters(self, drift_magnitude: float) -> None:
        base_ewc = getattr(self.configs, 'lambda_ewc_base', 0.01)
        self.configs.lambda_ewc = base_ewc * torch.exp(-torch.tensor(drift_magnitude))

        base_cons = getattr(self.configs, 'lambda_cons_base', 0.001)
        self.configs.lambda_cons = base_cons * (1 + drift_magnitude)

    def reset_states(self) -> None:
        self.prev_hidden = None
        self.prev_gates = None
        self.drift_detector.reset()
        self.continual_learner.reset()

