"""
Dynamic Feature Gating Mechanism for Stream-DAD.

This module implements the core innovation of Stream-DAD: the dynamic gating
network that adaptively selects relevant features based on drift sensitivity
and temporal correlation patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DynamicGatingNetwork(nn.Module):
    """
    Dynamic gating network that computes feature importance weights based on
    temporal context, drift sensitivity, and spatial correlations.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension from encoder
    """

    def __init__(self, input_dim: int, hidden_dim: int, configs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.configs = configs

        # Dimensions for gate computation inputs
        # [prev_hidden; drift_signal; correlations]
        gate_input_dim = hidden_dim + input_dim + input_dim

        # Gate computation network
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()  # Ensures gates in [0, 1]
        )

        # Gradient-based importance tracking
        self.register_buffer('importance_ema', torch.ones(input_dim) / input_dim)
        self.ema_alpha =  0.9

        # Temperature for softmax modulation
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self,
                prev_hidden: Optional[torch.Tensor],
                drift_signal: torch.Tensor,
                correlations: torch.Tensor,
                importance_gradients: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute dynamic feature gates."""
        batch_size = drift_signal.shape[0]
        device = drift_signal.device

        # Handle missing previous hidden state
        if prev_hidden is None:
            prev_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Ensure prev_hidden has correct batch size
        if prev_hidden.shape[0] != batch_size:
            if len(prev_hidden.shape) == 3:  # Multilayer GRU case
                prev_hidden = prev_hidden[-1]  # Take last layer
            # Adjust batch dimension
            if prev_hidden.shape[0] == 1 and batch_size > 1:
                prev_hidden = prev_hidden.repeat(batch_size, 1)
            elif prev_hidden.shape[0] > batch_size:
                prev_hidden = prev_hidden[:batch_size]

        # Concatenate inputs for gate computation
        gate_input = torch.cat([prev_hidden, drift_signal, correlations], dim=-1)

        # Compute base gates
        gates = self.gate_network(gate_input)

        # Modulate with gradient-based importance
        if importance_gradients is not None:
            if self.training:  # Update EMA only during training
                current_importance = importance_gradients.mean(dim=0)
                self.importance_ema = (self.ema_alpha * self.importance_ema +
                                       (1 - self.ema_alpha) * current_importance)
            # Apply softmax modulation
            importance_weights = F.softmax(self.importance_ema / self.temperature, dim=-1)
            gates = gates * importance_weights.unsqueeze(0)

        return gates

    @staticmethod
    def compute_correlations_standard(x: torch.Tensor,
                                      correlation_threshold: float = 0.3) -> torch.Tensor:
        """
        Standard correlation computation (fallback for small datasets).

        Args:
            x: Input tensor [batch_size, window_size, input_dim]
            correlation_threshold: Threshold for significant correlations

        Returns:
            correlations: Correlation scores for each feature [batch_size, input_dim]
        """
        batch_size, window_size, input_dim = x.shape
        device = x.device

        correlations = torch.zeros(batch_size, input_dim, device=device)

        for b in range(batch_size):
            sample = x[b]  # [window_size, input_dim]

            # Compute correlation matrix
            sample_centered = sample - sample.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(sample_centered.T, sample_centered) / (window_size - 1)

            # Standard deviations
            std_devs = torch.sqrt(torch.diag(cov_matrix)) + 1e-8

            # Correlation matrix
            corr_matrix = cov_matrix / (std_devs.unsqueeze(0) * std_devs.unsqueeze(1))

            # Count significant correlations
            significant_mask = (torch.abs(corr_matrix) > correlation_threshold).float()
            correlations[b] = significant_mask.sum(dim=1) - 1  # Exclude self-correlation

        return correlations


class AdaptiveGatingStrategy:
    """
    Adaptive strategy for adjusting gating behavior based on drift conditions.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.drift_history = []
        self.performance_history = []

    def should_increase_sparsity(self,
                                 current_drift: float,
                                 current_performance: float,
                                 window_size: int = 10) -> bool:
        """
        Determine if sparsity should be increased based on recent performance.

        Args:
            current_drift: Current drift magnitude
            current_performance: Current anomaly detection performance
            window_size: Window size for history analysis

        Returns:
            should_increase: Whether to increase sparsity
        """
        # Update history
        self.drift_history.append(current_drift)
        self.performance_history.append(current_performance)

        # Keep only recent history
        if len(self.drift_history) > window_size:
            self.drift_history = self.drift_history[-window_size:]
            self.performance_history = self.performance_history[-window_size:]

        # Not enough history
        if len(self.performance_history) < 5:
            return False

        # Check if performance is declining with high feature usage
        recent_performance = np.mean(self.performance_history[-3:])
        earlier_performance = np.mean(self.performance_history[:-3])

        performance_declining = recent_performance < earlier_performance - 0.02
        high_drift = current_drift > 0.3

        return performance_declining and high_drift

    def should_decrease_sparsity(self,
                                 current_drift: float,
                                 current_performance: float,
                                 window_size: int = 10) -> bool:
        """
        Determine if sparsity should be decreased.

        Args:
            current_drift: Current drift magnitude
            current_performance: Current anomaly detection performance
            window_size: Window size for history analysis

        Returns:
            should_decrease: Whether to decrease sparsity
        """
        # Low drift periods may benefit from more features
        low_drift = current_drift < 0.1

        # Performance below expected threshold
        performance_low = current_performance < 0.8

        return low_drift and performance_low

    def get_adaptive_lambda(self,
                            base_lambda: float,
                            current_drift: float) -> float:
        """
        Get adaptive regularization parameter based on drift.

        Args:
            base_lambda: Base regularization parameter
            current_drift: Current drift magnitude

        Returns:
            adaptive_lambda: Adapted regularization parameter
        """
        # Increase sparsity during high drift
        if current_drift > 0.5:
            return base_lambda * 2.0
        elif current_drift > 0.2:
            return base_lambda * 1.5
        elif current_drift < 0.05:
            return base_lambda * 0.5
        else:
            return base_lambda


def create_gating_network(input_dim: int, hidden_dim: int, config: Dict) -> DynamicGatingNetwork:
    """
    Factory function to create a dynamic gating network with sensible defaults.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        config: Configuration dictionary

    Returns:
        gating_network: Initialized dynamic gating network
    """
    default_config = {
        'dropout': 0.1,
        'importance_ema_alpha': 0.9,
        'lambda_l1': 0.0001,
        'lambda_entropy': 0.001,
        'correlation_threshold': 0.3
    }

    # Update with user config
    default_config.update(config)

    return DynamicGatingNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        config=default_config
    )
    gates: Feature
    gates[batch_size, input_dim]


# batch_size = drift_signal.shape[0]
# device = drift_signal.device
#
# # Handle missing previous hidden state
# if prev_hidden is None:
#     prev_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
#
# # Ensure prev_hidden has correct batch size
# if prev_hidden.shape[0] != batch_size:
#     # Take the last layer if prev_hidden is from multilayer GRU
#     if len(prev_hidden.shape) == 3:
#         prev_hidden = prev_hidden[-1]  # [batch_size, hidden_dim]
#
#     # Repeat or truncate to match batch size
#     if prev_hidden.shape[0] == 1 and batch_size > 1:
#         prev_hidden = prev_hidden.repeat(batch_size, 1)
#     elif prev_hidden.shape[0] > batch_size:
#         prev_hidden = prev_hidden[:batch_size]
#
# # Concatenate all inputs for gate computation
# gate_input = torch.cat([prev_hidden, drift_signal, correlations], dim=-1)
#
# # Compute base gates
# gates = self.gate_network(gate_input)
#
# # Modulate gates with gradient-based importance
# if importance_gradients is not None:
#     # Update EMA of importance
#     if self.training:
#         current_importance = importance_gradients.mean(dim=0)
#         self.importance_ema = (self.ema_alpha * self.importance_ema +
#                                (1 - self.ema_alpha) * current_importance)
#
#     # Apply softmax with temperature to importance scores
#     importance_weights = F.softmax(self.importance_ema / self.temperature, dim=-1)
#
#     # Modulate gates
#     gates = gates * importance_weights.unsqueeze(0)
#
# return gates


def compute_sparsity_loss(self, gates: torch.Tensor) -> torch.Tensor:


    """
    Compute sparsity regularization loss for gates.

    Args:
    gates: Feature gates [batch_size, input_dim]

    Returns:
    sparsity_loss: Combined L1 and entropy regularization
        """
    # L1 sparsity penalty
    l1_loss = torch.norm(gates, p=1, dim=-1).mean()

    # Entropy regularization (prevents premature convergence)
    entropy_loss = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean()

    # Combined loss
    lambda1 = self.config.get('lambda_l1', 0.0001)
    lambda2 = self.config.get('lambda_entropy', 0.001)

    sparsity_loss = lambda1 * l1_loss + lambda2 * entropy_loss

    return sparsity_loss


def get_gate_statistics(self, gates: torch.Tensor) -> Dict[str, float]:


    """
    Compute statistics about current gate values.

    Args:
    gates: Feature gates [batch_size, input_dim]

    Returns:
    stats: Dictionary of gate statistics
    """
    gates_mean = gates.mean(dim=0)  # Average over batch

    stats = {
        'mean_gate_value': gates_mean.mean().item(),
        'std_gate_value': gates_mean.std().item(),
        'min_gate_value': gates_mean.min().item(),
        'max_gate_value': gates_mean.max().item(),
        'sparsity_ratio': (gates_mean < 0.1).float().mean().item(),
        'active_features': (gates_mean > 0.5).sum().item(),
        'gate_entropy': -torch.sum(gates_mean * torch.log(gates_mean + 1e-8)).item()
    }

    return stats


class GradientBasedImportance(nn.Module):


    """
    Computes gradient-based feature importance scores.

    This module tracks the gradient magnitudes with respect to input features
    to identify which features most significantly impact the reconstruction loss.
    """


def __init__(self, input_dim: int, ema_alpha: float = 0.9):
    super().__init__()
    self.input_dim = input_dim
    self.ema_alpha = ema_alpha

    # Exponential moving average of gradient magnitudes
    self.register_buffer('gradient_ema', torch.zeros(input_dim))


