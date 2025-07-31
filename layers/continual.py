"""
Continual Learning Components for Stream-DAD.

This module implements the continual learning mechanisms including Elastic Weight
Consolidation (EWC), drift buffer management, and adaptation strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class FisherInformationComputer:
    """
    Computes and maintains Fisher Information Matrix for EWC regularization.

    Uses diagonal approximation for computational efficiency while maintaining
    the essential properties needed for continual learning.
    """

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.fisher_dict = {}
        self.optimal_params = {}
        self.update_frequency = config.get('fisher_update_freq', 100)
        self.decay_factor = config.get('fisher_decay', 0.95)
        self.sparse_threshold = config.get('fisher_sparse_threshold', 1e-6)

    def compute_fisher_information(self, data_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix using diagonal approximation.

        Args:
            data_batch: Batch of data for Fisher computation [batch_size, window_size, input_dim]

        Returns:
            fisher_dict: Dictionary mapping parameter names to Fisher information
        """
        self.model.eval()
        fisher_dict = {}

        # Store original requires_grad states
        original_grad_states = {}
        for name, param in self.model.named_parameters():
            original_grad_states[name] = param.requires_grad
            param.requires_grad = True

        try:
            total_samples = 0

            # Process batch
            for sample in data_batch:
                sample = sample.unsqueeze(0)  # Add batch dimension
                sample.requires_grad_(True)

                # Forward pass
                output = self.model(sample, return_gates=True)

                # Compute negative log-likelihood (reconstruction loss)
                reconstructed = output['reconstructed']
                loss = F.mse_loss(reconstructed, sample)

                # Compute gradients
                gradients = torch.autograd.grad(
                    outputs=loss,
                    inputs=list(self.model.parameters()),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                )

                # Accumulate squared gradients
                for (name, param), grad in zip(self.model.named_parameters(), gradients):
                    if grad is not None:
                        if name not in fisher_dict:
                            fisher_dict[name] = torch.zeros_like(param)
                        fisher_dict[name] += grad.pow(2)

                total_samples += 1

            # Average over samples
            for name in fisher_dict:
                fisher_dict[name] /= total_samples

                # Apply sparsity threshold
                if self.sparse_threshold > 0:
                    mask = fisher_dict[name] > self.sparse_threshold
                    fisher_dict[name] = fisher_dict[name] * mask.float()

        finally:
            # Restore original requires_grad states
            for name, param in self.model.named_parameters():
                param.requires_grad = original_grad_states[name]

            self.model.train()

        return fisher_dict

    def update_fisher(self, data_batch: torch.Tensor, incremental: bool = True) -> None:
        """
        Update Fisher information with new data.

        Args:
            data_batch: New data for Fisher computation
            incremental: Whether to update incrementally or replace
        """
        new_fisher = self.compute_fisher_information(data_batch)

        if incremental and self.fisher_dict:
            # Incremental update with decay
            for name in new_fisher:
                if name in self.fisher_dict:
                    self.fisher_dict[name] = (self.decay_factor * self.fisher_dict[name] +
                                              (1 - self.decay_factor) * new_fisher[name])
                else:
                    self.fisher_dict[name] = new_fisher[name]
        else:
            # Full replacement
            self.fisher_dict = new_fisher

        logger.debug(f"Updated Fisher information for {len(self.fisher_dict)} parameters")

    def store_optimal_parameters(self) -> None:
        """Store current parameters as optimal for EWC."""
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

    def get_fisher_for_param(self, param_name: str) -> torch.Tensor:
        """Get Fisher information for a specific parameter."""
        return self.fisher_dict.get(param_name, torch.zeros_like(
            dict(self.model.named_parameters())[param_name]))

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'fisher_dict': self.fisher_dict,
            'optimal_params': self.optimal_params
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.fisher_dict = state.get('fisher_dict', {})
        self.optimal_params = state.get('optimal_params', {})


class DriftBuffer:
    """
    Memory-efficient buffer for storing drift-significant samples.

    Maintains a diverse set of samples from high-drift periods for replay
    during continual learning updates.
    """

    def __init__(self, capacity: int, diversity_threshold: float = 0.1):
        self.capacity = capacity
        self.diversity_threshold = diversity_threshold
        self.buffer = []
        self.scores = []
        self.timestamps = []

    def add_sample(self, sample: torch.Tensor, drift_score: float, timestamp: int) -> None:
        """
        Add a sample to the drift buffer.

        Args:
            sample: Data sample [window_size, input_dim]
            drift_score: Drift significance score
            timestamp: Time step when sample was observed
        """
        sample_np = sample.detach().cpu().numpy()

        if len(self.buffer) < self.capacity:
            # Buffer not full, add directly if diverse enough
            if self._is_diverse_enough(sample_np):
                self.buffer.append(sample_np)
                self.scores.append(drift_score)
                self.timestamps.append(timestamp)
        else:
            # Buffer full, replace if score is higher and sample is diverse
            min_score_idx = np.argmin(self.scores)

            if drift_score > self.scores[min_score_idx] and self._is_diverse_enough(sample_np):
                self.buffer[min_score_idx] = sample_np
                self.scores[min_score_idx] = drift_score
                self.timestamps[min_score_idx] = timestamp

    def _is_diverse_enough(self, new_sample: np.ndarray) -> bool:
        """
        Check if new sample adds sufficient diversity to the buffer.

        Args:
            new_sample: Candidate sample

        Returns:
            is_diverse: Whether sample is diverse enough
        """
        if len(self.buffer) == 0:
            return True

        # Compute minimum distance to existing samples
        distances = []
        for existing_sample in self.buffer:
            distance = np.linalg.norm(new_sample.flatten() - existing_sample.flatten())
            distances.append(distance)

        min_distance = min(distances)
        return min_distance > self.diversity_threshold

    def sample_batch(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """
        Sample a batch from the buffer with importance-based probability.

        Args:
            batch_size: Number of samples to return
            device: Device to place tensors on

        Returns:
            batch: List of sampled tensors
        """
        if len(self.buffer) == 0:
            return []

        # Compute sampling probabilities proportional to drift scores
        scores_array = np.array(self.scores)
        probabilities = scores_array / scores_array.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probabilities,
            replace=False
        )

        # Return sampled tensors
        batch = []
        for idx in indices:
            sample_tensor = torch.from_numpy(self.buffer[idx]).float().to(device)
            batch.append(sample_tensor)

        return batch

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {'size': 0, 'avg_score': 0.0, 'diversity': 0.0}

        # Compute average pairwise distance (diversity measure)
        total_distance = 0.0
        count = 0

        for i in range(len(self.buffer)):
            for j in range(i + 1, len(self.buffer)):
                distance = np.linalg.norm(
                    self.buffer[i].flatten() - self.buffer[j].flatten()
                )
                total_distance += distance
                count += 1

        avg_diversity = total_distance / count if count > 0 else 0.0

        return {
            'size': len(self.buffer),
            'avg_score': np.mean(self.scores),
            'diversity': avg_diversity,
            'capacity_utilization': len(self.buffer) / self.capacity
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.scores.clear()
        self.timestamps.clear()


class ContinualLearner:
    """
    Main continual learning controller that orchestrates EWC, drift adaptation,
    and memory management for Stream-DAD.
    """

    def __init__(self, model: nn.Module, configs):
        self.model = model
        self.configs = configs

        # Fisher information computer
        self.fisher_computer = FisherInformationComputer(model, {})

        # Drift buffer
        self.drift_buffer = DriftBuffer(
            capacity= getattr(configs, 'capacity', 1000),
            diversity_threshold=0.1
        )

        # EWC parameters
        self.ewc_lambda = 0.01
        self.ewc_alpha = 0.9  # For adaptive EWC weight

        # Update counters
        self.update_count = 0
        self.fisher_update_interval = 100

        # Performance tracking for adaptation
        self.performance_history = deque(maxlen=100)

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Returns:
            ewc_loss: Elastic weight consolidation loss
        """
        if not self.fisher_computer.fisher_dict or not self.fisher_computer.optimal_params:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        ewc_loss = 0.0
        device = next(self.model.parameters()).device

        for name, param in self.model.named_parameters():
            if name in self.fisher_computer.fisher_dict and name in self.fisher_computer.optimal_params:
                fisher_info = self.fisher_computer.fisher_dict[name].to(device)
                optimal_param = self.fisher_computer.optimal_params[name].to(device)

                # EWC penalty: F_i * (θ_i - θ*_i)^2
                param_diff = param - optimal_param
                ewc_loss += torch.sum(fisher_info * param_diff.pow(2))

        return ewc_loss

    def update_fisher_information(self, data_batch: torch.Tensor) -> None:
        """
        Update Fisher information with new data batch.

        Args:
            data_batch: Recent data for Fisher computation
        """
        self.fisher_computer.update_fisher(data_batch, incremental=True)
        self.fisher_computer.store_optimal_parameters()

        logger.debug("Updated Fisher information and stored optimal parameters")

    def should_update_fisher(self) -> bool:
        """Determine if Fisher information should be updated."""
        return (self.update_count % self.fisher_update_interval == 0 and
                self.update_count > 0)

    def adapt_to_drift(self,
                       drift_magnitude: float,
                       recent_data: torch.Tensor,
                       current_performance: float) -> Dict[str, float]:
        """
        Adapt learning parameters based on detected drift.

        Args:
            drift_magnitude: Current drift magnitude
            recent_data: Recent data samples
            current_performance: Current anomaly detection performance

        Returns:
            adaptation_info: Dictionary with adaptation details
        """
        # Update performance history
        self.performance_history.append(current_performance)

        # Adaptive EWC weight based on drift magnitude and performance
        performance_trend = self._compute_performance_trend()
        adapted_ewc_lambda = self._adapt_ewc_weight(drift_magnitude, performance_trend)

        # Update Fisher information if needed
        if self.should_update_fisher():
            self.update_fisher_information(recent_data)

        # Determine if buffer update is needed
        drift_threshold = self.config.get('drift_threshold', 0.3)
        if drift_magnitude > drift_threshold:
            self._update_drift_buffer(recent_data, drift_magnitude)

        adaptation_info = {
            'original_ewc_lambda': self.ewc_lambda,
            'adapted_ewc_lambda': adapted_ewc_lambda,
            'drift_magnitude': drift_magnitude,
            'performance_trend': performance_trend,
            'buffer_size': len(self.drift_buffer.buffer)
        }

        # Update model's EWC lambda
        if hasattr(self.model, 'config'):
            self.model.config['lambda_ewc'] = adapted_ewc_lambda

        return adaptation_info

    def _compute_performance_trend(self, window_size: int = 10) -> float:
        """
        Compute recent performance trend.

        Args:
            window_size: Window size for trend computation

        Returns:
            trend: Performance trend (-1 to 1, negative means declining)
        """
        if len(self.performance_history) < window_size:
            return 0.0

        recent_performance = list(self.performance_history)[-window_size:]

        # Simple linear trend
        x = np.arange(len(recent_performance))
        y = np.array(recent_performance)

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            # Normalize slope to [-1, 1] range
            trend = np.tanh(slope * 10)  # Scale factor for sensitivity
        else:
            trend = 0.0

        return trend

    def _adapt_ewc_weight(self, drift_magnitude: float, performance_trend: float) -> float:
        """
        Adapt EWC weight based on drift and performance.

        Args:
            drift_magnitude: Current drift magnitude
            performance_trend: Recent performance trend

        Returns:
            adapted_weight: Adapted EWC regularization weight
        """
        base_lambda = self.ewc_lambda

        # Decrease EWC weight during high drift for faster adaptation
        drift_factor = np.exp(-drift_magnitude * 2)

        # Increase EWC weight if performance is declining
        performance_factor = 1.0
        if performance_trend < -0.1:  # Performance declining
            performance_factor = 1.5
        elif performance_trend > 0.1:  # Performance improving
            performance_factor = 0.8

        adapted_weight = base_lambda * drift_factor * performance_factor

        # Clamp to reasonable range
        adapted_weight = np.clip(adapted_weight, 0.001, 0.1)

        return adapted_weight

    def _update_drift_buffer(self, data: torch.Tensor, drift_score: float) -> None:
        """
        Update drift buffer with high-drift samples.

        Args:
            data: Recent data samples
            drift_score: Drift significance score
        """
        # Add samples to buffer (process each sample in the batch)
        for i, sample in enumerate(data):
            timestamp = self.update_count * data.shape[0] + i
            self.drift_buffer.add_sample(sample, drift_score, timestamp)

    def get_replay_batch(self, batch_size: int) -> List[torch.Tensor]:
        """
        Get a batch of samples from drift buffer for replay.

        Args:
            batch_size: Desired batch size

        Returns:
            replay_batch: List of replay samples
        """
        device = next(self.model.parameters()).device
        return self.drift_buffer.sample_batch(batch_size, device)

    def compute_replay_loss(self, replay_batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss on replay samples to prevent forgetting.

        Args:
            replay_batch: Batch of replay samples

        Returns:
            replay_loss: Loss on replay samples
        """
        if not replay_batch:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        total_loss = 0.0
        count = 0

        for sample in replay_batch:
            sample = sample.unsqueeze(0)  # Add batch dimension
            output = self.model(sample)
            loss = F.mse_loss(output['reconstructed'], sample)
            total_loss += loss
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0)

    def step(self) -> None:
        """Update internal counters and states."""
        self.update_count += 1

    def reset(self) -> None:
        """Reset continual learning state."""
        self.fisher_computer.fisher_dict.clear()
        self.fisher_computer.optimal_params.clear()
        self.drift_buffer.clear()
        self.performance_history.clear()
        self.update_count = 0

        logger.info("Reset continual learning state")

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'fisher_state': self.fisher_computer.get_state(),
            'buffer_state': {
                'buffer': self.drift_buffer.buffer,
                'scores': self.drift_buffer.scores,
                'timestamps': self.drift_buffer.timestamps
            },
            'update_count': self.update_count,
            'performance_history': list(self.performance_history)
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        # Load Fisher information
        if 'fisher_state' in state:
            self.fisher_computer.load_state(state['fisher_state'])

        # Load buffer state
        if 'buffer_state' in state:
            buffer_state = state['buffer_state']
            self.drift_buffer.buffer = buffer_state.get('buffer', [])
            self.drift_buffer.scores = buffer_state.get('scores', [])
            self.drift_buffer.timestamps = buffer_state.get('timestamps', [])

        # Load counters and history
        self.update_count = state.get('update_count', 0)
        performance_history = state.get('performance_history', [])
        self.performance_history = deque(performance_history, maxlen=self.config.get('history_length', 100))

        logger.info("Loaded continual learning state")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about continual learning state."""
        stats = {
            'fisher_parameters': len(self.fisher_computer.fisher_dict),
            'buffer_stats': self.drift_buffer.get_statistics(),
            'update_count': self.update_count,
            'avg_recent_performance': np.mean(list(self.performance_history)) if self.performance_history else 0.0,
            'performance_trend': self._compute_performance_trend()
        }

        return stats