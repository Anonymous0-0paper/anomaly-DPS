"""
Drift Detection and Monitoring for Stream-DAD.

This module implements the drift detection mechanisms that compute drift sensitivity
signals and monitor distribution changes in real-time.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitors concept drift by tracking distributional changes in input features.

    Computes drift sensitivity signals that feed into the dynamic gating mechanism
    and provides early warning of distribution shifts.
    """

    def __init__(self, input_dim: int, window_size: int = 50, configs : Dict=None):
        self.input_dim = input_dim
        self.window_size = window_size
        self.configs = configs or {}

        # Rolling statistics for each feature
        self.feature_histories = [deque(maxlen=window_size) for _ in range(input_dim)]
        self.running_means = torch.zeros(input_dim)
        self.running_vars = torch.ones(input_dim)
        self.running_counts = torch.zeros(input_dim)

        # Drift magnitude tracking
        self.drift_history = deque(maxlen=window_size)
        self.current_drift_magnitude = 0.0

        # Adaptive threshold for drift detection
        self.drift_threshold = 0.5
        self.adaptation_rate = 0.1

        # Change point detection
        self.change_points = []
        self.last_change_point = 0
        self.min_change_interval = 100

        # Statistical test parameters
        self.significance_level = 0.05
        self.test_window_size = 30

        # Drift type classification
        self.drift_types = ['gradual', 'sudden', 'recurring']
        self.detected_drift_type = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute drift sensitivity signals for input batch.

        Args:
            x: Input tensor [batch_size, window_size, input_dim]

        Returns:
            drift_signals: Drift sensitivity per feature [batch_size, input_dim]
        """
        batch_size = x.shape[0]
        device = x.device

        # Move running statistics to correct device
        self.running_means = self.running_means.to(device)
        self.running_vars = self.running_vars.to(device)
        self.running_counts = self.running_counts.to(device)

        drift_signals = torch.zeros(batch_size, self.input_dim, device=device)

        for b in range(batch_size):
            sample = x[b]  # [window_size, input_dim]
            drift_signals[b] = self._compute_sample_drift(sample)

        # Update drift magnitude tracking
        current_magnitude = drift_signals.mean(dim=0).norm().item()
        self.current_drift_magnitude = current_magnitude
        self.drift_history.append(current_magnitude)

        return drift_signals

    def _compute_sample_drift(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Compute drift sensitivity for a single sample.

        Args:
            sample: Single sample [window_size, input_dim]

        Returns:
            drift_signal: Drift sensitivity per feature [input_dim]
        """
        window_size, input_dim = sample.shape
        device = sample.device
        drift_signal = torch.zeros(input_dim, device=device)

        # Compute current sample statistics
        sample_means = sample.mean(dim=0)
        sample_vars = sample.var(dim=0, unbiased=False)

        for i in range(input_dim):
            # Update feature history
            self.feature_histories[i].extend(sample[:, i].cpu().numpy())

            # Compute drift using normalized deviation
            if self.running_counts[i] > 0:
                historical_mean = self.running_means[i]
                historical_std = torch.sqrt(self.running_vars[i]) + 1e-8

                # Normalized deviation from historical statistics
                deviation = torch.abs(sample_means[i] - historical_mean) / historical_std
                drift_signal[i] = deviation

            # Update running statistics with exponential moving average
            alpha = self.adaptation_rate
            if self.running_counts[i] == 0:
                self.running_means[i] = sample_means[i]
                self.running_vars[i] = sample_vars[i]
                self.running_counts[i] = window_size
            else:
                self.running_means[i] = (1 - alpha) * self.running_means[i] + alpha * sample_means[i]
                self.running_vars[i] = (1 - alpha) * self.running_vars[i] + alpha * sample_vars[i]
                self.running_counts[i] += window_size

        return drift_signal

    def detect_change_points(self, significance_test: bool = True) -> List[int]:
        """
        Detect change points in the drift signal using statistical tests.

        Args:
            significance_test: Whether to use statistical significance testing

        Returns:
            change_points: List of detected change point timestamps
        """
        if len(self.drift_history) < self.test_window_size * 2:
            return []

        recent_drift = list(self.drift_history)
        change_points = []

        if significance_test:
            # Use Welch's t-test for change point detection
            for i in range(self.test_window_size, len(recent_drift) - self.test_window_size):
                before_window = recent_drift[i-self.test_window_size:i]
                after_window = recent_drift[i:i+self.test_window_size]

                # Compute t-statistic
                t_stat, p_value = self._welch_t_test(before_window, after_window)

                if p_value < self.significance_level:
                    timestamp = len(self.drift_history) - len(recent_drift) + i
                    if timestamp - self.last_change_point > self.min_change_interval:
                        change_points.append(timestamp)
                        self.last_change_point = timestamp
        else:
            # Simple threshold-based detection
            threshold = np.mean(recent_drift) + 2 * np.std(recent_drift)
            for i, magnitude in enumerate(recent_drift):
                if magnitude > threshold:
                    timestamp = len(self.drift_history) - len(recent_drift) + i
                    if timestamp - self.last_change_point > self.min_change_interval:
                        change_points.append(timestamp)
                        self.last_change_point = timestamp

        self.change_points.extend(change_points)
        return change_points

    def _welch_t_test(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """
        Perform Welch's t-test for equal means.

        Args:
            sample1: First sample
            sample2: Second sample

        Returns:
            t_stat: T-statistic
            p_value: P-value (approximate)
        """
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

        # Welch's t-statistic
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        if pooled_se == 0:
            return 0.0, 1.0

        t_stat = (mean1 - mean2) / pooled_se

        # Approximate degrees of freedom (Welch-Satterthwaite equation)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

        # Approximate p-value using t-distribution (simplified)
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))

        return t_stat, p_value

    def _t_cdf(self, t: float, df: float) -> float:
        """Approximate t-distribution CDF."""
        # Simple approximation for t-distribution CDF
        if df > 30:
            # Use normal approximation for large df
            return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-2 * t**2 / np.pi)))
        else:
            # Rough approximation for small df
            return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-t**2 / df)))

    def classify_drift_type(self, lookback_window: int = 100) -> str:
        """
        Classify the type of drift based on recent patterns.

        Args:
            lookback_window: Number of recent samples to analyze

        Returns:
            drift_type: Classified drift type ('gradual', 'sudden', 'recurring')
        """
        if len(self.drift_history) < lookback_window:
            return 'insufficient_data'

        recent_drift = list(self.drift_history)[-lookback_window:]

        # Compute trends and patterns
        trend = np.polyfit(range(len(recent_drift)), recent_drift, 1)[0]
        volatility = np.std(recent_drift)

        # Detect periodicity using autocorrelation
        autocorr = np.correlate(recent_drift, recent_drift, mode='full')
        autocorr = autocorr[autocorr.size // 2:]

        # Find peaks in autocorrelation (excluding lag 0)
        peaks = []
        for i in range(5, min(50, len(autocorr))):
            if (autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and
                    autocorr[i] > 0.3 * autocorr[0]):
                peaks.append(i)

        # Classification logic
        if len(peaks) > 0 and max(autocorr[peaks]) > 0.5 * autocorr[0]:
            drift_type = 'recurring'
        elif abs(trend) > 0.01 and volatility < 0.1:
            drift_type = 'gradual'
        elif volatility > 0.2:
            drift_type = 'sudden'
        else:
            drift_type = 'stable'

        self.detected_drift_type = drift_type
        return drift_type

    def get_current_drift_magnitude(self) -> float:
        """Get current drift magnitude."""
        return self.current_drift_magnitude

    def get_drift_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive drift statistics.

        Returns:
            stats: Dictionary containing drift statistics
        """
        if not self.drift_history:
            return {'error': 'No drift history available'}

        recent_drift = list(self.drift_history)

        stats = {
            'current_magnitude': self.current_drift_magnitude,
            'mean_magnitude': np.mean(recent_drift),
            'std_magnitude': np.std(recent_drift),
            'max_magnitude': np.max(recent_drift),
            'drift_trend': np.polyfit(range(len(recent_drift)), recent_drift, 1)[0] if len(recent_drift) > 1 else 0.0,
            'detected_type': self.detected_drift_type,
            'num_change_points': len(self.change_points),
            'time_since_last_change': len(self.drift_history) - self.last_change_point if self.last_change_point > 0 else -1
        }

        return stats

    def is_high_drift_period(self, threshold_multiplier: float = 1.5) -> bool:
        """
        Determine if currently in a high drift period.

        Args:
            threshold_multiplier: Multiplier for adaptive threshold

        Returns:
            is_high_drift: Whether current period has high drift
        """
        if len(self.drift_history) < 10:
            return False

        recent_drift = list(self.drift_history)
        adaptive_threshold = np.mean(recent_drift) + threshold_multiplier * np.std(recent_drift)

        return self.current_drift_magnitude > adaptive_threshold

    def reset(self) -> None:
        """Reset drift detector state."""
        for history in self.feature_histories:
            history.clear()

        self.running_means.zero_()
        self.running_vars.fill_(1.0)
        self.running_counts.zero_()
        self.drift_history.clear()
        self.change_points.clear()
        self.current_drift_magnitude = 0.0
        self.last_change_point = 0
        self.detected_drift_type = None

        logger.info("Reset drift detector state")

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'running_means': self.running_means.cpu(),
            'running_vars': self.running_vars.cpu(),
            'running_counts': self.running_counts.cpu(),
            'drift_history': list(self.drift_history),
            'change_points': self.change_points,
            'current_drift_magnitude': self.current_drift_magnitude,
            'last_change_point': self.last_change_point,
            'detected_drift_type': self.detected_drift_type,
            'feature_histories': [list(hist) for hist in self.feature_histories]
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.running_means = state.get('running_means', torch.zeros(self.input_dim))
        self.running_vars = state.get('running_vars', torch.ones(self.input_dim))
        self.running_counts = state.get('running_counts', torch.zeros(self.input_dim))

        drift_history = state.get('drift_history', [])
        self.drift_history = deque(drift_history, maxlen=self.window_size)

        self.change_points = state.get('change_points', [])
        self.current_drift_magnitude = state.get('current_drift_magnitude', 0.0)
        self.last_change_point = state.get('last_change_point', 0)
        self.detected_drift_type = state.get('detected_drift_type', None)

        # Restore feature histories
        feature_histories = state.get('feature_histories', [[] for _ in range(self.input_dim)])
        self.feature_histories = [deque(hist, maxlen=self.window_size) for hist in feature_histories]

        logger.info("Loaded drift detector state")


class MultiScaleDriftDetector:
    """
    Multi-scale drift detector that operates at different temporal scales
    to capture both short-term fluctuations and long-term trends.
    """

    def __init__(self, input_dim: int, scales: List[int] = None, config: Dict = None):
        self.input_dim = input_dim
        self.scales = scales or [10, 50, 200]  # Different window sizes
        self.config = config or {}

        # Create detector for each scale
        self.detectors = {}
        for scale in self.scales:
            detector_config = self.config.copy()
            detector_config['drift_threshold'] = detector_config.get('drift_threshold', 0.5) / np.sqrt(scale)
            self.detectors[scale] = DriftDetector(input_dim, scale, detector_config)

        # Aggregation weights for different scales
        self.scale_weights = self._compute_scale_weights()

    def _compute_scale_weights(self) -> Dict[int, float]:
        """Compute weights for aggregating across scales."""
        # Inverse relationship: smaller scales get higher weights for responsiveness
        total_weight = sum(1.0 / scale for scale in self.scales)
        return {scale: (1.0 / scale) / total_weight for scale in self.scales}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale drift signals.

        Args:
            x: Input tensor [batch_size, window_size, input_dim]

        Returns:
            drift_signals: Aggregated drift sensitivity [batch_size, input_dim]
        """
        batch_size = x.shape[0]
        device = x.device

        aggregated_drift = torch.zeros(batch_size, self.input_dim, device=device)

        # Compute drift at each scale
        for scale, detector in self.detectors.items():
            scale_drift = detector(x)
            weight = self.scale_weights[scale]
            aggregated_drift += weight * scale_drift

        return aggregated_drift

    def get_scale_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get drift statistics for each scale."""
        return {scale: detector.get_drift_statistics()
                for scale, detector in self.detectors.items()}

    def detect_multi_scale_changes(self) -> Dict[int, List[int]]:
        """Detect change points at each scale."""
        return {scale: detector.detect_change_points()
                for scale, detector in self.detectors.items()}

    def reset(self) -> None:
        """Reset all scale detectors."""
        for detector in self.detectors.values():
            detector.reset()

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'scales': self.scales,
            'scale_weights': self.scale_weights,
            'detector_states': {scale: detector.get_state()
                                for scale, detector in self.detectors.items()}
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.scales = state.get('scales', self.scales)
        self.scale_weights = state.get('scale_weights', self.scale_weights)

        detector_states = state.get('detector_states', {})
        for scale, detector_state in detector_states.items():
            if int(scale) in self.detectors:
                self.detectors[int(scale)].load_state(detector_state)


def create_drift_detector(input_dim: int, config: Dict) -> DriftDetector:
    """
    Factory function to create drift detector with sensible defaults.

    Args:
        input_dim: Number of input features
        config: Configuration dictionary

    Returns:
        detector: Initialized drift detector
    """
    default_config = {
        'window_size': 50,
        'drift_threshold': 0.5,
        'adaptation_rate': 0.1,
        'min_change_interval': 100,
        'significance_level': 0.05,
        'test_window_size': 30
    }

    # Update with user config
    default_config.update(config)

    return DriftDetector(
        input_dim=input_dim,
        window_size=default_config['window_size'],
        configs=default_config
    )