import numpy as np
import torch

class TimeSeriesFeatureEngineer:
    """
    Feature engineering for time series data.
    Supports: statistical, lag, delta, and frequency features.
    """
    def __init__(self, lags=[1, 2, 3], window=5, add_time_features=False, sampling_rate=1):
        self.lags = lags
        self.window = window
        self.add_time_features = add_time_features
        self.sampling_rate = sampling_rate

    def transform(self, x, timestamps=None):
        """
        x: np.ndarray or torch.Tensor, shape (seq_len, n_features) or (batch, seq_len, n_features)
        timestamps: np.ndarray or None, shape (seq_len,) or (batch, seq_len)
        Returns: augmented features, same type as input
        """
        is_tensor = torch.is_tensor(x)
        if is_tensor:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        # Handle batch or single sequence
        if x_np.ndim == 2:
            x_np = x_np[None, ...]
        batch, seq_len, n_features = x_np.shape
        features = [x_np]
        # Statistical features (rolling window)
        for stat_func in [np.mean, np.std, np.min, np.max]:
            stat_feat = np.zeros_like(x_np)
            for b in range(batch):
                for f in range(n_features):
                    stat_feat[b, :, f] = np.convolve(x_np[b, :, f], np.ones(self.window)/self.window, mode='same')
            features.append(stat_feat)
        # Lag features
        for lag in self.lags:
            lagged = np.zeros_like(x_np)
            lagged[:, lag:, :] = x_np[:, :-lag, :]
            features.append(lagged)
        # Delta features
        delta = np.diff(x_np, axis=1, prepend=x_np[:, :1, :])
        features.append(delta)
        # Frequency features (first few FFT coefficients)
        fft = np.fft.fft(x_np, axis=1)
        fft_real = np.real(fft[:, :, :3])  # Take first 3 coefficients
        fft_imag = np.imag(fft[:, :, :3])
        features.append(fft_real)
        features.append(fft_imag)
        # Time features (if timestamps provided)
        if self.add_time_features and timestamps is not None:
            if is_tensor:
                timestamps = timestamps.detach().cpu().numpy()
            if timestamps.ndim == 1:
                timestamps = timestamps[None, ...]
            # Example: hour of day, day of week (assuming unix timestamp)
            hour = ((timestamps // 3600) % 24) / 23.0
            day = ((timestamps // (3600*24)) % 7) / 6.0
            features.append(hour[..., None])
            features.append(day[..., None])
        # Concatenate all features
        out = np.concatenate(features, axis=-1)
        if is_tensor:
            out = torch.from_numpy(out).to(x.device).type_as(x)
        if out.shape[0] == 1:
            out = out[0]
        return out

    @staticmethod
    def get_output_dim(input_dim, lags=[1,2,3], add_time_features=False):
        # input_dim: original number of features
        stat_feats = 4 * input_dim  # mean, std, min, max
        lag_feats = len(lags) * input_dim
        delta_feats = input_dim
        fft_feats = 6  # 3 real, 3 imag (per feature)
        freq_feats = fft_feats * input_dim
        time_feats = 0
        if add_time_features:
            time_feats = 2  # hour and day
        return input_dim + stat_feats + lag_feats + delta_feats + freq_feats + time_feats 