import numpy as np
import torch
from torch.utils.data import Dataset

class AnomalyDetectionDatasets(Dataset):
    """Robust dataset class supporting supervised and unsupervised training modes"""

    def __init__(self, features, labels, window_size=10, stride=5, num_classes=2, training_mode='supervised'):
        """
        Parameters:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        window_size: Time window size
        stride: Sliding step size
        num_classes: Number of classes
        training_mode: 'supervised' - supervised mode, 'unsupervised' - unsupervised mode
        """
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.num_classes = num_classes
        self.training_mode = training_mode
        self.num_features = features.shape[1] if len(features.shape) > 1 else 1

        # Validate data_loader
        if len(features) == 0:
            raise ValueError("Feature data_loader is empty")
        if len(labels) == 0:
            raise ValueError("Label data_loader is empty")
        if window_size <= 0:
            raise ValueError(f"Invalid window size: {window_size}")
        if stride <= 0:
            raise ValueError(f"Invalid stride: {stride}")

        # Special handling for unsupervised mode
        if training_mode == 'unsupervised':
            # Use only normal samples
            normal_indices = np.where(labels == 0)[0]
            if len(normal_indices) == 0:
                raise ValueError("Unsupervised mode requires normal samples")

            self.features = features[normal_indices]
            self.labels = np.zeros(len(normal_indices), dtype=int)
            print(f"Unsupervised mode: Using {len(normal_indices)} normal samples")
            features = self.features
            labels = self.labels

        # Create sample sequences
        self.samples = []
        for i in range(0, len(features) - window_size + 1, stride):
            self.samples.append(i)

        # Handle no samples case
        if len(self.samples) == 0:
            print(f"Warning: No samples created (data_loader length={len(features)}, window size={window_size}, stride={stride})")

            # Case 1: Data length is less than window size
            if len(features) < window_size:
                # Calculate padding needed
                padding_needed = window_size - len(features)

                # Correctly pad feature data_loader (2D array)
                if len(features.shape) == 1:
                    # If it's 1D data_loader, convert to 2D first
                    features = features.reshape(-1, 1)
                self.features = np.pad(
                    features,
                    ((0, padding_needed), (0, 0)),  # Pad in sample dimension
                    mode='constant'
                )

                # Pad label data_loader (1D array)
                self.labels = np.pad(
                    labels,
                    (0, padding_needed),  # Pad at the end
                    mode='constant'
                )
                self.samples = [0]  # Only one sample
                print(f"Padded data_loader: Features shape={self.features.shape}, Labels length={len(self.labels)}")

            # Case 2: Data length is sufficient but stride is too large
            else:
                # Use the last possible window
                self.samples = [max(0, len(features) - window_size)]
                print(f"Using last window: Start position={self.samples[0]}")

        # Precompute class distribution (safe way)
        self.class_counts = self._precompute_class_counts()

    def _precompute_class_counts(self):
        """Safely precompute class distribution"""
        counts = np.zeros(self.num_classes, dtype=int)

        # Handle no samples case
        if len(self.samples) == 0:
            return counts

        # Use reliable loop method
        for i in self.samples:
            start = max(0, min(i, len(self.labels) - 1))
            end = min(start + self.window_size, len(self.labels))
            seq_labels = self.labels[start:end]

            if self.num_classes == 2:
                label = 1 if np.any(seq_labels == 1) else 0
                counts[label] += 1
            else:
                label_counts = np.bincount(seq_labels, minlength=self.num_classes)
                if len(label_counts) > 0:  # Ensure there is data_loader
                    label = np.argmax(label_counts)
                    counts[label] += 1

        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        end = start + self.window_size

        # Handle boundary cases
        if end > len(self.features):
            # Need padding
            padding = end - len(self.features)

            # Get existing data_loader
            seq_features = self.features[start:]
            seq_labels = self.labels[start:]

            # Correctly pad feature data_loader (2D array)
            if len(seq_features) == 0:
                # If completely out of range, create all-zero data_loader
                seq_features = np.zeros((self.window_size, self.num_features))
                seq_labels = np.zeros(self.window_size)
            else:
                # Pad in sample dimension
                seq_features = np.pad(
                    seq_features,
                    ((0, padding), (0, 0)),  # Pad in sample dimension
                    mode='constant'
                )
                # Pad label data_loader (1D array)
                seq_labels = np.pad(
                    seq_labels,
                    (0, padding),  # Pad at the end
                    mode='constant'
                )
        else:
            seq_features = self.features[start:end]
            seq_labels = self.labels[start:end]

        # Determine the main label for the window
        if self.num_classes == 2:
            label = 1 if np.any(seq_labels == 1) else 0
        else:
            label_counts = np.bincount(seq_labels, minlength=self.num_classes)
            label = np.argmax(label_counts) if len(label_counts) > 0 else 0

        # Return format
        item = {
            'x': torch.tensor(seq_features, dtype=torch.float32),
            'y': torch.tensor(label, dtype=torch.long)
        }

        # Add reconstruction target for unsupervised mode
        if self.training_mode == 'unsupervised':
            item['target'] = torch.tensor(seq_features, dtype=torch.float32)

        return item
